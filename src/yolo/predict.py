"""Run YOLO segmentation inference and write instance prediction artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tifffile import TiffFile

from common.instance_predictions import (
    instance_map_from_masks,
    instance_map_path,
    save_yolo_mask_npz,
    ultralytics_result_prediction_arrays,
    write_instance_map_tiff,
    yolo_mask_npz_path,
)
from common.manifest_io import (
    collect_manifest_image_paths,
    default_patch_manifest_path,
    load_dataset_manifest,
)
from yolo.config import default_scratch_root, variant_choices
from yolo.dataset_yaml import load_yaml_dataset_config, resolve_split_dir
from yolo.pipeline import resolve_variant_paths
from yolo.train import _parse_device

_PATCH_IMAGE_SUFFIXES = {".tif", ".tiff"}


def load_image_for_yolo(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix not in {".tif", ".tiff"}:
        raise ValueError(f"Expected .tif / .tiff image, got {path.suffix!r} for {path}")
    with TiffFile(path) as tif:
        series = tif.series[0]
        image = series.asarray()
        axes = series.axes

    if image.ndim == 2:
        return np.expand_dims(image.astype(np.uint8, copy=False), axis=-1)

    if axes == "CYX":
        image = np.transpose(image, (1, 2, 0))
    elif axes == "YXC":
        pass
    elif image.ndim == 3 and image.shape[0] < min(image.shape[1], image.shape[2]):
        image = np.transpose(image, (1, 2, 0))

    return np.clip(image, 0, 255).astype(np.uint8, copy=False)


def collect_yolo_patch_image_paths(
    dataset_root: Path, config: dict[str, Any]
) -> list[Path]:
    for split_name in ("test", "val"):
        rel = config.get(split_name)
        if not rel:
            continue
        if split_name == "test" and config.get("val"):
            print(
                "YOLO patch predict: using the `test` split only. "
                "The dataset YAML also defines `val`, which is ignored.",
                file=sys.stderr,
            )
        image_dir = resolve_split_dir(dataset_root, str(rel))
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Missing image directory for split {split_name!r}: {image_dir}"
            )
        image_paths = [
            image_path
            for image_path in sorted(image_dir.iterdir())
            if image_path.suffix.lower() in _PATCH_IMAGE_SUFFIXES
        ]
        if not image_paths:
            raise ValueError(f"No patch images found under {image_dir}")
        return image_paths
    raise ValueError("Dataset YAML must define a `test` or `val` split")


def _load_whole_image_paths(args: argparse.Namespace) -> list[Path]:
    if args.manifest is not None:
        return [image_path for image_path, _ in collect_manifest_image_paths(args.manifest)]
    if args.image is None:
        raise ValueError("whole mode requires --manifest or --image")
    return [args.image.resolve()]


class _NumpyPredictionResult:
    def __init__(self, image: np.ndarray, object_prediction_list: list[Any]) -> None:
        self.image = image
        self.object_prediction_list = object_prediction_list


def _perform_ultralytics_inference_preserve_channels(
    detection_model: Any, image: np.ndarray
) -> None:
    """Run Ultralytics on a numpy image slice without BGR channel reordering.

    Mirrors SAHI's Ultralytics backend internals (_original_predictions, mask
    tensors) so multi-channel TIFFs stay channel-ordered. Fragile across
    ultralytics/sahi upgrades; re-check when bumping those dependencies.
    """
    import torch
    from ultralytics.engine.results import Masks

    kwargs = {
        "cfg": detection_model.config_path,
        "verbose": False,
        "conf": detection_model.confidence_threshold,
        "device": detection_model.device,
    }
    if detection_model.image_size is not None:
        kwargs = {"imgsz": detection_model.image_size, **kwargs}

    prediction_result = detection_model.model(np.ascontiguousarray(image), **kwargs)
    if detection_model.has_mask:
        if not prediction_result[0].masks:
            device = getattr(detection_model.model, "device", "cpu")
            prediction_result[0].masks = Masks(
                torch.tensor([], device=device), prediction_result[0].boxes.orig_shape
            )
        prediction_result = [
            (result.boxes.data, result.masks.data) for result in prediction_result
        ]
    else:
        prediction_result = [result.boxes.data for result in prediction_result]

    detection_model._original_predictions = prediction_result
    detection_model._original_shape = image.shape


def _get_sliced_prediction_preserve_channels(
    image: np.ndarray,
    detection_model: Any,
    *,
    slice_height: int,
    slice_width: int,
    overlap_height_ratio: float,
    overlap_width_ratio: float,
) -> _NumpyPredictionResult:
    from sahi.predict import POSTPROCESS_NAME_TO_CLASS, filter_predictions
    from sahi.slicing import get_slice_bboxes

    height, width = image.shape[:2]
    slice_bboxes = get_slice_bboxes(
        image_height=height,
        image_width=width,
        auto_slice_resolution=False,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )
    postprocess = POSTPROCESS_NAME_TO_CLASS["GREEDYNMM"](
        match_threshold=0.5,
        match_metric="IOS",
        class_agnostic=False,
    )

    object_prediction_list: list[Any] = []
    for tlx, tly, brx, bry in slice_bboxes:
        image_slice = image[tly:bry, tlx:brx]
        _perform_ultralytics_inference_preserve_channels(detection_model, image_slice)
        detection_model.convert_original_predictions(
            shift_amount=[tlx, tly],
            full_shape=[height, width],
        )
        predictions = filter_predictions(
            detection_model.object_prediction_list,
            exclude_classes_by_name=None,
            exclude_classes_by_id=None,
        )
        for object_prediction in predictions:
            if object_prediction:
                object_prediction_list.append(
                    object_prediction.get_shifted_object_prediction()
                )

    if len(object_prediction_list) > 1:
        object_prediction_list = postprocess(object_prediction_list)
    return _NumpyPredictionResult(
        image=image, object_prediction_list=object_prediction_list
    )


def sahi_predictions_to_mask_arrays(
    predictions: list[Any], height: int, width: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from common.instance_predictions import resize_mask_nearest

    masks: list[np.ndarray] = []
    scores: list[float] = []
    classes: list[int] = []
    for pred in predictions:
        if pred.mask is None:
            continue
        mask = pred.mask.bool_mask
        if mask is None:
            continue
        if mask.shape != (height, width):
            mask = resize_mask_nearest(mask, height, width)
        masks.append(np.asarray(mask, dtype=np.float32))
        scores.append(float(pred.score.value))
        classes.append(int(pred.category.id))
    if not masks:
        return (
            np.zeros((0, height, width), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.stack(masks, axis=0),
        np.asarray(scores, dtype=np.float32),
        np.asarray(classes, dtype=np.float32),
    )


def _write_prediction_artifacts(
    *,
    output_dir: Path,
    sample_id: str,
    image_path: Path,
    height: int,
    width: int,
    instance_map: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    orig_shape: tuple[int, ...],
) -> None:
    inst_path = instance_map_path(output_dir, sample_id)
    write_instance_map_tiff(inst_path, instance_map)
    npz_path = yolo_mask_npz_path(output_dir, sample_id)
    save_yolo_mask_npz(
        npz_path,
        masks=masks,
        scores=scores,
        classes=classes,
        orig_shape=np.array(orig_shape),
        image_path=str(image_path),
    )
    n_inst = int(np.sum(np.unique(instance_map) != 0))
    print(f"Wrote {inst_path} and {npz_path} ({n_inst} instances)")


def device_for_sahi(device: int | str | list[int]) -> str:
    if device == "cpu" or device == -1:
        return "cpu"
    if isinstance(device, list):
        if not device:
            return "cpu"
        return f"cuda:{device[0]}"
    if isinstance(device, int):
        if device < 0:
            return "cpu"
        return f"cuda:{device}"
    if isinstance(device, str):
        if device == "cpu":
            return "cpu"
        if "," in device:
            first = device.split(",")[0].strip()
            return f"cuda:{first}" if first.lstrip("-").isdigit() else device
        if device.lstrip("-").isdigit():
            return f"cuda:{device}"
        return device
    return str(device)


def _resolve_data_yaml(args: argparse.Namespace) -> Path | None:
    if args.data is not None:
        path = args.data.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {path}")
        return path
    if args.variant is None:
        return None
    resolved = resolve_variant_paths(variant_name=args.variant)
    if not resolved.data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {resolved.data_yaml}")
    return resolved.data_yaml


def _resolve_patch_manifest(args: argparse.Namespace) -> Path | None:
    if args.manifest is not None:
        return args.manifest.resolve()
    if args.variant is None:
        return None
    candidate = default_patch_manifest_path(
        default_scratch_root(), "test", args.variant
    )
    return candidate if candidate.is_file() else None


def _run_patch_predict_from_manifest(
    args: argparse.Namespace, manifest_path: Path
) -> None:
    from ultralytics import YOLO

    doc = load_dataset_manifest(manifest_path)
    if args.variant is not None and args.variant != doc.variant:
        raise ValueError(
            f"--variant {args.variant!r} does not match manifest {doc.variant!r}"
        )
    pairs = collect_manifest_image_paths(manifest_path)
    n_samples = len(pairs)
    device = _parse_device(args.device)
    model = YOLO(str(Path(args.weights).resolve()))
    print(
        f"YOLO patch predict: {n_samples} sample(s) from manifest {manifest_path}",
        flush=True,
    )

    for idx, (image_path, sample_id) in enumerate(pairs):
        print(f"Predicting {sample_id} ({idx + 1}/{n_samples})...", flush=True)
        image = load_image_for_yolo(image_path)
        h, w = int(image.shape[0]), int(image.shape[1])
        results = model.predict(
            source=np.ascontiguousarray(image),
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            verbose=False,
            retina_masks=True,
        )
        result = results[0]
        instance_map, masks, scores, classes = ultralytics_result_prediction_arrays(
            result, height=h, width=w
        )
        _write_prediction_artifacts(
            output_dir=args.output_dir,
            sample_id=sample_id,
            image_path=image_path,
            height=h,
            width=w,
            instance_map=instance_map,
            masks=masks,
            scores=scores,
            classes=classes,
            orig_shape=tuple(result.orig_shape),
        )


def run_patch_predict(args: argparse.Namespace, data_yaml: Path | None) -> None:
    manifest_path = _resolve_patch_manifest(args)
    if manifest_path is not None:
        print(f"YOLO patch predict: using manifest {manifest_path}", file=sys.stderr)
        _run_patch_predict_from_manifest(args, manifest_path)
        return

    if data_yaml is None:
        raise ValueError("patch mode requires --manifest, --variant, or --data")

    from ultralytics import YOLO

    dataset_root, config = load_yaml_dataset_config(data_yaml)
    image_paths = collect_yolo_patch_image_paths(dataset_root, config)
    n_samples = len(image_paths)

    device = _parse_device(args.device)
    model = YOLO(str(Path(args.weights).resolve()))
    print(f"YOLO patch predict: {n_samples} image(s) from {data_yaml}", flush=True)

    for idx, image_path in enumerate(image_paths):
        print(f"Predicting {image_path.stem} ({idx + 1}/{n_samples})...", flush=True)
        image = load_image_for_yolo(image_path)
        h, w = int(image.shape[0]), int(image.shape[1])
        results = model.predict(
            source=np.ascontiguousarray(image),
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            verbose=False,
            retina_masks=True,
        )
        result = results[0]
        instance_map, masks, scores, classes = ultralytics_result_prediction_arrays(
            result, height=h, width=w
        )
        _write_prediction_artifacts(
            output_dir=args.output_dir,
            sample_id=image_path.stem,
            image_path=image_path,
            height=h,
            width=w,
            instance_map=instance_map,
            masks=masks,
            scores=scores,
            classes=classes,
            orig_shape=tuple(result.orig_shape),
        )


def run_whole_predict(args: argparse.Namespace) -> None:
    from sahi import AutoDetectionModel

    image_paths = _load_whole_image_paths(args)
    n_samples = len(image_paths)

    device = device_for_sahi(_parse_device(args.device))
    weights_path = Path(args.weights).resolve()
    print(
        f"YOLO whole predict (SAHI): {n_samples} image(s), "
        f"slice={args.slice_height}x{args.slice_width}, "
        f"overlap=({args.overlap_height_ratio}, {args.overlap_width_ratio})",
        flush=True,
    )
    print(f"Loading detection model from {weights_path}...", flush=True)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(weights_path),
        confidence_threshold=args.conf,
        device=device,
        image_size=args.imgsz,
    )

    for idx, tiff_path in enumerate(image_paths):
        if not tiff_path.is_file():
            raise FileNotFoundError(f"Image not found: {tiff_path}")
        print(f"Predicting {tiff_path.stem} ({idx + 1}/{n_samples})...", flush=True)
        image = load_image_for_yolo(tiff_path)
        height, width = image.shape[:2]
        result = _get_sliced_prediction_preserve_channels(
            image,
            detection_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_height_ratio,
            overlap_width_ratio=args.overlap_width_ratio,
        )
        masks, scores, classes = sahi_predictions_to_mask_arrays(
            result.object_prediction_list, height, width
        )
        instance_map = instance_map_from_masks(
            masks, scores, height=height, width=width
        )
        _write_prediction_artifacts(
            output_dir=args.output_dir,
            sample_id=tiff_path.stem,
            image_path=tiff_path,
            height=height,
            width=width,
            instance_map=instance_map,
            masks=masks,
            scores=scores,
            classes=classes,
            orig_shape=(height, width),
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO instance prediction export.")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--unit",
        choices=("patch", "whole"),
        required=True,
    )
    parser.add_argument("--variant", choices=variant_choices(), default=None)
    parser.add_argument("--data", default=None, type=Path)
    parser.add_argument("--manifest", default=None, type=Path)
    parser.add_argument("--image", default=None, type=Path)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--slice-height", type=int, default=1024)
    parser.add_argument("--slice-width", type=int, default=1024)
    parser.add_argument("--overlap-height-ratio", type=float, default=0.5)
    parser.add_argument("--overlap-width-ratio", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.unit == "patch":
        data_yaml = _resolve_data_yaml(args)
        if data_yaml is None and args.manifest is None:
            if args.variant is None or _resolve_patch_manifest(args) is None:
                parser.error(
                    "patch mode requires --manifest, --variant (with patch "
                    "manifest on scratch), or --data"
                )
        run_patch_predict(args, data_yaml)
        return

    if args.manifest is None and args.image is None:
        parser.error("whole mode requires --manifest or --image")
    run_whole_predict(args)


if __name__ == "__main__":
    main()
