"""Run YOLO segmentation inference and write instance prediction labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
from tifffile import TiffFile

from common.manifest_io import collect_manifest_image_paths
from common.yolo_seg_labels import YoloSegPredRow, write_yolo_seg_pred_label_file
from yolo.config import variant_choices
from yolo.dataset_yaml import (
    default_labels_dir,
    load_yaml_dataset_config,
    resolve_split_dir,
)
from yolo.pipeline import resolve_variant_paths
from yolo.train import _parse_device

_PATCH_IMAGE_SUFFIXES = {".tif", ".tiff"}


def _mask_to_polygons(mask: np.ndarray) -> list[np.ndarray]:
    import cv2

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons: list[np.ndarray] = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        polygons.append(points.astype(np.float32))
    return polygons


def _contour_area(points: np.ndarray) -> float:
    import cv2

    return float(cv2.contourArea(points.reshape(-1, 1, 2)))


def _largest_polygon_points(mask: np.ndarray) -> np.ndarray | None:
    polygons = _mask_to_polygons(mask)
    if not polygons:
        return None
    return max(polygons, key=_contour_area)


def ultralytics_result_to_pred_rows(
    result: Any, height: int, width: int, *, class_id: int = 0
) -> list[YoloSegPredRow]:
    import cv2

    if result.masks is None or len(result.masks) == 0:
        return []
    data = result.masks.data.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy()
    rows: list[YoloSegPredRow] = []
    for i in range(data.shape[0]):
        mask = data[i]
        if mask.shape != (height, width):
            mask = cv2.resize(
                mask.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        binary = mask > 0.5
        points = _largest_polygon_points(binary)
        if points is None:
            continue
        cid = int(cls[i]) if i < len(cls) else class_id
        score = float(conf[i]) if i < len(conf) else 0.0
        rows.append(YoloSegPredRow(class_id=cid, points=points, confidence=score))
    return rows


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
) -> tuple[Path, list[Path]]:
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
        label_dir = default_labels_dir(dataset_root, split_name, image_dir)
        return label_dir, image_paths
    raise ValueError("Dataset YAML must define a `test` or `val` split")


def _load_whole_image_paths(args: argparse.Namespace) -> list[Path]:
    if args.manifest is not None:
        return [image_path for image_path, _ in collect_manifest_image_paths(args.manifest)]
    if args.test_tiff is None:
        raise ValueError("whole mode requires --manifest or --test-tiff")
    return [args.test_tiff.resolve()]


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


def sahi_predictions_to_pred_rows(
    predictions: list[Any], height: int, width: int
) -> list[YoloSegPredRow]:
    rows: list[YoloSegPredRow] = []
    for pred in predictions:
        score = float(pred.score.value)
        category_id = int(pred.category.id)
        mask = pred.mask.bool_mask if pred.mask is not None else None
        if mask is None:
            continue
        if mask.shape != (height, width):
            import cv2

            mask = cv2.resize(
                mask.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ) > 0
        points = _largest_polygon_points(mask)
        if points is None:
            continue
        rows.append(
            YoloSegPredRow(class_id=category_id, points=points, confidence=score)
        )
    return rows


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


def run_patch_predict(args: argparse.Namespace, data_yaml: Path) -> None:
    from ultralytics import YOLO

    dataset_root, config = load_yaml_dataset_config(data_yaml)
    _label_dir, image_paths = collect_yolo_patch_image_paths(dataset_root, config)
    labels_dir = args.output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    device = _parse_device(args.device)
    model = YOLO(str(Path(args.weights).resolve()))

    for image_path in image_paths:
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
        rows = ultralytics_result_to_pred_rows(results[0], h, w)
        out_path = labels_dir / f"{image_path.stem}.txt"
        write_yolo_seg_pred_label_file(
            out_path, rows, image_width=w, image_height=h
        )
        print(f"Wrote {out_path} ({len(rows)} instances)")


def run_whole_predict(args: argparse.Namespace) -> None:
    from sahi import AutoDetectionModel

    image_paths = _load_whole_image_paths(args)
    labels_dir = args.output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    device = device_for_sahi(_parse_device(args.device))
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(Path(args.weights).resolve()),
        confidence_threshold=args.conf,
        device=device,
        image_size=args.imgsz,
    )

    for tiff_path in image_paths:
        if not tiff_path.is_file():
            raise FileNotFoundError(f"Image not found: {tiff_path}")
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
        rows = sahi_predictions_to_pred_rows(
            result.object_prediction_list, height, width
        )
        out_path = labels_dir / f"{tiff_path.stem}.txt"
        write_yolo_seg_pred_label_file(
            out_path, rows, image_width=width, image_height=height
        )
        print(f"Wrote {out_path} ({len(rows)} instances)")


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
    parser.add_argument("--test-tiff", default=None, type=Path)
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
        if data_yaml is None:
            parser.error("patch mode requires --variant or --data")
        run_patch_predict(args, data_yaml)
        return

    if args.manifest is None and args.test_tiff is None:
        parser.error("whole mode requires --manifest or --test-tiff")
    run_whole_predict(args)


if __name__ == "__main__":
    main()
