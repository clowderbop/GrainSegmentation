from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from tifffile import TiffFile


from evaluation.metrics import (
    compute_aji,
    compute_instance_metrics_dict,
    get_instances,
)

from evaluation.reporting import (
    build_instance_eval_report,
    build_sample_row,
    count_instances,
    json_safe_for_dump,
)
from common.geometry import load_image_space_polygons
from common.image_io import load_tiff_single_channel_mask, validate_semantic_labels
from common.instance_maps import (
    binary_masks_to_instance_map_by_confidence,
    dt_annotations_to_instance_map,
    gt_annotations_to_instance_map,
    segmentation_to_binary_mask,
)
from yolo.coco_instance_ap import (
    build_gt_annotations,
    evaluate_mask_ap,
    object_predictions_to_coco_dt,
)
from yolo.config import variant_choices
from yolo.dataset_yaml import (
    default_labels_dir,
    load_yaml_dataset_config,
    resolve_split_dir,
)
from yolo.pipeline import resolve_variant_paths
from yolo.train import _parse_device


def _visualization_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return image[:, :, :3]


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    display = _visualization_image(image)
    if display.ndim == 2:
        display = np.repeat(display[:, :, None], 3, axis=2)
    elif display.shape[2] == 1:
        display = np.repeat(display, 3, axis=2)
    return np.clip(display, 0, 255).astype(np.uint8, copy=False)


def write_mask_overlay_visual(
    image: np.ndarray, pred_map: np.ndarray, out_path: Path
) -> None:
    import tifffile

    visual = _as_rgb_uint8(image).astype(np.float32)
    labels = np.unique(pred_map)
    labels = labels[labels > 0]
    for label in labels:
        mask = pred_map == label
        color = np.array(
            [
                (37 * int(label)) % 255,
                (97 * int(label)) % 255,
                (173 * int(label)) % 255,
            ],
            dtype=np.float32,
        )
        visual[mask] = (0.45 * visual[mask]) + (0.55 * color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        out_path,
        np.clip(visual, 0, 255).astype(np.uint8),
        photometric="rgb",
        compression="deflate",
    )


def _mask_to_polygons(mask: np.ndarray) -> list[Any]:
    import cv2
    from shapely.geometry import Polygon

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons: list[Any] = []
    for contour in contours:
        points = contour.reshape(-1, 2)
        if len(points) < 3:
            continue
        polygon = Polygon([(float(x), float(y)) for x, y in points])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty and polygon.area > 0:
            polygons.append(polygon)
    return polygons


def _segmentation_to_polygons(
    segmentation: list | dict, height: int, width: int
) -> list[Any]:
    from shapely.geometry import Polygon

    if isinstance(segmentation, list):
        polygons: list[Any] = []
        for ring in segmentation:
            if len(ring) < 6:
                continue
            coords = list(zip(ring[0::2], ring[1::2], strict=False))
            polygon = Polygon([(float(x), float(y)) for x, y in coords])
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty and polygon.area > 0:
                polygons.append(polygon)
        return polygons
    mask = segmentation_to_binary_mask(segmentation, height, width)
    return _mask_to_polygons(mask)


def write_predicted_masks_gpkg(
    dt_annotations: list[dict[str, Any]], *, height: int, width: int, out_path: Path
) -> None:
    import geopandas as gpd

    records: list[dict[str, Any]] = []
    geometries: list[Any] = []
    for instance_id, ann in enumerate(dt_annotations, start=1):
        segmentation = ann.get("segmentation")
        if segmentation is None or segmentation == [] or segmentation == {}:
            continue
        for polygon in _segmentation_to_polygons(segmentation, height, width):
            records.append(
                {
                    "instance_id": instance_id,
                    "category_id": int(ann.get("category_id", 0)),
                    "score": float(ann.get("score", 0.0)),
                }
            )
            geometries.append(polygon)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    geodata = gpd.GeoDataFrame(records, geometry=geometries)
    if geodata.empty:
        geodata = gpd.GeoDataFrame(
            {
                "instance_id": [],
                "category_id": [],
                "score": [],
                "geometry": [],
            },
            geometry="geometry",
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="'crs' was not provided.*",
            category=UserWarning,
        )
        geodata.to_file(out_path, layer="predictions", driver="GPKG")


class _NumpyPredictionResult:
    def __init__(self, image: np.ndarray, object_prediction_list: list[Any]) -> None:
        self.image = image
        self.object_prediction_list = object_prediction_list

    def export_visuals(self, export_dir: str, file_name: str) -> None:
        from sahi.utils.cv import visualize_object_predictions

        visualize_object_predictions(
            image=np.ascontiguousarray(_visualization_image(self.image)),
            object_prediction_list=self.object_prediction_list,
            output_dir=export_dir,
            file_name=file_name,
            export_format="png",
        )


def _perform_ultralytics_inference_preserve_channels(
    detection_model: Any, image: np.ndarray
) -> None:
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
    elif detection_model.is_obb:
        device = getattr(detection_model.model, "device", "cpu")
        prediction_result = [
            (
                torch.cat(
                    [
                        result.obb.xyxy,
                        result.obb.conf.unsqueeze(-1),
                        result.obb.cls.unsqueeze(-1),
                    ],
                    dim=1,
                )
                if result.obb is not None
                else torch.empty((0, 6), device=device),
                result.obb.xyxyxyxy
                if result.obb is not None
                else torch.empty((0, 4, 2), device=device),
            )
            for result in prediction_result
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
    verbose: int = 0,
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
    start = time.time()
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
    if verbose:
        print(
            f"Performed multichannel sliced prediction on {len(slice_bboxes)} slices "
            f"in {time.time() - start:.2f}s."
        )
    return _NumpyPredictionResult(
        image=image, object_prediction_list=object_prediction_list
    )


_PATCH_IMAGE_SUFFIXES = {".tif", ".tiff"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        )
    parser.add_argument(
        "--mode",
        choices=("val", "sahi", "patches"),
        required=True,
        )
    parser.add_argument(
        "--weights",
        required=True,
        )
    parser.add_argument(
        "--variant",
        choices=variant_choices(),
        default=None,
        )
    parser.add_argument(
        "--data",
        default=None,
        type=Path,
        )
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--device",
        default="0",
        )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        )
    parser.add_argument(
        "--name",
        default="test",
        )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=False,
        )
    parser.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        )
    parser.add_argument(
        "--save-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        )

    parser.add_argument("--slice-height", type=int, default=1024)
    parser.add_argument("--slice-width", type=int, default=1024)
    parser.add_argument(
        "--overlap-height-ratio",
        type=float,
        default=0.5,
        )
    parser.add_argument(
        "--overlap-width-ratio",
        type=float,
        default=0.5,
        )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        )
    parser.add_argument(
        "--test-tiff",
        type=Path,
        default=None,
        )
    parser.add_argument(
        "--test-gpkg",
        type=Path,
        default=None,
        )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        )
    parser.add_argument(
        "--mask-stem-suffix",
        default="_labels",
        )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        )
    parser.add_argument(
        "--sahi-out-dir",
        type=Path,
        default=None,
        )
    parser.add_argument(
        "--run-ultralytics-val",
        action=argparse.BooleanOptionalAction,
        default=False,
        )

    args = parser.parse_args(argv)
    if args.mode == "sahi":
        if args.manifest is not None:
            pass
        elif args.test_tiff is not None and args.test_gpkg is not None:
            pass
        else:
            parser.error("sahi requires --manifest or both --test-tiff and --test-gpkg")
    elif args.mode == "patches":
        if args.output_json is None:
            parser.error("--mode patches requires --output-json")
        if not args.variant and not args.data:
            parser.error("one of --variant or --data is required")
    elif not args.variant and not args.data:
        parser.error("one of --variant or --data is required")
    if args.slice_height <= 0 or args.slice_width <= 0:
        parser.error("slice dimensions must be positive")
    return args


def _resolve_data_yaml(args: argparse.Namespace) -> Path:
    if args.data is not None:
        path = args.data.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {path}")
        return path
    resolved = resolve_variant_paths(variant_name=args.variant)
    if not resolved.data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {resolved.data_yaml}")
    return resolved.data_yaml




def collect_yolo_patch_pairs(
    dataset_root: Path, config: dict[str, Any]
) -> tuple[Path, list[Path]]:
    for split_name in ("test", "val"):
        rel = config.get(split_name)
        if not rel:
            continue
        if split_name == "test" and config.get("val"):
            print(
                "YOLO patch evaluation: using the `test` split only. "
                "The dataset YAML also defines `val`, which is ignored for this run.",
                file=sys.stderr,
            )
        image_dir = resolve_split_dir(dataset_root, str(rel))
        if not image_dir.is_dir():
            raise FileNotFoundError(
                f"Missing image directory for split {split_name!r}: {image_dir}"
            )
        label_dir = default_labels_dir(dataset_root, split_name, image_dir)
        image_paths: list[Path] = []
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in _PATCH_IMAGE_SUFFIXES:
                continue
            image_paths.append(image_path)
        if not image_paths:
            raise ValueError(f"No patch images found under {image_dir}")
        return label_dir, image_paths
    raise ValueError(
        "Dataset YAML must define a `test` or `val` split for patch evaluation"
    )


def ultralytics_result_to_instance_map(
    result: Any, height: int, width: int
) -> np.ndarray:
    import cv2

    if result.masks is None or len(result.masks) == 0:
        return np.zeros((height, width), dtype=np.int32)
    data = result.masks.data.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    masks_list: list[np.ndarray] = []
    for i in range(data.shape[0]):
        m = data[i]
        if m.shape != (height, width):
            m = cv2.resize(
                m.astype(np.float32),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
        masks_list.append(m > 0.5)
    stacked = np.stack(masks_list, axis=0)
    return binary_masks_to_instance_map_by_confidence(stacked, conf)


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

    image = np.clip(image, 0, 255).astype(np.uint8, copy=False)
    return image


def load_semantic_patch_mask(path: Path) -> np.ndarray:
    arr = load_tiff_single_channel_mask(path)
    return validate_semantic_labels(arr, path, allow_float=True)


def _optional_metric_attr(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _metric_section(obj: Any) -> dict[str, Any]:
    keys = (
        "map",
        "map50",
        "map75",
        "maps",
        "mp",
        "mr",
        "p",
        "r",
        "f1",
        "ap_class_index",
        "image_metrics",
    )
    out: dict[str, Any] = {}
    for key in keys:
        value = _optional_metric_attr(obj, key)
        if value is not None:
            out[key] = json_safe_for_dump(value)
    return out


def _collect_val_metrics(metrics: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for section in ("box", "seg", "mask", "pose", "obb"):
        values = _metric_section(_optional_metric_attr(metrics, section))
        if values:
            payload[section] = values
    for key in ("speed", "results_dict", "fitness"):
        value = _optional_metric_attr(metrics, key)
        if value is not None:
            payload[key] = json_safe_for_dump(value)
    return payload


def write_val_metrics_json(
    metrics: Any, *, project: Path | None, name: str
) -> Path | None:
    if project is None:
        return None
    out_path = project.resolve() / name / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            json_safe_for_dump(_collect_val_metrics(metrics)), indent=2, allow_nan=False
        ),
        encoding="utf-8",
    )
    print(f"Wrote val metrics JSON to {out_path}")
    return out_path


def run_patches(args: argparse.Namespace, data_yaml: Path) -> dict[str, Any]:
    from ultralytics import YOLO

    dataset_root, config = load_yaml_dataset_config(data_yaml)
    label_dir, image_paths = collect_yolo_patch_pairs(dataset_root, config)

    device = _parse_device(args.device)
    model = YOLO(str(Path(args.weights).resolve()))
    sample_rows: list[dict[str, Any]] = []

    mask_suffix = str(args.mask_stem_suffix)

    for image_path in image_paths:
        image = load_image_for_yolo(image_path)
        h, w = int(image.shape[0]), int(image.shape[1])
        mask_path = label_dir / (f"{image_path.stem}{mask_suffix}{image_path.suffix}")
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Pre-computed semantic mask not found for {image_path}: expected {mask_path}. "
                "Write UNet-aligned `{stem}_labels.<ext>` rasters next to YOLO labels "
                "(see SLURM/preprocessing/08_create_unet_test_patches_from_yolo_patches.sh or "
                "src/data_prep/crop_unet_masks_from_yolo_patches.py). "
                "Polygon .txt labels are not used for patch metrics."
            )
        sem = load_semantic_patch_mask(mask_path)
        if sem.shape != (h, w):
            raise ValueError(
                f"Mask shape {sem.shape} does not match image shape {(h, w)} for {mask_path}"
            )
        gt_map = get_instances(sem, interior_class=1)

        results = model.predict(
            source=np.ascontiguousarray(image),
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            verbose=False,
            retina_masks=True,
        )
        pred_map = ultralytics_result_to_instance_map(results[0], h, w)

        aji = float(compute_aji(gt_map, pred_map))
        inst = {
            k: float(v)
            for k, v in compute_instance_metrics_dict(gt_map, pred_map).items()
        }
        metrics: dict[str, float] = {"aji": aji, **inst}
        gt_n = count_instances(gt_map)
        pred_n = count_instances(pred_map)
        sample_rows.append(
            build_sample_row(
                image_path.stem,
                metrics=metrics,
                gt_instances=gt_n,
                pred_instances=pred_n,
                empty_gt=gt_n == 0,
                extra={"image_path": str(image_path.resolve())},
            )
        )
        print(
            f"{image_path.name}: AJI={aji:.4f} mF1@0.5:0.95="
            f"{metrics['mF1_iou50_95']:.4f} GT={gt_n} Pred={pred_n}"
        )

    if not sample_rows:
        print(
            "ERROR: YOLO patch evaluation produced no per-image metric rows.",
            file=sys.stderr,
        )
        sys.exit(1)

    extras: dict[str, Any] | None = None
    if args.run_ultralytics_val:
        val_metrics = run_val(args, data_yaml)
        extras = {"ultralytics": _collect_val_metrics(val_metrics)}

    report = build_instance_eval_report(
        model_type="yolo",
        variant=args.variant,
        unit="patch",
        samples=sample_rows,
        extras=extras,
    )
    assert args.output_json is not None
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(json_safe_for_dump(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote patch instance metrics to {args.output_json}")
    return report


def run_val(args: argparse.Namespace, data_yaml: Path) -> Any:
    from ultralytics import YOLO

    device = _parse_device(args.device)
    model = YOLO(str(Path(args.weights).resolve()))
    val_kwargs: dict[str, Any] = dict(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        split="test",
        plots=args.plots,
        half=args.half,
    )
    if args.save_json:
        val_kwargs["save_json"] = True
    val_kwargs["name"] = args.name
    if args.project is not None:
        val_kwargs["project"] = str(args.project.resolve())
    metrics = model.val(**val_kwargs)
    write_val_metrics_json(metrics, project=args.project, name=args.name)
    return metrics


def _resolve_manifest_path(raw: str, manifest_dir: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p.resolve()
    return (manifest_dir / p).resolve()


def _load_sahi_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.manifest is not None:
        manifest_path = args.manifest.resolve()
        manifest_dir = manifest_path.parent
        raw = json.loads(args.manifest.read_text(encoding="utf-8"))
        pairs: list[tuple[Path, Path]] = []
        for index, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise ValueError(f"manifest[{index}] must be an object")
            tiff = entry.get("test_tiff") or entry.get("tiff")
            gpkg = entry.get("test_gpkg") or entry.get("gpkg")
            if not tiff or not gpkg:
                raise ValueError(
                    f"manifest[{index}] needs test_tiff and test_gpkg keys"
                )
            pairs.append(
                (
                    _resolve_manifest_path(str(tiff), manifest_dir),
                    _resolve_manifest_path(str(gpkg), manifest_dir),
                )
            )
        return pairs
    return [(args.test_tiff.resolve(), args.test_gpkg.resolve())]


def aggregate_sahi_means(
    per_image: list[dict[str, Any]],
) -> dict[str, float | None]:
    coco_mean_keys = (
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
    )
    instance_mean_keys = (
        "aji",
        "precision_iou50",
        "recall_iou50",
        "f1_iou50",
        "precision_iou75",
        "recall_iou75",
        "f1_iou75",
        "mP_iou50_95",
        "mR_iou50_95",
        "mF1_iou50_95",
    )
    out: dict[str, float | None] = {}
    for key in coco_mean_keys:
        values = [
            float(row[key])
            for row in per_image
            if np.isfinite(row[key]) and row[key] >= 0
        ]
        out[f"mean_{key}"] = float(np.mean(values)) if values else None
    for key in instance_mean_keys:
        values = [
            float(row[key])
            for row in per_image
            if key in row and np.isfinite(row[key]) and row[key] >= 0
        ]
        out[f"mean_{key}"] = float(np.mean(values)) if values else None
    return out


def run_sahi(args: argparse.Namespace) -> dict[str, Any]:
    from sahi import AutoDetectionModel

    pairs = _load_sahi_pairs(args)
    device = device_for_sahi(_parse_device(args.device))
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=str(Path(args.weights).resolve()),
        confidence_threshold=args.conf,
        device=device,
    )

    per_image: list[dict[str, Any]] = []
    for image_id, (tiff_path, gpkg_path) in enumerate(pairs, start=1):
        if not tiff_path.is_file():
            raise FileNotFoundError(f"test TIFF not found: {tiff_path}")
        if not gpkg_path.is_file():
            raise FileNotFoundError(f"test GPKG not found: {gpkg_path}")

        image = load_image_for_yolo(tiff_path)
        height, width = image.shape[:2]
        polygons = load_image_space_polygons(gpkg_path)
        gt_anns = build_gt_annotations(
            polygons,
            image_id=image_id,
            height=height,
            width=width,
        )
        result = _get_sliced_prediction_preserve_channels(
            image,
            detection_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_height_ratio,
            overlap_width_ratio=args.overlap_width_ratio,
            verbose=0,
        )
        dt_anns = object_predictions_to_coco_dt(
            result.object_prediction_list,
            image_id=image_id,
            height=height,
            width=width,
        )
        summary = evaluate_mask_ap(
            image_id=image_id,
            file_name=tiff_path.name,
            height=height,
            width=width,
            gt_annotations=gt_anns,
            dt_annotations=dt_anns,
        )
        gt_map = gt_annotations_to_instance_map(gt_anns, height, width)
        pred_map = dt_annotations_to_instance_map(dt_anns, height, width)
        aji = float(compute_aji(gt_map, pred_map))
        inst_metrics = {
            k: float(v)
            for k, v in compute_instance_metrics_dict(gt_map, pred_map).items()
        }

        row: dict[str, Any] = {
            "test_tiff": str(tiff_path),
            "test_gpkg": str(gpkg_path),
            "image_id": image_id,
            "gt_instances": len(gt_anns),
            "pred_instances": len(dt_anns),
            "empty_gt": len(gt_anns) == 0,
            "aji": aji,
        }
        row.update(inst_metrics)
        row.update(summary.to_dict())
        per_image.append(row)
        if args.sahi_out_dir is not None:
            out_root = args.sahi_out_dir.resolve()
            out_root.mkdir(parents=True, exist_ok=True)
            sub = out_root / tiff_path.stem
            sub.mkdir(parents=True, exist_ok=True)
            write_mask_overlay_visual(image, pred_map, sub / "prediction_visual.tif")
            write_predicted_masks_gpkg(
                dt_anns,
                height=height,
                width=width,
                out_path=sub / "predicted_masks.gpkg",
            )
        print(
            f"{tiff_path.name}: AP={summary.ap_50_95:.4f} AP50={summary.ap_50:.4f} "
            f"AJI={aji:.4f} mF1@0.5:0.95={inst_metrics['mF1_iou50_95']:.4f} "
            f"GT={len(gt_anns)} Pred={len(dt_anns)}"
        )

    aggregate: dict[str, Any] = {"per_image": per_image}
    aggregate.update(aggregate_sahi_means(per_image))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(json_safe_for_dump(aggregate), indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"Wrote metrics JSON to {args.output_json}")
    return aggregate


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.mode == "sahi":
        run_sahi(args)
        return

    data_yaml = _resolve_data_yaml(args)
    if args.mode == "patches":
        run_patches(args, data_yaml)
        return

    run_val(args, data_yaml)


if __name__ == "__main__":
    main()
