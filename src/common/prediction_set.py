"""Instance prediction set schema v1: load, save, validate, and merged instance view."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pycocotools import mask as mask_utils

from common.instance_maps import yolo_detections_to_instance_map_by_score
from common.mask_ops import masks_hw_to_binary, resize_mask_nearest

PREDICTION_SETS_SUBDIR = "prediction_sets"
GRAIN_CLASS_ID = 0
Producer = Literal["yolo", "unet"]


@dataclass(frozen=True)
class PredictionSet:
    schema_version: int
    height: int
    width: int
    producer: Producer
    detections: tuple[dict[str, Any], ...]


def prediction_set_filename(sample_id: str) -> str:
    return f"{sample_id}.json"


def prediction_set_path(output_root: Path | str, sample_id: str) -> Path:
    return Path(output_root) / PREDICTION_SETS_SUBDIR / prediction_set_filename(sample_id)


def binary_mask_to_segmentation(mask: np.ndarray, *, height: int, width: int) -> dict[str, Any]:
    binary = np.asarray(mask, dtype=bool)
    if binary.shape != (height, width):
        binary = resize_mask_nearest(binary.astype(np.uint8), height, width).astype(bool)
    if not binary.any():
        return {"size": [int(height), int(width)], "counts": "0"}
    rle = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": [int(height), int(width)], "counts": counts}


def segmentation_to_binary_mask(segmentation: dict[str, Any]) -> np.ndarray:
    decoded = mask_utils.decode(segmentation)
    return decoded.astype(bool)


def _append_yolo_detection(
    detections: list[dict[str, Any]],
    binary: np.ndarray,
    score: float,
    *,
    height: int,
    width: int,
) -> None:
    if binary.shape != (height, width):
        binary = resize_mask_nearest(binary.astype(np.uint8), height, width).astype(bool)
    if not binary.any():
        return
    detections.append(
        {
            "segmentation": binary_mask_to_segmentation(binary, height=height, width=width),
            "score": float(score),
            "category_id": GRAIN_CLASS_ID,
        }
    )


def _sahi_object_score(pred: Any) -> float:
    score_obj = getattr(pred, "score", None)
    if score_obj is None:
        raise ValueError("SAHI object prediction missing score")
    value = getattr(score_obj, "value", None)
    if value is None:
        raise ValueError("SAHI object prediction score has no value")
    return float(value)


def validate_prediction_set(payload: dict[str, Any]) -> PredictionSet:
    try:
        schema_version = int(payload["schema_version"])
        height = int(payload["height"])
        width = int(payload["width"])
        producer = str(payload["producer"])
        raw_detections = payload["detections"]
    except KeyError as exc:
        raise ValueError(f"Prediction set missing required key: {exc}") from exc

    if schema_version != 1:
        raise ValueError(f"Unsupported schema_version: {schema_version}")
    if producer not in ("yolo", "unet"):
        raise ValueError(f"Invalid producer: {producer!r}")
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid image size: {height}x{width}")
    if not isinstance(raw_detections, list):
        raise ValueError('"detections" must be a list')

    detections: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_detections):
        if not isinstance(raw, dict):
            raise ValueError(f"detections[{index}] must be an object")
        seg = raw.get("segmentation")
        if not isinstance(seg, dict) or "size" not in seg or "counts" not in seg:
            raise ValueError(f"detections[{index}] requires COCO RLE segmentation")
        category_id = raw.get("category_id")
        if category_id != GRAIN_CLASS_ID:
            raise ValueError(
                f"detections[{index}] category_id must be {GRAIN_CLASS_ID}, got {category_id!r}"
            )
        has_score = "score" in raw
        if producer == "yolo":
            if not has_score:
                raise ValueError(f"yolo detections[{index}] requires score")
            detections.append(
                {
                    "segmentation": seg,
                    "score": float(raw["score"]),
                    "category_id": GRAIN_CLASS_ID,
                }
            )
        else:
            if has_score:
                raise ValueError(f"unet detections[{index}] must not include score")
            detections.append(
                {
                    "segmentation": seg,
                    "category_id": GRAIN_CLASS_ID,
                }
            )

    return PredictionSet(
        schema_version=schema_version,
        height=height,
        width=width,
        producer=producer,  # type: ignore[arg-type]
        detections=tuple(detections),
    )


def prediction_set_to_dict(prediction_set: PredictionSet) -> dict[str, Any]:
    return {
        "schema_version": prediction_set.schema_version,
        "height": prediction_set.height,
        "width": prediction_set.width,
        "producer": prediction_set.producer,
        "detections": list(prediction_set.detections),
    }


def save_prediction_set(path: Path | str, payload: dict[str, Any] | PredictionSet) -> None:
    if isinstance(payload, PredictionSet):
        data = prediction_set_to_dict(payload)
    else:
        data = prediction_set_to_dict(validate_prediction_set(payload))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def load_prediction_set(path: Path | str) -> PredictionSet:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Prediction set {path} must be a JSON object")
    return validate_prediction_set(payload)


def build_yolo_prediction_set_from_ultralytics(
    result: Any,
    *,
    height: int,
    width: int,
) -> PredictionSet:
    """Encode Ultralytics proposals one mask at a time (no dense ``(N, H, W)`` stack)."""
    if result.masks is None or len(result.masks) == 0:
        return PredictionSet(
            schema_version=1,
            height=height,
            width=width,
            producer="yolo",
            detections=(),
        )
    detections: list[dict[str, Any]] = []
    mask_tensors = result.masks.data
    box_scores = result.boxes.conf
    for index in range(len(result.masks)):
        mask_plane = mask_tensors[index].detach().cpu().numpy()
        score = float(box_scores[index].detach().cpu().item())
        binary = masks_hw_to_binary(np.asarray(mask_plane)[None, ...])[0]
        _append_yolo_detection(detections, binary, score, height=height, width=width)
    return PredictionSet(
        schema_version=1,
        height=height,
        width=width,
        producer="yolo",
        detections=tuple(detections),
    )


def build_yolo_prediction_set_from_sahi_predictions(
    predictions: list[Any],
    *,
    height: int,
    width: int,
) -> PredictionSet:
    """Encode SAHI object predictions one mask at a time (no dense ``(N, H, W)`` stack)."""
    detections: list[dict[str, Any]] = []
    for pred in predictions:
        mask_obj = getattr(pred, "mask", None)
        if mask_obj is None:
            continue
        mask = getattr(mask_obj, "bool_mask", None)
        if mask is None:
            continue
        binary = np.asarray(mask, dtype=bool)
        score = _sahi_object_score(pred)
        _append_yolo_detection(detections, binary, score, height=height, width=width)
    return PredictionSet(
        schema_version=1,
        height=height,
        width=width,
        producer="yolo",
        detections=tuple(detections),
    )


def build_unet_prediction_set_from_instance_map(
    instance_map: np.ndarray,
) -> PredictionSet:
    """Encode disjoint extracted grains from a U-Net instance label map."""
    arr = np.asarray(instance_map)
    if arr.ndim != 2:
        raise ValueError(f"instance_map must be 2D, got shape {arr.shape}")
    height, width = int(arr.shape[0]), int(arr.shape[1])
    detections: list[dict[str, Any]] = []
    for label_id in sorted(int(x) for x in np.unique(arr) if x != 0):
        binary = arr == label_id
        if not binary.any():
            continue
        detections.append(
            {
                "segmentation": binary_mask_to_segmentation(
                    binary, height=height, width=width
                ),
                "category_id": GRAIN_CLASS_ID,
            }
        )
    return PredictionSet(
        schema_version=1,
        height=height,
        width=width,
        producer="unet",
        detections=tuple(detections),
    )


def yolo_prediction_set_to_coco_dt(
    prediction_set: PredictionSet | Path | str,
    *,
    image_id: int,
    height: int,
    width: int,
) -> list[dict[str, object]]:
    """Build COCO detection annotations from YOLO detector proposals (RLE + scores)."""
    if not isinstance(prediction_set, PredictionSet):
        prediction_set = load_prediction_set(prediction_set)
    if prediction_set.producer != "yolo":
        raise ValueError(
            f"mask AP requires producer 'yolo', got {prediction_set.producer!r}"
        )
    if prediction_set.height != height or prediction_set.width != width:
        raise ValueError(
            f"Prediction set size {prediction_set.height}x{prediction_set.width} "
            f"does not match {height}x{width}"
        )
    detections: list[dict[str, object]] = []
    for det in prediction_set.detections:
        binary = segmentation_to_binary_mask(det["segmentation"])
        if not binary.any():
            continue
        ys, xs = np.where(binary)
        x0, x1 = float(xs.min()), float(xs.max())
        y0, y1 = float(ys.min()), float(ys.max())
        detections.append(
            {
                "image_id": int(image_id),
                "category_id": int(det["category_id"]),
                "segmentation": det["segmentation"],
                "bbox": [x0, y0, x1 - x0 + 1.0, y1 - y0 + 1.0],
                "score": float(det["score"]),
            }
        )
    return detections


def prediction_set_to_merged_instance_view(prediction_set: PredictionSet) -> np.ndarray:
    height, width = prediction_set.height, prediction_set.width
    if not prediction_set.detections:
        return np.zeros((height, width), dtype=np.int32)

    if prediction_set.producer == "yolo":
        return yolo_detections_to_instance_map_by_score(
            prediction_set.detections,
            height=height,
            width=width,
            decode_segmentation=segmentation_to_binary_mask,
        )

    out = np.zeros((height, width), dtype=np.int32)
    for index, det in enumerate(prediction_set.detections):
        mask = segmentation_to_binary_mask(det["segmentation"])
        out[mask] = index + 1
    return out


__all__ = [
    "GRAIN_CLASS_ID",
    "PREDICTION_SETS_SUBDIR",
    "PredictionSet",
    "Producer",
    "binary_mask_to_segmentation",
    "build_unet_prediction_set_from_instance_map",
    "build_yolo_prediction_set_from_sahi_predictions",
    "build_yolo_prediction_set_from_ultralytics",
    "load_prediction_set",
    "prediction_set_filename",
    "prediction_set_path",
    "prediction_set_to_dict",
    "prediction_set_to_merged_instance_view",
    "save_prediction_set",
    "segmentation_to_binary_mask",
    "validate_prediction_set",
    "yolo_prediction_set_to_coco_dt",
]
