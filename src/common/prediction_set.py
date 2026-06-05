"""Instance prediction set schema v1: load, save, validate, and merged instance view."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable, Mapping
from typing import Any, Literal, overload

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


def yolo_detection_mask_in_section(
    det: Mapping[str, Any],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Decode one YOLO grain mask into whole-image coordinates.

    Crop-local segmentations use optional ``offset_y`` / ``offset_x`` (association output).
    """
    mask = segmentation_to_binary_mask(det["segmentation"])
    offset_y = int(det.get("offset_y", 0))
    offset_x = int(det.get("offset_x", 0))
    if offset_y == 0 and offset_x == 0 and mask.shape == (height, width):
        return mask
    crop_h, crop_w = mask.shape
    if offset_y < 0 or offset_x < 0:
        raise ValueError(f"Invalid mask offset: ({offset_y}, {offset_x})")
    if offset_y + crop_h > height or offset_x + crop_w > width:
        raise ValueError(
            f"Mask crop {crop_h}x{crop_w} at ({offset_y}, {offset_x}) "
            f"extends outside section {height}x{width}"
        )
    plane = np.zeros((height, width), dtype=bool)
    plane[offset_y : offset_y + crop_h, offset_x : offset_x + crop_w] = mask
    return plane


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
            det: dict[str, Any] = {
                "segmentation": seg,
                "score": float(raw["score"]),
                "category_id": GRAIN_CLASS_ID,
            }
            if "offset_y" in raw or "offset_x" in raw:
                if "offset_y" not in raw or "offset_x" not in raw:
                    raise ValueError(
                        f"yolo detections[{index}] offset_y and offset_x must both be set"
                    )
                det["offset_y"] = int(raw["offset_y"])
                det["offset_x"] = int(raw["offset_x"])
            detections.append(det)
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
    mask_threshold: float = 0.5,
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
        binary = masks_hw_to_binary(
            np.asarray(mask_plane)[None, ...], threshold=mask_threshold
        )[0]
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
    return _prediction_set_from_instance_map(
        instance_map, producer="unet", include_score=False
    )


def build_yolo_prediction_set_from_instance_map(
    instance_map: np.ndarray,
    *,
    score_for_label: Callable[[int], float],
) -> PredictionSet:
    """Encode disjoint YOLO grains from a score-merged instance label map."""
    return _prediction_set_from_instance_map(
        instance_map,
        producer="yolo",
        include_score=True,
        score_for_label=score_for_label,
    )


@overload
def _prediction_set_from_instance_map(
    instance_map: np.ndarray,
    *,
    producer: Producer,
    include_score: Literal[False],
    score_for_label: None = ...,
) -> PredictionSet: ...


@overload
def _prediction_set_from_instance_map(
    instance_map: np.ndarray,
    *,
    producer: Producer,
    include_score: Literal[True],
    score_for_label: Callable[[int], float],
) -> PredictionSet: ...


def _prediction_set_from_instance_map(
    instance_map: np.ndarray,
    *,
    producer: Producer,
    include_score: bool,
    score_for_label: Callable[[int], float] | None = None,
) -> PredictionSet:
    arr = np.asarray(instance_map)
    if arr.ndim != 2:
        raise ValueError(f"instance_map must be 2D, got shape {arr.shape}")
    height, width = int(arr.shape[0]), int(arr.shape[1])
    detections: list[dict[str, Any]] = []
    for label_id in sorted(int(x) for x in np.unique(arr) if x != 0):
        binary = arr == label_id
        if not binary.any():
            continue
        det: dict[str, Any] = {
            "segmentation": binary_mask_to_segmentation(
                binary, height=height, width=width
            ),
            "category_id": GRAIN_CLASS_ID,
        }
        if include_score:
            if score_for_label is None:
                raise ValueError("score_for_label is required for YOLO prediction sets")
            det["score"] = float(score_for_label(label_id))
        detections.append(det)
    return PredictionSet(
        schema_version=1,
        height=height,
        width=width,
        producer=producer,
        detections=tuple(detections),
    )


def assert_yolo_grains_non_overlapping(prediction_set: PredictionSet) -> None:
    """Raise if any two YOLO grain masks in a canonical instance prediction set overlap."""
    if prediction_set.producer != "yolo":
        raise ValueError(
            f"non-overlap check requires producer 'yolo', got {prediction_set.producer!r}"
        )
    height, width = prediction_set.height, prediction_set.width
    occupied = np.zeros((height, width), dtype=bool)
    for det in prediction_set.detections:
        mask = yolo_detection_mask_in_section(det, height=height, width=width)
        if np.any(occupied & mask):
            raise ValueError("YOLO instance prediction set has overlapping grain masks")
        occupied |= mask


def merge_yolo_proposals_by_score(prediction_set: PredictionSet) -> PredictionSet:
    """Merge overlapping YOLO detector proposals into non-overlapping grains."""
    if prediction_set.producer != "yolo":
        raise ValueError(
            f"score merge requires producer 'yolo', got {prediction_set.producer!r}"
        )
    height, width = prediction_set.height, prediction_set.width
    instance_map = yolo_detections_to_instance_map_by_score(
        prediction_set.detections,
        height=height,
        width=width,
        decode_segmentation=segmentation_to_binary_mask,
    )
    proposals = prediction_set.detections
    return build_yolo_prediction_set_from_instance_map(
        instance_map,
        score_for_label=lambda label_id: float(proposals[label_id - 1]["score"]),
    )


def yolo_prediction_set_to_coco_dt(
    prediction_set: PredictionSet | Path | str,
    *,
    image_id: int,
    height: int,
    width: int,
) -> list[dict[str, object]]:
    """Build COCO detection annotations from a YOLO instance prediction set (RLE + scores)."""
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
        binary = yolo_detection_mask_in_section(
            det, height=height, width=width
        )
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

    out = np.zeros((height, width), dtype=np.int32)
    for index, det in enumerate(prediction_set.detections):
        mask = yolo_detection_mask_in_section(det, height=height, width=width)
        out[mask] = index + 1
    return out


__all__ = [
    "GRAIN_CLASS_ID",
    "PREDICTION_SETS_SUBDIR",
    "PredictionSet",
    "Producer",
    "assert_yolo_grains_non_overlapping",
    "binary_mask_to_segmentation",
    "build_unet_prediction_set_from_instance_map",
    "build_yolo_prediction_set_from_instance_map",
    "build_yolo_prediction_set_from_ultralytics",
    "load_prediction_set",
    "merge_yolo_proposals_by_score",
    "prediction_set_filename",
    "prediction_set_path",
    "prediction_set_to_dict",
    "prediction_set_to_merged_instance_view",
    "save_prediction_set",
    "segmentation_to_binary_mask",
    "validate_prediction_set",
    "yolo_detection_mask_in_section",
    "yolo_prediction_set_to_coco_dt",
]
