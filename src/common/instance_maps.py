from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
from pycocotools import mask as mask_utils


def segmentation_to_binary_mask(
    segmentation: list | dict, height: int, width: int
) -> np.ndarray:
    if isinstance(segmentation, dict):
        return mask_utils.decode(segmentation).astype(bool)
    rles = mask_utils.frPyObjects(segmentation, height, width)
    if isinstance(rles, list):
        rle = mask_utils.merge(rles) if len(rles) > 1 else rles[0]
    else:
        rle = rles
    return mask_utils.decode(rle).astype(bool)


def gt_annotations_to_instance_map(
    gt_annotations: list[dict[str, Any]], height: int, width: int
) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.int32)
    sorted_anns = sorted(gt_annotations, key=lambda a: int(a["id"]))
    for ann in sorted_anns:
        lid = int(ann["id"])
        seg = ann["segmentation"]
        if seg is None or seg == [] or seg == {}:
            continue
        m = segmentation_to_binary_mask(seg, height, width)
        out[m] = lid
    return out


def binary_masks_to_instance_map_by_score(
    masks_hw: np.ndarray, scores: np.ndarray
) -> np.ndarray:
    """Rasterize masks in ascending score order (later paints win)."""
    if masks_hw.ndim != 3:
        raise ValueError(f"masks_hw must be (n, H, W), got {masks_hw.shape}")
    n, h, w = masks_hw.shape
    if scores.shape[0] != n:
        raise ValueError("scores length must match number of masks")
    order = np.argsort(scores.astype(np.float64))
    out = np.zeros((h, w), dtype=np.int32)
    for idx in order:
        raw = masks_hw[idx]
        m = raw > 0.5 if raw.dtype in (np.float32, np.float64) else raw.astype(bool)
        out[m] = int(idx) + 1
    return out


def yolo_detections_to_instance_map_by_score(
    detections: Sequence[Mapping[str, Any]],
    *,
    height: int,
    width: int,
    decode_segmentation: Callable[[Mapping[str, Any]], np.ndarray],
) -> np.ndarray:
    """Merge YOLO detections one decoded mask at a time (ascending score paint order)."""
    if not detections:
        return np.zeros((height, width), dtype=np.int32)
    scores = np.asarray([float(det["score"]) for det in detections], dtype=np.float64)
    order = np.argsort(scores)
    out = np.zeros((height, width), dtype=np.int32)
    for det_index in order:
        binary = decode_segmentation(detections[det_index]["segmentation"])
        out[binary] = int(det_index) + 1
    return out


def _sahi_object_score(pred: Any) -> float:
    score_obj = getattr(pred, "score", None)
    if score_obj is None:
        raise ValueError("SAHI object prediction missing score")
    value = getattr(score_obj, "value", None)
    if value is None:
        raise ValueError("SAHI object prediction score has no value")
    return float(value)


def sahi_object_binary_mask(
    pred: Any, *, height: int, width: int, mask_threshold: float
) -> np.ndarray | None:
    """Decode one SAHI object prediction to a full-section boolean mask."""
    from common.mask_ops import masks_hw_to_binary, resize_mask_nearest

    mask_obj = getattr(pred, "mask", None)
    if mask_obj is None:
        return None
    float_mask = getattr(mask_obj, "float_mask", None)
    if float_mask is not None:
        binary = masks_hw_to_binary(
            np.asarray(float_mask, dtype=np.float32)[None, ...],
            threshold=mask_threshold,
        )[0]
    else:
        mask = getattr(mask_obj, "bool_mask", None)
        if mask is None:
            return None
        binary = np.asarray(mask, dtype=bool)
    if binary.shape != (height, width):
        binary = resize_mask_nearest(binary.astype(np.uint8), height, width).astype(bool)
    return binary


def _prediction_bbox_int_slice(
    pred: Any, *, height: int, width: int
) -> tuple[int, int, int, int]:
    bbox_obj = getattr(pred, "bbox", None)
    if bbox_obj is not None and hasattr(bbox_obj, "to_xyxy"):
        bbox = bbox_obj.to_xyxy()
    else:
        mask_obj = getattr(pred, "mask", None)
        if mask_obj is not None:
            stored = getattr(mask_obj, "bool_mask", None)
            if isinstance(stored, np.ndarray) and stored.ndim == 2:
                ox, oy = _mask_shift_xy(mask_obj)
                crop_h, crop_w = stored.shape
                bbox = [float(ox), float(oy), float(ox + crop_w), float(oy + crop_h)]
            else:
                bbox = [0.0, 0.0, float(width), float(height)]
        else:
            bbox = [0.0, 0.0, float(width), float(height)]
    x1 = max(0, min(width, int(np.floor(float(bbox[0])))))
    y1 = max(0, min(height, int(np.floor(float(bbox[1])))))
    x2 = max(0, min(width, int(np.ceil(float(bbox[2])))))
    y2 = max(0, min(height, int(np.ceil(float(bbox[3])))))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return y1, x1, y2, x2


def _stored_bool_mask_array(
    mask_obj: Any, *, height: int, width: int
) -> np.ndarray | None:
    """Dense ``bool_mask`` on adapter/fake objects; ``None`` for SAHI ``Mask`` (use segmentation)."""
    if type(mask_obj).__name__ == "Mask" and str(type(mask_obj).__module__).startswith(
        "sahi"
    ):
        return None
    candidate = getattr(mask_obj, "bool_mask", None)
    if isinstance(candidate, np.ndarray) and candidate.ndim == 2:
        return candidate
    return None


def _mask_shift_xy(mask_obj: Any) -> tuple[int, int]:
    shift = getattr(mask_obj, "shift_amount", (0, 0))
    if shift is None:
        return 0, 0
    if len(shift) >= 2:
        return int(shift[0]), int(shift[1])
    return 0, 0


def _binary_mask_in_bbox(
    mask_obj: Any,
    *,
    y1: int,
    x1: int,
    y2: int,
    x2: int,
    height: int,
    width: int,
    mask_threshold: float,
) -> np.ndarray | None:
    """Mask pixels covering only the prediction bbox (avoids SAHI full-section fillPoly)."""
    stored = _stored_bool_mask_array(mask_obj, height=height, width=width)
    if stored is not None:
        if stored.shape == (height, width):
            return stored[y1:y2, x1:x2]
        ox, oy = _mask_shift_xy(mask_obj)
        crop_h, crop_w = stored.shape
        target_y1 = max(y1, oy)
        target_x1 = max(x1, ox)
        target_y2 = min(y2, oy + crop_h)
        target_x2 = min(x2, ox + crop_w)
        if target_y2 <= target_y1 or target_x2 <= target_x1:
            return np.zeros((max(0, y2 - y1), max(0, x2 - x1)), dtype=bool)
        sy0, sx0 = target_y1 - oy, target_x1 - ox
        sy1, sx1 = target_y2 - oy, target_x2 - ox
        out = np.zeros((y2 - y1, x2 - x1), dtype=bool)
        out[target_y1 - y1 : target_y2 - y1, target_x1 - x1 : target_x2 - x1] = stored[
            sy0:sy1, sx0:sx1
        ]
        return out

    segmentation = getattr(mask_obj, "segmentation", None)
    if segmentation is None or segmentation == [] or segmentation == {}:
        float_mask = getattr(mask_obj, "float_mask", None)
        if float_mask is None:
            return None
        from common.mask_ops import masks_hw_to_binary, resize_mask_nearest

        binary = masks_hw_to_binary(
            np.asarray(float_mask, dtype=np.float32)[None, ...],
            threshold=mask_threshold,
        )[0]
        if binary.shape == (height, width):
            return binary[y1:y2, x1:x2]
        binary = resize_mask_nearest(binary.astype(np.uint8), height, width).astype(bool)
        return binary[y1:y2, x1:x2]

    if isinstance(segmentation, dict):
        full = mask_utils.decode(segmentation).astype(bool)
        return full[y1:y2, x1:x2]

    from common.mask_ops import rasterize_coco_polygons_in_box

    return rasterize_coco_polygons_in_box(
        segmentation, y1=y1, x1=x1, y2=y2, x2=x2
    )


def sahi_object_section_binary_mask(
    pred: Any, *, height: int, width: int, mask_threshold: float
) -> np.ndarray | None:
    """One full-section boolean mask for encode / parity (bbox-local decode when possible)."""
    mask_obj = getattr(pred, "mask", None)
    if mask_obj is None:
        return None
    y1, x1, y2, x2 = _prediction_bbox_int_slice(pred, height=height, width=width)
    crop = _binary_mask_in_bbox(
        mask_obj,
        y1=y1,
        x1=x1,
        y2=y2,
        x2=x2,
        height=height,
        width=width,
        mask_threshold=mask_threshold,
    )
    if crop is None or not crop.any():
        return None
    if crop.shape == (height, width):
        return crop
    full = np.zeros((height, width), dtype=bool)
    full[y1:y2, x1:x2] = crop
    return full


def paint_sahi_prediction_into_instance_map(
    out: np.ndarray,
    pred: Any,
    instance_id: int,
    *,
    height: int,
    width: int,
    mask_threshold: float,
) -> bool:
    """Paint one SAHI prediction into ``out`` using only its bbox window when possible."""
    mask_obj = getattr(pred, "mask", None)
    if mask_obj is None:
        return False
    y1, x1, y2, x2 = _prediction_bbox_int_slice(pred, height=height, width=width)
    binary = _binary_mask_in_bbox(
        mask_obj,
        y1=y1,
        x1=x1,
        y2=y2,
        x2=x2,
        height=height,
        width=width,
        mask_threshold=mask_threshold,
    )
    if binary is None or not binary.any():
        return False
    out[y1:y2, x1:x2][binary] = instance_id
    return True


def score_merged_instance_map_from_sahi_predictions(
    predictions: Sequence[Any],
    *,
    height: int,
    width: int,
    mask_threshold: float,
) -> np.ndarray:
    """Paint overlapping SAHI proposals by ascending score (no prediction-set RLE round-trip).

    Decodes and paints one mask at a time so slice-merged full-section masks are not
    all materialized in a list before painting (ADR 0005; avoids candidate OOM).
    """
    if not predictions:
        return np.zeros((height, width), dtype=np.int32)
    scores = np.asarray(
        [_sahi_object_score(pred) for pred in predictions], dtype=np.float64
    )
    order = np.argsort(scores)
    out = np.zeros((height, width), dtype=np.int32)
    for det_index in order:
        paint_sahi_prediction_into_instance_map(
            out,
            predictions[det_index],
            int(det_index) + 1,
            height=height,
            width=width,
            mask_threshold=mask_threshold,
        )
    return out
