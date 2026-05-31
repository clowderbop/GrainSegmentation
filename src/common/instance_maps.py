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
