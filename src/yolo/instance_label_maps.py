from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


def read_yolo_seg_label_polygons(
    label_path: Path, height: int, width: int
) -> list[np.ndarray]:
    """
    YOLO segmentation labels: one instance per line, ``class x1 y1 x2 y2 ...`` normalized [0,1].
    Returns a list of ``(N, 2)`` float32 polygons in pixel coordinates.
    """
    polygons: list[np.ndarray] = []
    with label_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            values = [float(value) for value in line.split()]
            if len(values) < 7 or (len(values) - 1) % 2 != 0:
                raise ValueError(f"Invalid segmentation label row in {label_path}")
            points = np.asarray(values[1:], dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= float(width)
            points[:, 1] *= float(height)
            polygons.append(points)
    return polygons


def polygons_to_instance_map(
    polygons: list[np.ndarray], height: int, width: int
) -> np.ndarray:
    """Paint instance ids ``1..N`` in list order; later polygons overwrite overlaps."""
    import cv2

    out = np.zeros((height, width), dtype=np.int32)
    for i, pts in enumerate(polygons, start=1):
        if pts.size == 0:
            continue
        pts_i = np.round(pts).astype(np.int32)
        cv2.fillPoly(out, [pts_i], int(i))
    return out


def yolo_seg_label_txt_to_instance_map(
    label_path: Path, height: int, width: int
) -> np.ndarray:
    """Dense instance map from a YOLO ``.txt`` segmentation label file."""
    polygons = read_yolo_seg_label_polygons(label_path, height, width)
    return polygons_to_instance_map(polygons, height, width)


def binary_masks_to_instance_map_by_confidence(
    masks_hw: np.ndarray, confidences: np.ndarray
) -> np.ndarray:
    """
    Stack of ``n`` boolean ``(H, W)`` masks and length-``n`` confidences.

    Paints in ascending confidence so higher-confidence instances overwrite overlaps.
    Pixel values are ``1..n`` corresponding to the row index in ``masks_hw`` (matches
    ``dt_annotations_to_instance_map`` index semantics).
    """
    if masks_hw.ndim != 3:
        raise ValueError(f"masks_hw must be (n, H, W), got {masks_hw.shape}")
    n, h, w = masks_hw.shape
    if confidences.shape[0] != n:
        raise ValueError("confidences length must match number of masks")
    out = np.zeros((h, w), dtype=np.int32)
    order = np.argsort(confidences.astype(np.float64))
    for idx in order:
        m = masks_hw[idx]
        out[m] = int(idx) + 1
    return out


def segmentation_to_binary_mask(
    segmentation: list | dict, height: int, width: int
) -> np.ndarray:
    """Decode COCO polygon list or RLE dict to ``(H, W)`` bool mask."""
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
    """
    Paint each GT annotation's segmentation with ``ann[\"id\"]`` (same ids as ``build_gt_annotations``).

    Annotations are processed in ascending ``id`` so overlaps are deterministic regardless
    of list order. Later ids still overwrite earlier pixels where polygons overlap.
    """
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


def _prediction_score(record: dict[str, Any]) -> float:
    raw = record.get("score")
    if raw is None:
        return 0.0
    return float(raw)


def dt_annotations_to_instance_map(
    dt_annotations: list[dict[str, Any]], height: int, width: int
) -> np.ndarray:
    """
    Rasterize detections to ``1..N`` instance labels.

    Predictions are painted in ascending score order so higher-confidence masks win overlaps.
    """
    out = np.zeros((height, width), dtype=np.int32)
    indexed = list(enumerate(dt_annotations, start=1))
    indexed.sort(key=lambda pair: _prediction_score(pair[1]))
    for label, ann in indexed:
        seg = ann.get("segmentation")
        if seg is None or seg == [] or seg == {}:
            continue
        m = segmentation_to_binary_mask(seg, height, width)
        out[m] = label
    return out
