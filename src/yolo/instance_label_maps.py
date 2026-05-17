from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.instance_maps import (
    dt_annotations_to_instance_map,
    gt_annotations_to_instance_map,
    segmentation_to_binary_mask,
)


def read_yolo_seg_label_polygons(
    label_path: Path, height: int, width: int
) -> list[np.ndarray]:
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
    polygons = read_yolo_seg_label_polygons(label_path, height, width)
    return polygons_to_instance_map(polygons, height, width)


def binary_masks_to_instance_map_by_confidence(
    masks_hw: np.ndarray, confidences: np.ndarray
) -> np.ndarray:
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
