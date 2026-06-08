"""Mask resize, rasterize, and score-merge helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from skimage.transform import resize as sk_resize

from common.instance_maps import binary_masks_to_instance_map_by_score


def resize_mask_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    resized = sk_resize(
        mask.astype(np.float32),
        (height, width),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    return resized


def resize_masks_hw(masks_hw: np.ndarray, height: int, width: int) -> np.ndarray:
    if masks_hw.ndim != 3:
        raise ValueError(f"masks_hw must be (n, H, W), got {masks_hw.shape}")
    if masks_hw.shape[0] == 0:
        return masks_hw.reshape(0, height, width)
    return np.stack(
        [
            resize_mask_nearest(masks_hw[i], height, width)
            for i in range(masks_hw.shape[0])
        ],
        axis=0,
    )


def rasterize_polygon_rings(
    canvas: np.ndarray,
    rings_xy: Sequence[np.ndarray],
    *,
    fill_value: int = 1,
) -> None:
    """Fill closed rings on a uint8/bool canvas (image or bbox-local coordinates)."""
    if not rings_xy:
        return
    import cv2

    cv2.fillPoly(canvas, list(rings_xy), fill_value)


def rasterize_coco_polygons_in_box(
    segmentation: list | list[list],
    *,
    y1: int,
    x1: int,
    y2: int,
    x2: int,
) -> np.ndarray:
    """Rasterize COCO polygon lists into a bbox-sized boolean plane."""
    box_h = y2 - y1
    box_w = x2 - x1
    if box_h <= 0 or box_w <= 0:
        return np.zeros((0, 0), dtype=bool)
    if not segmentation:
        return np.zeros((box_h, box_w), dtype=bool)
    polys = (
        [segmentation] if isinstance(segmentation[0], (int, float)) else segmentation
    )
    rings: list[np.ndarray] = []
    for poly in polys:
        if not poly:
            continue
        local = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        local = local.copy()
        local[:, 0] -= x1
        local[:, 1] -= y1
        rings.append(np.rint(local).astype(np.int32))
    canvas = np.zeros((box_h, box_w), dtype=np.uint8)
    rasterize_polygon_rings(canvas, rings, fill_value=1)
    return canvas.astype(bool)


def masks_hw_to_binary(masks_hw: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    if masks_hw.dtype in (np.float32, np.float64):
        return masks_hw > threshold
    return masks_hw.astype(bool)


def instance_map_from_masks(
    masks_hw: np.ndarray,
    scores: np.ndarray,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    if masks_hw.ndim != 3:
        raise ValueError(f"masks_hw must be (n, H, W), got {masks_hw.shape}")
    if masks_hw.shape[0] == 0:
        return np.zeros((height, width), dtype=np.int32)
    resized = resize_masks_hw(masks_hw, height, width)
    binary = masks_hw_to_binary(resized)
    return binary_masks_to_instance_map_by_score(binary, scores)


__all__ = [
    "instance_map_from_masks",
    "masks_hw_to_binary",
    "rasterize_coco_polygons_in_box",
    "rasterize_polygon_rings",
    "resize_mask_nearest",
    "resize_masks_hw",
]
