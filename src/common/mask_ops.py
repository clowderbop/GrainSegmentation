"""Mask resize and score-merge helpers for instance prediction sets."""

from __future__ import annotations

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
        [resize_mask_nearest(masks_hw[i], height, width) for i in range(masks_hw.shape[0])],
        axis=0,
    )


def masks_hw_to_binary(masks_hw: np.ndarray) -> np.ndarray:
    if masks_hw.dtype in (np.float32, np.float64):
        return masks_hw > 0.5
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
    "resize_mask_nearest",
    "resize_masks_hw",
]
