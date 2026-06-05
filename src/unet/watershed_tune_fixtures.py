"""Synthetic train whole-section masks for watershed tune tests and smoke checks."""

from __future__ import annotations

import numpy as np

TRAIN_WHOLE_SECTION_SHAPE = (10_000, 52_000)
DEFAULT_LOCAL_MASK_SHAPE = (64, 64)


def _paint_box(
    label_map: np.ndarray, instance_id: int, r0: int, c0: int, r1: int, c1: int
) -> None:
    label_map[r0:r1, c0:c1] = instance_id


def two_grain_merged_instance_view(height: int = 64, width: int = 64) -> np.ndarray:
    """Ground-truth merged instance view with two disjoint grains."""
    gt = np.zeros((height, width), dtype=np.int32)
    _paint_box(gt, 1, 8, 8, 28, 28)
    _paint_box(gt, 2, 36, 36, 56, 56)
    return gt


def cached_semantic_pred_two_grain(height: int = 64, width: int = 64) -> np.ndarray:
    """Cached semantic prediction raster aligned to ``two_grain_merged_instance_view``."""
    semantic = np.zeros((height, width), dtype=np.uint8)
    semantic[8:28, 8:28] = 1
    semantic[36:56, 36:56] = 1
    return semantic


def cached_semantic_pred_speckle_prone(height: int = 64, width: int = 64) -> np.ndarray:
    """Semantic pred with interior notch; ``min_distance=1`` yields many watershed markers."""
    semantic = cached_semantic_pred_two_grain(height, width)
    semantic[17:19, 8:28] = 0
    return semantic


def large_shape_sparse_two_grain_masks(
    *,
    height: int,
    width: int,
    local_height: int = DEFAULT_LOCAL_MASK_SHAPE[0],
    local_width: int = DEFAULT_LOCAL_MASK_SHAPE[1],
) -> tuple[np.ndarray, np.ndarray]:
    """Large declared canvas with two-grain masks embedded at the origin only."""
    if local_height > height or local_width > width:
        raise ValueError("local mask must fit inside declared geometry")
    gt = np.zeros((height, width), dtype=np.int32)
    semantic = np.zeros((height, width), dtype=np.uint8)
    gt[:local_height, :local_width] = two_grain_merged_instance_view(
        local_height, local_width
    )
    semantic[:local_height, :local_width] = cached_semantic_pred_two_grain(
        local_height, local_width
    )
    return gt, semantic
