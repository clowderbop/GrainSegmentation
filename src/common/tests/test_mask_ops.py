"""Tests for common.mask_ops."""

from __future__ import annotations

import numpy as np

from common.mask_ops import instance_map_from_masks


def test_instance_map_from_masks_respects_score_order() -> None:
    """INTENT: instance_map_from_masks assigns overlapping pixels to the higher-scoring mask."""
    masks = np.zeros((2, 8, 8), dtype=np.float32)
    masks[0, 2:6, 2:6] = 1.0
    masks[1, 2:6, 2:6] = 1.0
    low_score = instance_map_from_masks(
        masks, np.array([0.2, 0.9]), height=8, width=8
    )
    assert int(low_score[3, 3]) == 2
