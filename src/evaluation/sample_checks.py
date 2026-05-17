"""Combine image/mask sanity checks before semantic-mask evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.image_io import validate_image_mask_sample, validate_semantic_labels


def semantic_mask_after_sample_validation(
    images: list[np.ndarray],
    mask: np.ndarray,
    mask_path: str | Path,
) -> np.ndarray:
    validate_image_mask_sample(images, mask, mask_path)
    return validate_semantic_labels(mask, str(mask_path))
