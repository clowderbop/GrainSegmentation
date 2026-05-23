"""TIFF loading helpers shared by U-Net and evaluation code."""

from __future__ import annotations

import numpy as np

from common.image_io import (
    TIFF_SUFFIXES,
    load_tiff_rgb_hwc_float,
    load_tiff_single_channel_mask,
    validate_image_mask_sample,
    validate_semantic_labels,
)


def mask_extensions(mask_ext: str | None) -> list[str]:
    if mask_ext is None:
        return [".tif", ".tiff"]
    ext = mask_ext if mask_ext.startswith(".") else f".{mask_ext}"
    if ext.lower() not in TIFF_SUFFIXES:
        raise ValueError(f"mask_ext must be .tif or .tiff, got {mask_ext!r}")
    return [ext]


def load_rgb_image(path: str) -> np.ndarray:
    """Load one 3-channel input as float HWC in [0, 1]."""
    return load_tiff_rgb_hwc_float(path)


def load_raster_mask(path: str) -> np.ndarray:
    """Load a single-channel semantic mask TIFF as int32 HxW."""
    return load_tiff_single_channel_mask(path)


def validate_loaded_sample(
    images: list[np.ndarray], mask: np.ndarray, mask_path: str
) -> None:
    validate_image_mask_sample(images, mask, mask_path)


def validate_mask_labels(mask: np.ndarray, mask_path: str) -> np.ndarray:
    return validate_semantic_labels(mask, mask_path)
