"""UNet-style sample discovery and TIFF loading (shared by training and evaluation)."""

from __future__ import annotations

import os
from glob import glob
from typing import Any

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


def list_samples(
    image_dir: str,
    mask_dir: str | None,
    image_suffixes: list[str],
    mask_ext: str | None,
    mask_stem_suffix: str,
    num_inputs: int,
) -> list[dict[str, Any]]:
    if not image_suffixes:
        raise ValueError("image_suffixes must not be empty")

    img1_suffix = image_suffixes[0]
    img1_pattern = os.path.join(image_dir, f"*{img1_suffix}.*")
    img1_paths = sorted(glob(img1_pattern))
    if not img1_paths:
        raise ValueError(f"No images found for pattern: {img1_pattern}")

    samples: list[dict[str, Any]] = []
    for img1_path in img1_paths:
        base_name = os.path.basename(img1_path)
        stem, _ = os.path.splitext(base_name)
        if not stem.endswith(img1_suffix):
            continue
        base_stem = stem[: -len(img1_suffix)]
        image_paths: list[str] = []
        for idx, suffix in enumerate(image_suffixes[:num_inputs]):
            img_ext = os.path.splitext(img1_path)[1]
            img_path = os.path.join(
                os.path.dirname(img1_path), f"{base_stem}{suffix}{img_ext}"
            )
            if not os.path.exists(img_path):
                raise FileNotFoundError(
                    f"Missing image for input {idx + 1} ({suffix}): {img_path}"
                )
            image_paths.append(img_path)

        sample: dict[str, Any] = {"images": image_paths, "id": base_stem}
        if mask_dir is not None:
            mask_exts = mask_extensions(mask_ext)
            mask_path = None
            for ext in mask_exts:
                candidate = os.path.join(
                    mask_dir, f"{base_stem}{mask_stem_suffix}{ext}"
                )
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            if mask_path is None:
                raise FileNotFoundError(
                    f"Missing raster mask for {base_stem} in {mask_dir}"
                )
            sample["mask"] = mask_path

        samples.append(sample)
    return samples
