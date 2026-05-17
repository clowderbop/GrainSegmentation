from __future__ import annotations

import re
from typing import Any

import numpy as np

PATCH_STEM_RE = re.compile(r"^region_(\d+)_y(\d+)_x(\d+)$")


def compute_starts(size: int, patch_size: int, stride: int) -> list[int]:
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if stride > patch_size:
        raise ValueError("stride must not exceed patch_size")
    if size <= patch_size:
        return [0]

    last_start = size - patch_size
    starts = list(range(0, last_start + 1, stride))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def compute_starts_tf(size: Any, patch_size: Any, stride: Any) -> Any:
    import tensorflow as tf

    size = tf.cast(size, tf.int32)
    patch_size = tf.cast(patch_size, tf.int32)
    stride = tf.cast(stride, tf.int32)

    def get_starts():
        limit = size - patch_size
        starts = tf.range(0, limit + 1, stride)
        last_start = starts[-1]
        return tf.cond(
            tf.equal(last_start, limit),
            lambda: starts,
            lambda: tf.concat([starts, [limit]], axis=0),
        )

    return tf.cond(
        size <= patch_size, lambda: tf.constant([0], dtype=tf.int32), get_starts
    )


def region_bounds(
    height: int, width: int, tile_size: int
) -> list[tuple[int, int, int, int]]:
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")

    bounds: list[tuple[int, int, int, int]] = []
    for y0 in range(0, height, tile_size):
        y1 = min(y0 + tile_size, height)
        for x0 in range(0, width, tile_size):
            x1 = min(x0 + tile_size, width)
            bounds.append((y0, y1, x0, x1))
    return bounds


def parse_region_patch_stem(name_without_ext: str) -> tuple[int, int, int]:
    match = PATCH_STEM_RE.match(name_without_ext)
    if match is None:
        raise ValueError(
            f"Expected stem like region_0123_y01234_x05678, got: {name_without_ext!r}"
        )
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def sample_origin_xy(sample_id: str) -> tuple[int, int]:
    """Return ``(x0, y0)`` for a patch stem like ``region_*_y*_x*``.

    Raises ``ValueError`` if ``sample_id`` does not match the patch naming scheme.
    Use :func:`sample_origin_xy_or_whole_image` for full-section rasters whose
    file stems are not patch ids (coordinates align at the section origin).
    """
    _, y0, x0 = parse_region_patch_stem(sample_id)
    return x0, y0


def sample_origin_xy_or_whole_image(sample_id: str) -> tuple[int, int]:
    """Patch origin from stem, or ``(0, 0)`` when the stem is not a patch id."""
    try:
        return sample_origin_xy(sample_id)
    except ValueError:
        return 0, 0


def tile_patch_bounds(
    region_idx: int,
    patch_y0: int,
    patch_x0: int,
    *,
    height: int,
    width: int,
    tile_size: int,
    patch_size: int,
) -> tuple[int, int, int, int]:
    regions = region_bounds(height, width, tile_size)
    if region_idx < 0 or region_idx >= len(regions):
        raise ValueError(
            f"region_idx {region_idx} out of range for "
            f"{len(regions)} tiles (H={height}, W={width}, tile_size={tile_size})"
        )
    ry0, ry1, rx0, rx1 = regions[region_idx]
    if patch_y0 < ry0 or patch_y0 >= ry1 or patch_x0 < rx0 or patch_x0 >= rx1:
        raise ValueError(
            f"Patch origin (y={patch_y0}, x={patch_x0}) outside "
            f"region {region_idx} bounds y=[{ry0},{ry1}), x=[{rx0},{rx1})"
        )
    patch_y1 = min(patch_y0 + patch_size, ry1)
    patch_x1 = min(patch_x0 + patch_size, rx1)
    return patch_y0, patch_y1, patch_x0, patch_x1


def extract_padded_patch_2d(
    array: np.ndarray,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    patch_size: int,
) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"Array must be 2D, got shape {array.shape}")
    patch = array[y0:y1, x0:x1]
    out = np.zeros((patch_size, patch_size), dtype=array.dtype)
    h, w = patch.shape
    out[:h, :w] = patch
    return out


def extract_padded_patch_channel_first(
    image: np.ndarray,
    patch_bounds: tuple[int, int, int, int],
    patch_size: int,
) -> np.ndarray:
    y0, y1, x0, x1 = patch_bounds
    patch = image[:, y0:y1, x0:x1]
    padded = np.zeros((image.shape[0], patch_size, patch_size), dtype=image.dtype)
    padded[:, : patch.shape[1], : patch.shape[2]] = patch
    return padded


def build_coverage_bin_ids(
    coverages_arr: np.ndarray,
    *,
    coverage_bins: int,
    group_count: int,
) -> np.ndarray:
    total = coverages_arr.shape[0]
    if total <= 0:
        raise ValueError("At least one region is required for splitting.")
    if group_count <= 0:
        raise ValueError("group_count must be > 0")

    bins = min(coverage_bins, max(1, total // group_count))
    if bins <= 1:
        return np.zeros(total, dtype=int)

    bin_edges = np.quantile(coverages_arr, np.linspace(0.0, 1.0, bins + 1))
    if np.allclose(bin_edges, bin_edges[0]):
        return np.zeros(total, dtype=int)
    return np.digitize(coverages_arr, bin_edges[1:-1], right=True)
