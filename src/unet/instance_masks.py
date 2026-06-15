"""Watershed instance extraction from U-Net semantic label maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    generate_binary_structure,
)

from common.labeled_components import drop_small_components
from common.semantic_instance import (
    SEMANTIC_BOUNDARY_CLASS,
    SEMANTIC_INTERIOR_CLASS,
)

WatershedConnectivity = Literal[1, 2]

RIDGE_LEVEL_OFFSET = 1.0
BOUNDARY_DILATION_CONNECTIVITY: WatershedConnectivity = 2

__all__ = [
    "WatershedConnectivity",
    "WatershedSemanticPrep",
    "build_watershed_semantic_prep",
    "watershed_area_filter",
    "watershed_base_extraction",
    "watershed_peak_coordinates",
    "watershed_seed_count",
]


@dataclass(frozen=True)
class WatershedSemanticPrep:
    interior: np.ndarray
    boundary: np.ndarray
    distance_transform: np.ndarray
    auto_ridge_level: float


def _binary_structure(ndim: int, connectivity: WatershedConnectivity) -> np.ndarray:
    if connectivity not in (1, 2):
        raise ValueError(f"connectivity must be 1 or 2, got {connectivity}")
    return generate_binary_structure(ndim, connectivity)


def _compute_auto_ridge_level(
    distance_transform: np.ndarray, interior: np.ndarray
) -> float:
    if not np.any(interior):
        return 0.0
    dt = distance_transform
    neg_dt = -dt[interior]
    return float(-neg_dt.min() + dt.max() + RIDGE_LEVEL_OFFSET)


def _distance_transform_for_peaks(
    distance_transform: np.ndarray,
    h_maxima: int,
) -> np.ndarray:
    if h_maxima < 0:
        raise ValueError(f"h_maxima must be >= 0, got {h_maxima}")
    if h_maxima == 0:
        return distance_transform
    from skimage.morphology import h_maxima as sk_h_maxima

    return sk_h_maxima(distance_transform, h_maxima)


def watershed_peak_coordinates(
    distance_transform: np.ndarray,
    interior: np.ndarray,
    *,
    h_maxima: int = 0,
    min_distance: int = 1,
    footprint: np.ndarray | None = None,
    exclude_border: bool = False,
) -> np.ndarray:
    from skimage.feature import peak_local_max

    dt_for_peaks = _distance_transform_for_peaks(distance_transform, h_maxima)
    coordinates = peak_local_max(
        dt_for_peaks,
        min_distance=min_distance,
        footprint=footprint,
        labels=interior.astype(np.int32),
        exclude_border=exclude_border,
    )
    if coordinates.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    coord_arr = np.atleast_2d(np.asarray(coordinates, dtype=np.int64))
    if coord_arr.shape[-1] != 2:
        raise ValueError(
            f"peak_local_max must yield an array with shape (n_peaks, 2), got {coord_arr.shape}"
        )
    return coord_arr


def watershed_seed_count(
    distance_transform: np.ndarray,
    interior: np.ndarray,
    *,
    h_maxima: int = 0,
    min_distance: int = 1,
    exclude_border: bool = False,
) -> int:
    coord_arr = watershed_peak_coordinates(
        distance_transform,
        interior,
        h_maxima=h_maxima,
        min_distance=min_distance,
        exclude_border=exclude_border,
    )
    return int(coord_arr.shape[0])


def build_watershed_semantic_prep(
    semantic: np.ndarray,
    *,
    interior_class: int = SEMANTIC_INTERIOR_CLASS,
    boundary_class: int = SEMANTIC_BOUNDARY_CLASS,
) -> WatershedSemanticPrep:
    if semantic.ndim != 2:
        raise ValueError(f"semantic must be 2D, got shape {semantic.shape}")
    interior = semantic == interior_class
    boundary = semantic == boundary_class
    distance_transform = distance_transform_edt(interior).astype(np.float64)
    auto_ridge_level = _compute_auto_ridge_level(distance_transform, interior)
    return WatershedSemanticPrep(
        interior=interior,
        boundary=boundary,
        distance_transform=distance_transform,
        auto_ridge_level=auto_ridge_level,
    )


def watershed_base_extraction(
    prep: WatershedSemanticPrep,
    *,
    min_distance: int = 1,
    footprint: np.ndarray | None = None,
    exclude_border: bool = False,
    boundary_dilate_iter: int = 0,
    ridge_level: float | None = None,
    watershed_connectivity: WatershedConnectivity = 1,
    h_maxima: int = 0,
) -> np.ndarray:
    from skimage.segmentation import watershed

    interior = prep.interior
    if not np.any(interior):
        return np.zeros(interior.shape, dtype=np.int32)

    dt = prep.distance_transform
    resolved_ridge = prep.auto_ridge_level if ridge_level is None else ridge_level

    elev = np.full(interior.shape, resolved_ridge, dtype=np.float64)
    elev[interior] = -dt[interior]

    bd = prep.boundary
    if boundary_dilate_iter > 0:
        struct = _binary_structure(interior.ndim, BOUNDARY_DILATION_CONNECTIVITY)
        bd = binary_dilation(
            prep.boundary, structure=struct, iterations=boundary_dilate_iter
        )
    elev[bd] = resolved_ridge

    coord_arr = watershed_peak_coordinates(
        dt,
        interior,
        h_maxima=h_maxima,
        min_distance=min_distance,
        footprint=footprint,
        exclude_border=exclude_border,
    )
    markers = np.zeros(interior.shape, dtype=np.int32)
    if coord_arr.shape[0] == 0:
        raise ValueError(
            "Watershed marker detection found no local maxima despite non-empty "
            f"interior mask; min_distance={min_distance}, "
            f"exclude_border={exclude_border}"
        )
    for i, (row, col) in enumerate(coord_arr):
        markers[int(row), int(col)] = i + 1

    ws_connectivity = max(1, int(watershed_connectivity))
    segmented = watershed(
        elev,
        markers,
        mask=interior,
        connectivity=ws_connectivity,
    ).astype(np.int32)

    segmented[interior & (segmented <= 0)] = 0
    return segmented


def watershed_area_filter(
    base_label_map: np.ndarray,
    min_area_px: int,
) -> np.ndarray:
    if min_area_px > 0:
        return drop_small_components(base_label_map, min_area_px)
    return base_label_map
