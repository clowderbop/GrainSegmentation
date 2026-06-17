"""Phased watershed extraction: semantic prep, base extraction, area filter."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pytest

from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    generate_binary_structure,
)

from unet.extraction_tune_scoring import (
    WatershedParamSet,
    instance_map_for_watershed_params,
)
from unet.instance_masks import (
    WatershedConnectivity,
    build_watershed_semantic_prep,
    watershed_area_filter,
    watershed_base_extraction,
    watershed_seed_count,
)
from unet.tests.watershed_tune_grid_fixtures import (
    load_grid_from_axes,
    minimal_grid_axes,
)
from unet.watershed_tune_grid import iter_watershed_tune_param_sets


def _two_grain_semantic(height: int = 64, width: int = 64) -> np.ndarray:
    semantic = np.zeros((height, width), dtype=np.uint8)
    semantic[8:28, 8:28] = 1
    semantic[36:56, 36:56] = 1
    return semantic


def _speckle_interior_semantic(height: int = 120, width: int = 120) -> np.ndarray:
    """One large grain plus many tiny interior-class noise islands (separate CCs)."""
    semantic = np.zeros((height, width), dtype=np.uint8)
    semantic[40:80, 40:80] = 1
    for row in range(8, height - 8, 8):
        for col in range(8, width - 8, 8):
            if row + 3 <= 40 or row >= 80 or col + 3 <= 40 or col >= 80:
                semantic[row : row + 3, col : col + 3] = 1
    return semantic


def _two_grain_semantic_with_boundaries(
    height: int = 64, width: int = 64
) -> np.ndarray:
    semantic = _two_grain_semantic(height, width)
    for r0, c0, r1, c1 in ((8, 8, 28, 28), (36, 36, 56, 56)):
        semantic[r0, c0:c1] = 2
        semantic[r1 - 1, c0:c1] = 2
        semantic[r0:r1, c0] = 2
        semantic[r0:r1, c1 - 1] = 2
    return semantic


def _monolithic_watershed_reference(
    semantic: np.ndarray,
    *,
    interior_class: int = 1,
    boundary_class: int = 2,
    min_distance: int = 1,
    footprint: np.ndarray | None = None,
    exclude_border: bool = False,
    boundary_dilate_iter: int = 0,
    ridge_level: float | None = None,
    watershed_connectivity: WatershedConnectivity = 1,
    min_area_px: int = 0,
) -> np.ndarray:
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    from common.labeled_components import drop_small_components

    interior = semantic == interior_class
    boundary = semantic == boundary_class
    if not np.any(interior):
        return np.zeros_like(semantic, dtype=np.int32)

    dt = distance_transform_edt(interior)
    if ridge_level is None:
        neg_dt = -dt[interior]
        ridge_level = float(-neg_dt.min() + dt.max() + 1.0)

    elev = np.full(semantic.shape, ridge_level, dtype=np.float64)
    elev[interior] = -dt[interior].astype(np.float64)

    bd = boundary
    if boundary_dilate_iter > 0:
        struct = generate_binary_structure(semantic.ndim, 2)
        bd = binary_dilation(
            boundary, structure=struct, iterations=boundary_dilate_iter
        )
    elev[bd] = ridge_level

    interior_labels = interior.astype(np.int32)
    coordinates = peak_local_max(
        dt,
        min_distance=min_distance,
        footprint=footprint,
        labels=interior_labels,
        exclude_border=exclude_border,
    )
    markers = np.zeros(semantic.shape, dtype=np.int32)
    if coordinates.size == 0:
        raise ValueError(
            "Watershed marker detection found no local maxima despite non-empty "
            f"interior mask; min_distance={min_distance}, "
            f"exclude_border={exclude_border}"
        )
    coord_arr = np.atleast_2d(np.asarray(coordinates, dtype=np.int64))
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
    if min_area_px > 0:
        segmented = drop_small_components(segmented, min_area_px)
    return segmented


def _phased_instance_map(semantic: np.ndarray, params: WatershedParamSet) -> np.ndarray:
    prep = build_watershed_semantic_prep(semantic)
    base = watershed_base_extraction(
        prep,
        min_distance=params.min_distance,
        exclude_border=params.exclude_border,
        boundary_dilate_iter=params.boundary_dilate_iter,
        watershed_connectivity=cast(
            WatershedConnectivity, params.watershed_connectivity
        ),
        ridge_level=params.ridge_level,
    )
    return watershed_area_filter(base, params.min_area_px)


def test_watershed_semantic_prep_handles_empty_interior() -> None:
    """INTENT: watershed semantic prep returns zero base extraction for empty interior masks."""
    semantic = np.zeros((32, 32), dtype=np.uint8)
    prep = build_watershed_semantic_prep(semantic)

    assert prep.interior.shape == semantic.shape
    assert not np.any(prep.interior)
    assert prep.auto_ridge_level == 0.0

    base = watershed_base_extraction(
        prep,
        min_distance=5,
        exclude_border=False,
        boundary_dilate_iter=0,
        watershed_connectivity=1,
    )
    assert np.all(base == 0)


def test_watershed_semantic_prep_exposes_shape_and_dtype_invariants() -> None:
    """INTENT: watershed semantic prep exposes expected shape, dtype, and mask invariants for two-grain input."""
    semantic = _two_grain_semantic()
    prep = build_watershed_semantic_prep(semantic)

    assert prep.interior.shape == semantic.shape
    assert prep.boundary.shape == semantic.shape
    assert prep.distance_transform.shape == semantic.shape
    assert prep.interior.dtype == np.bool_
    assert prep.boundary.dtype == np.bool_
    assert prep.distance_transform.dtype == np.float64
    assert np.any(prep.interior)
    assert not np.any(prep.interior & prep.boundary)
    assert prep.auto_ridge_level > 0


def test_watershed_base_extraction_h_maxima_zero_matches_monolithic_reference() -> None:
    """INTENT: h_maxima=0 preserves pre-change watershed base extraction output."""
    semantic = _two_grain_semantic_with_boundaries()
    prep = build_watershed_semantic_prep(semantic)
    reference = _monolithic_watershed_reference(
        semantic,
        min_distance=5,
        exclude_border=False,
        boundary_dilate_iter=0,
        watershed_connectivity=1,
        min_area_px=0,
        ridge_level=None,
    )
    actual = watershed_base_extraction(
        prep,
        h_maxima=0,
        min_distance=5,
        exclude_border=False,
        boundary_dilate_iter=0,
        watershed_connectivity=1,
        ridge_level=None,
    )
    np.testing.assert_array_equal(actual, reference)


def test_h_maxima_suppresses_interior_speckle_seeds() -> None:
    """INTENT: positive h_maxima yields fewer watershed seeds on speckled interior DT."""
    prep = build_watershed_semantic_prep(_speckle_interior_semantic())
    seeds_unfiltered = watershed_seed_count(
        prep.distance_transform,
        prep.interior,
        h_maxima=0,
        min_distance=1,
        exclude_border=False,
    )
    seeds_filtered = watershed_seed_count(
        prep.distance_transform,
        prep.interior,
        h_maxima=4,
        min_distance=1,
        exclude_border=False,
    )
    assert seeds_unfiltered > seeds_filtered


def test_watershed_base_extraction_labels_grains_before_area_filter() -> None:
    """INTENT: watershed base extraction labels multiple grains before min-area filtering."""
    prep = build_watershed_semantic_prep(_two_grain_semantic())
    base = watershed_base_extraction(
        prep,
        min_distance=5,
        exclude_border=False,
        boundary_dilate_iter=0,
        watershed_connectivity=1,
        ridge_level=None,
    )

    assert base.dtype == np.int32
    assert base.shape == prep.interior.shape
    assert int(base.max()) >= 2
    assert np.all(base[~prep.interior] == 0)


def test_watershed_area_filter_drops_small_components_and_relabels() -> None:
    """INTENT: watershed area filter drops sub-threshold components and relabels survivors contiguously."""
    base = np.zeros((10, 10), dtype=np.int32)
    base[1:3, 1:3] = 1  # 4 px — dropped
    base[1:6, 5:8] = 2  # 15 px — kept, becomes label 1
    base[6:10, 1:5] = 5  # 16 px — kept, relabeled to 2

    filtered = watershed_area_filter(base, min_area_px=10)

    assert filtered.dtype == np.int32
    assert np.array_equal(np.unique(filtered[filtered > 0]), np.array([1, 2]))
    labels, counts = np.unique(filtered, return_counts=True)
    for label_id, count in zip(labels, counts, strict=True):
        if label_id == 0:
            continue
        assert count >= 10


@pytest.mark.parametrize(
    "params",
    [
        WatershedParamSet(5, 0, 1, 0, False, None),
        WatershedParamSet(9, 1, 2, 64, True, None),
        WatershedParamSet(11, 1, 1, 256, False, None),
    ],
)
def test_phased_extraction_matches_monolithic_for_representative_params(
    params: WatershedParamSet,
) -> None:
    """INTENT: phased watershed extraction matches monolithic reference for representative param sets."""
    semantic = _two_grain_semantic_with_boundaries()
    reference = _monolithic_watershed_reference(
        semantic,
        min_distance=params.min_distance,
        boundary_dilate_iter=params.boundary_dilate_iter,
        watershed_connectivity=cast(
            WatershedConnectivity, params.watershed_connectivity
        ),
        min_area_px=params.min_area_px,
        exclude_border=params.exclude_border,
        ridge_level=params.ridge_level,
    )
    phased = _phased_instance_map(semantic, params)
    np.testing.assert_array_equal(phased, reference)


def test_phased_extraction_matches_monolithic_for_minimal_tune_grid(
    tmp_path: Path,
) -> None:
    """INTENT: phased watershed extraction matches monolithic reference across a minimal tune grid."""
    semantic = _two_grain_semantic_with_boundaries()
    grid = load_grid_from_axes(
        tmp_path,
        minimal_grid_axes(h_maxima=[0, 4], watershed_connectivity=[1, 2]),
    )
    for params in iter_watershed_tune_param_sets(grid):
        reference = _monolithic_watershed_reference(
            semantic,
            min_distance=params.min_distance,
            boundary_dilate_iter=params.boundary_dilate_iter,
            watershed_connectivity=cast(
                WatershedConnectivity, params.watershed_connectivity
            ),
            min_area_px=params.min_area_px,
            exclude_border=params.exclude_border,
            ridge_level=params.ridge_level,
        )
        phased = _phased_instance_map(semantic, params)
        np.testing.assert_array_equal(phased, reference)


def test_instance_map_for_watershed_params_uses_phased_pipeline() -> None:
    """INTENT: instance_map_for_watershed_params delegates to the phased watershed pipeline."""
    semantic = _two_grain_semantic()
    params = WatershedParamSet(5, 0, 1, 0, False, None)
    expected = _phased_instance_map(semantic, params)
    actual = instance_map_for_watershed_params(semantic, params)
    np.testing.assert_array_equal(actual, expected)
