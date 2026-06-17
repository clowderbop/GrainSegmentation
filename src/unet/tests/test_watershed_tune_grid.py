"""Watershed tune grid YAML loader and wiring contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from unet.tune_watershed import _build_arg_parser
from unet.extraction_tune_scoring import WatershedParamSet
from unet.tests.watershed_tune_grid_fixtures import (
    grid_path_from_axes,
    load_grid_from_axes,
    minimal_grid_axes,
    write_watershed_tune_grid_config,
)
from unet.watershed_tune_grid import (
    first_watershed_tune_param_set,
    iter_watershed_tune_param_sets,
    load_watershed_tune_grid,
    watershed_tune_candidate_count,
)


def test_load_watershed_tune_grid_round_trips_h_maxima_axis(tmp_path: Path) -> None:
    """INTENT: grid loader preserves configured h_maxima axis values from YAML."""
    axes = minimal_grid_axes(h_maxima=[8, 16, 24])
    grid = load_grid_from_axes(tmp_path, axes)
    assert grid.h_maxima == (8, 16, 24)


def test_load_watershed_tune_grid_accepts_min_distance_without_pixel_scale_one(
    tmp_path: Path,
) -> None:
    """INTENT: grid loader accepts min_distance axes that omit pixel-scale value 1."""
    grid = load_grid_from_axes(tmp_path, minimal_grid_axes(min_distance=[5, 9]))
    assert 1 not in grid.min_distance


def test_load_watershed_tune_grid_rejects_pixel_scale_min_distance(
    tmp_path: Path,
) -> None:
    """INTENT: watershed tune grid loader rejects configs that include pixel-scale min_distance 1."""
    grid_path = tmp_path / "grid.yaml"
    write_watershed_tune_grid_config(
        grid_path,
        {
            "min_distance": [1, 5],
            "h_maxima": [0],
            "boundary_dilate_iter": [0],
            "watershed_connectivity": [1],
            "min_area_px": [0],
            "exclude_border": [0],
            "ridge_level": [None],
        },
    )
    with pytest.raises(ValueError, match="omit pixel-scale value 1"):
        load_watershed_tune_grid(grid_path)


def test_watershed_tune_candidate_count_matches_product_of_configured_axes(
    tmp_path: Path,
) -> None:
    """INTENT: watershed tune candidate count equals the product of configured grid axis lengths."""
    grid = load_grid_from_axes(
        tmp_path,
        {
            "min_distance": [5, 9],
            "h_maxima": [0],
            "boundary_dilate_iter": [0, 1],
            "watershed_connectivity": [2],
            "min_area_px": [0, 64],
            "exclude_border": [0, 1],
            "ridge_level": [None],
        },
    )
    assert watershed_tune_candidate_count(grid) == 2 * 1 * 2 * 1 * 2 * 2 * 1


def test_grid_param_iteration_order_is_stable_for_csv_diffing(tmp_path: Path) -> None:
    """INTENT: param iteration order is stable and starts at the first product combo."""
    grid = load_grid_from_axes(
        tmp_path,
        minimal_grid_axes(h_maxima=[0, 4], watershed_connectivity=[2]),
    )
    expected_first = WatershedParamSet(5, 0, 2, 0, False, None, h_maxima=0)
    assert first_watershed_tune_param_set(grid) == expected_first
    ordered = list(iter_watershed_tune_param_sets(grid))
    assert len(ordered) == watershed_tune_candidate_count(grid)
    assert ordered == list(iter_watershed_tune_param_sets(grid))


def test_tune_watershed_cli_accepts_grid_config(tmp_path: Path) -> None:
    """INTENT: tune_watershed CLI accepts --grid-config to override the default watershed grid."""
    grid_path = grid_path_from_axes(tmp_path, minimal_grid_axes())
    args = _build_arg_parser().parse_args(
        [
            "--preds-dir",
            "/tmp/preds",
            "--manifest",
            "m.json",
            "--gt-gpkg",
            "gt.gpkg",
            "--output-csv",
            "out.csv",
            "--grid-config",
            str(grid_path),
        ]
    )
    assert args.grid_config == grid_path
