"""Watershed tune grid YAML loader and wiring contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from unet.tune_watershed import _build_arg_parser
from unet.extraction_tune_scoring import WatershedParamSet
from unet.watershed_tune_grid import (
    first_watershed_tune_param_set,
    iter_watershed_tune_param_sets,
    load_watershed_tune_grid,
    watershed_tune_candidate_count,
)


def _write_grid_config(path: Path, grid: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump({"grid": grid}), encoding="utf-8")


def test_load_committed_watershed_tune_grid_excludes_pixel_scale_min_distance() -> None:
    """INTENT: committed watershed tune grid omits pixel-scale min_distance value 1."""
    grid = load_watershed_tune_grid().grid
    assert 1 not in grid.min_distance


def test_load_watershed_tune_grid_rejects_pixel_scale_min_distance(
    tmp_path: Path,
) -> None:
    """INTENT: watershed tune grid loader rejects configs that include pixel-scale min_distance 1."""
    grid_path = tmp_path / "grid.yaml"
    _write_grid_config(
        grid_path,
        {
            "min_distance": [1, 5],
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
    grid_path = tmp_path / "grid.yaml"
    _write_grid_config(
        grid_path,
        {
            "min_distance": [5, 9],
            "boundary_dilate_iter": [0, 1],
            "watershed_connectivity": [2],
            "min_area_px": [0, 64],
            "exclude_border": [0, 1],
            "ridge_level": [None],
        },
    )
    grid = load_watershed_tune_grid(grid_path).grid
    assert watershed_tune_candidate_count(grid) == 2 * 2 * 1 * 2 * 2 * 1


def test_default_grid_param_iteration_order_is_stable_for_csv_diffing() -> None:
    """INTENT: default grid param iteration order is stable and starts at the committed first combo."""
    grid = load_watershed_tune_grid().grid
    assert first_watershed_tune_param_set(grid) == WatershedParamSet(
        5, 0, 1, 0, False, None
    )
    ordered = list(iter_watershed_tune_param_sets(grid))
    assert len(ordered) == 72
    assert ordered == list(iter_watershed_tune_param_sets(grid))


def test_tune_watershed_cli_accepts_grid_config(tmp_path: Path) -> None:
    """INTENT: tune_watershed CLI accepts --grid-config to override the default watershed grid."""
    grid_path = tmp_path / "grid.yaml"
    _write_grid_config(
        grid_path,
        {
            "min_distance": [5],
            "boundary_dilate_iter": [0],
            "watershed_connectivity": [1],
            "min_area_px": [0],
            "exclude_border": [0],
            "ridge_level": [None],
        },
    )
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

