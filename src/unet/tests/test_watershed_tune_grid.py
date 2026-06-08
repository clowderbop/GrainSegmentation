"""Watershed tune grid YAML loader and wiring contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from common.variants import repo_root
from unet.slurm_watershed_tune import run_watershed_tuning_script_path
from unet.tune_watershed import _build_arg_parser
from unet.watershed_tune_grid import (
    WATERSHED_TUNE_GRID_CONFIG_REL,
    load_watershed_tune_grid,
    watershed_tune_candidate_count,
    watershed_tune_grid_path,
)


def _write_grid_config(path: Path, grid: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump({"grid": grid}), encoding="utf-8")


def test_watershed_tune_grid_path_defaults_to_config_yaml() -> None:
    assert watershed_tune_grid_path() == repo_root() / WATERSHED_TUNE_GRID_CONFIG_REL


def test_load_committed_watershed_tune_grid_excludes_pixel_scale_min_distance() -> None:
    grid = load_watershed_tune_grid().grid
    assert 1 not in grid.min_distance


def test_load_watershed_tune_grid_rejects_pixel_scale_min_distance(
    tmp_path: Path,
) -> None:
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


def test_committed_watershed_tune_grid_candidate_count_matches_loader() -> None:
    grid = load_watershed_tune_grid().grid
    assert watershed_tune_candidate_count(grid) == (
        len(grid.min_distance)
        * len(grid.boundary_dilate_iter)
        * len(grid.watershed_connectivity)
        * len(grid.min_area_px)
        * len(grid.exclude_border)
        * len(grid.ridge_level)
    )


def test_tune_watershed_cli_accepts_grid_config(tmp_path: Path) -> None:
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


def test_run_watershed_tuning_shell_passes_grid_config() -> None:
    text = run_watershed_tuning_script_path().read_text(encoding="utf-8")
    assert (
        f'GRID_CONFIG="${{GRID_CONFIG:-$REPO_ROOT/{WATERSHED_TUNE_GRID_CONFIG_REL}}}"'
        in text
    )
    assert "--grid-config" in text
    assert "--min-distance" not in text
