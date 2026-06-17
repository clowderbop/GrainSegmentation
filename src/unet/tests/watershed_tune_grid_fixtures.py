"""Minimal watershed tune grids for tests (tmp_path fixtures)."""

from __future__ import annotations

from pathlib import Path

import yaml

from unet.watershed_tune_grid import WatershedTuneGrid, load_watershed_tune_grid


def write_watershed_tune_grid_config(path: Path, grid: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump({"grid": grid}), encoding="utf-8")


def minimal_grid_axes(**overrides: object) -> dict[str, object]:
    axes: dict[str, object] = {
        "min_distance": [5],
        "h_maxima": [0],
        "boundary_dilate_iter": [0],
        "watershed_connectivity": [1],
        "min_area_px": [0],
        "exclude_border": [0],
        "ridge_level": [None],
    }
    axes.update(overrides)
    return axes


def load_grid_from_axes(tmp_path: Path, axes: dict[str, object]) -> WatershedTuneGrid:
    grid_path = tmp_path / "grid.yaml"
    write_watershed_tune_grid_config(grid_path, axes)
    return load_watershed_tune_grid(grid_path).grid


def grid_path_from_axes(tmp_path: Path, axes: dict[str, object]) -> Path:
    grid_path = tmp_path / "grid.yaml"
    write_watershed_tune_grid_config(grid_path, axes)
    return grid_path
