"""Load U-Net watershed hyperparameter tuning grids from YAML."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

from common import yaml_validate as yv
from common.variants import repo_root
from unet.extraction_tune_scoring import WatershedParamSet

WATERSHED_TUNE_GRID_CONFIG_REL = Path("config") / "watershed_tune_grid.yaml"


@dataclass(frozen=True)
class WatershedTuneGrid:
    min_distance: tuple[int, ...]
    boundary_dilate_iter: tuple[int, ...]
    watershed_connectivity: tuple[int, ...]
    min_area_px: tuple[int, ...]
    exclude_border: tuple[int, ...]
    ridge_level: tuple[float | None, ...]


@dataclass(frozen=True)
class WatershedTuneGridSpec:
    grid: WatershedTuneGrid


def watershed_tune_grid_path(path: Path | None = None) -> Path:
    return path or (repo_root() / WATERSHED_TUNE_GRID_CONFIG_REL)


def _require_ridge_level_list(
    raw: object, *, context: str
) -> tuple[float | None, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{context} must be a non-empty list")
    out: list[float | None] = []
    for index, item in enumerate(raw):
        item_context = f"{context}[{index}]"
        if item is None:
            out.append(None)
        elif isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{item_context} must be null or a number, got {item!r}")
        else:
            out.append(float(item))
    return tuple(out)


def _validate_loaded_grid(grid: WatershedTuneGrid, *, context: str) -> None:
    if any(value < 1 for value in grid.min_distance):
        raise ValueError(f"{context}.min_distance values must be >= 1")
    if 1 in grid.min_distance:
        raise ValueError(
            f"{context}.min_distance must omit pixel-scale value 1 on train whole sections"
        )
    if any(value < 0 for value in grid.boundary_dilate_iter):
        raise ValueError(f"{context}.boundary_dilate_iter values must be >= 0")
    if any(value < 0 for value in grid.min_area_px):
        raise ValueError(f"{context}.min_area_px values must be >= 0")
    invalid_connectivity = {
        value for value in grid.watershed_connectivity if value not in (1, 2)
    }
    if invalid_connectivity:
        raise ValueError(
            f"{context}.watershed_connectivity values must be 1 or 2, "
            f"got {sorted(invalid_connectivity)}"
        )
    invalid_exclude_border = {
        value for value in grid.exclude_border if value not in (0, 1)
    }
    if invalid_exclude_border:
        raise ValueError(
            f"{context}.exclude_border values must be 0 or 1, "
            f"got {sorted(invalid_exclude_border)}"
        )


def load_watershed_tune_grid(path: Path | None = None) -> WatershedTuneGridSpec:
    resolved = watershed_tune_grid_path(path)
    with resolved.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    doc = yv.require_mapping(raw, context=str(resolved))
    grid_raw = yv.require_mapping(doc.get("grid"), context="grid")
    grid = WatershedTuneGrid(
        min_distance=yv.require_int_list(
            grid_raw.get("min_distance"), context="grid.min_distance"
        ),
        boundary_dilate_iter=yv.require_int_list(
            grid_raw.get("boundary_dilate_iter"),
            context="grid.boundary_dilate_iter",
        ),
        watershed_connectivity=yv.require_int_list(
            grid_raw.get("watershed_connectivity"),
            context="grid.watershed_connectivity",
        ),
        min_area_px=yv.require_int_list(
            grid_raw.get("min_area_px"), context="grid.min_area_px"
        ),
        exclude_border=yv.require_int_list(
            grid_raw.get("exclude_border"), context="grid.exclude_border"
        ),
        ridge_level=_require_ridge_level_list(
            grid_raw.get("ridge_level"), context="grid.ridge_level"
        ),
    )
    _validate_loaded_grid(grid, context="grid")
    return WatershedTuneGridSpec(grid=grid)


def watershed_tune_candidate_count(grid: WatershedTuneGrid) -> int:
    return len(list(iter_watershed_tune_param_sets(grid)))


def iter_watershed_tune_param_sets(
    grid: WatershedTuneGrid,
) -> Iterable[WatershedParamSet]:
    """Yield grid combos in stable ``itertools.product`` axis order for CSV rows."""
    for tup in itertools.product(
        grid.min_distance,
        grid.boundary_dilate_iter,
        grid.watershed_connectivity,
        grid.min_area_px,
        grid.exclude_border,
        grid.ridge_level,
    ):
        md, bdi, wsc, mapx, exb, ridge = tup
        yield WatershedParamSet(
            min_distance=int(md),
            boundary_dilate_iter=int(bdi),
            watershed_connectivity=int(wsc),
            min_area_px=int(mapx),
            exclude_border=bool(int(exb)),
            ridge_level=ridge,
        )


def first_watershed_tune_param_set(grid: WatershedTuneGrid) -> WatershedParamSet:
    return next(iter_watershed_tune_param_sets(grid))
