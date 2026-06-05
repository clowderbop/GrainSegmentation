"""Production default axes for U-Net watershed hyperparameter tuning."""

from __future__ import annotations

import itertools
from typing import TypedDict


class WatershedTuneGridAxes(TypedDict):
    min_distance: tuple[int, ...]
    boundary_dilate_iter: tuple[int, ...]
    watershed_connectivity: tuple[int, ...]
    min_area_px: tuple[int, ...]
    exclude_border: tuple[int, ...]
    ridge_level: tuple[float | None, ...]


DEFAULT_MIN_DISTANCE: tuple[int, ...] = (3, 5, 9)
DEFAULT_BOUNDARY_DILATE_ITER: tuple[int, ...] = (0, 1)
DEFAULT_WATERSHED_CONNECTIVITY: tuple[int, ...] = (1, 2)
DEFAULT_MIN_AREA_PX: tuple[int, ...] = (0, 64, 256)
DEFAULT_EXCLUDE_BORDER: tuple[int, ...] = (0, 1)
DEFAULT_RIDGE_LEVEL: tuple[float | None, ...] = (None,)


def default_watershed_tune_grid_axes() -> WatershedTuneGridAxes:
    return WatershedTuneGridAxes(
        min_distance=DEFAULT_MIN_DISTANCE,
        boundary_dilate_iter=DEFAULT_BOUNDARY_DILATE_ITER,
        watershed_connectivity=DEFAULT_WATERSHED_CONNECTIVITY,
        min_area_px=DEFAULT_MIN_AREA_PX,
        exclude_border=DEFAULT_EXCLUDE_BORDER,
        ridge_level=DEFAULT_RIDGE_LEVEL,
    )


def watershed_tune_candidate_count(axes: WatershedTuneGridAxes) -> int:
    return len(
        list(
            itertools.product(
                axes["min_distance"],
                axes["boundary_dilate_iter"],
                axes["watershed_connectivity"],
                axes["min_area_px"],
                axes["exclude_border"],
                axes["ridge_level"],
            )
        )
    )


DEFAULT_WATERSHED_TUNE_CANDIDATE_COUNT = watershed_tune_candidate_count(
    default_watershed_tune_grid_axes()
)
