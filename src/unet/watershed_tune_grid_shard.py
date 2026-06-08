"""Axis-aligned partitions of watershed tune grids for parallel shard jobs."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterable

from unet.extraction_tune_scoring import WatershedParamSet
from unet.watershed_tune_grid import WatershedTuneGrid, iter_watershed_tune_param_sets


@dataclass(frozen=True)
class WatershedTuneShard:
    index: int
    min_distance: int
    boundary_dilate_iter: int


def watershed_tune_shard_count(grid: WatershedTuneGrid) -> int:
    return len(grid.min_distance) * len(grid.boundary_dilate_iter)


def iter_watershed_tune_shards(grid: WatershedTuneGrid) -> Iterable[WatershedTuneShard]:
    for index, (min_distance, boundary_dilate_iter) in enumerate(
        itertools.product(grid.min_distance, grid.boundary_dilate_iter),
        start=1,
    ):
        yield WatershedTuneShard(
            index=index,
            min_distance=int(min_distance),
            boundary_dilate_iter=int(boundary_dilate_iter),
        )


def iter_watershed_tune_param_sets_for_shard(
    grid: WatershedTuneGrid,
    shard: WatershedTuneShard,
) -> Iterable[WatershedParamSet]:
    """Yield one shard's combos in the same axis order as ``iter_watershed_tune_param_sets``."""
    for params in iter_watershed_tune_param_sets(grid):
        if (
            params.min_distance == shard.min_distance
            and params.boundary_dilate_iter == shard.boundary_dilate_iter
        ):
            yield params
