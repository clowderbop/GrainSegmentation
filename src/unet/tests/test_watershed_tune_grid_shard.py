"""Watershed tune grid axis-aligned shard partition contracts."""

from __future__ import annotations

from pathlib import Path

from unet.tests.watershed_tune_grid_fixtures import (
    load_grid_from_axes,
    minimal_grid_axes,
)
from unet.watershed_tune_grid import (
    iter_watershed_tune_param_sets,
    watershed_tune_candidate_count,
)
from unet.watershed_tune_grid_shard import (
    WatershedTuneShard,
    iter_watershed_tune_param_sets_for_shard,
    iter_watershed_tune_shards,
    watershed_tune_shard_combo_count,
    watershed_tune_shard_count,
)


def test_watershed_tune_shard_count_equals_min_distance_times_boundary_dilate_iter(
    tmp_path: Path,
) -> None:
    """INTENT: shard count equals the product of min_distance and boundary_dilate_iter axis lengths."""
    grid = load_grid_from_axes(
        tmp_path,
        minimal_grid_axes(min_distance=[5, 9], boundary_dilate_iter=[0, 1]),
    )
    assert watershed_tune_shard_count(grid) == len(grid.min_distance) * len(
        grid.boundary_dilate_iter
    )


def test_watershed_tune_shard_count_generalizes_to_custom_grid_shapes(
    tmp_path: Path,
) -> None:
    """INTENT: shard count scales with arbitrary valid min_distance and boundary_dilate_iter axes."""
    grid = load_grid_from_axes(
        tmp_path,
        minimal_grid_axes(
            min_distance=[5, 9],
            boundary_dilate_iter=[0, 1, 2],
        ),
    )
    assert watershed_tune_shard_count(grid) == 2 * 3


def test_iter_watershed_tune_shards_yields_one_based_axis_aligned_descriptors(
    tmp_path: Path,
) -> None:
    """INTENT: shard descriptors use 1-based indices and fix min_distance with boundary_dilate_iter."""
    grid = load_grid_from_axes(
        tmp_path,
        minimal_grid_axes(min_distance=[5, 9], boundary_dilate_iter=[0, 1]),
    )
    shards = list(iter_watershed_tune_shards(grid))
    assert len(shards) == watershed_tune_shard_count(grid)
    assert shards[0] == WatershedTuneShard(
        index=1,
        min_distance=grid.min_distance[0],
        boundary_dilate_iter=grid.boundary_dilate_iter[0],
    )
    assert all(
        shard.index == position for position, shard in enumerate(shards, start=1)
    )


def _shard_param_sets_union(grid):
    union: list = []
    for shard in iter_watershed_tune_shards(grid):
        union.extend(iter_watershed_tune_param_sets_for_shard(grid, shard))
    return union


def test_single_shard_axis_grid_partitions_full_grid_without_gaps_or_duplicates(
    tmp_path: Path,
) -> None:
    """INTENT: a single shard axis pair covers all tune combos exactly once."""
    grid = load_grid_from_axes(
        tmp_path,
        minimal_grid_axes(h_maxima=[0, 4], watershed_connectivity=[2]),
    )
    assert watershed_tune_shard_count(grid) == 1
    monolithic = list(iter_watershed_tune_param_sets(grid))
    assert len(monolithic) == watershed_tune_candidate_count(grid)
    assert _shard_param_sets_union(grid) == monolithic


def test_custom_grid_shards_partition_full_grid_without_gaps_or_duplicates(
    tmp_path: Path,
) -> None:
    """INTENT: shard partition generalizes to smaller custom grids without gaps or duplicates."""
    grid = load_grid_from_axes(
        tmp_path,
        minimal_grid_axes(
            boundary_dilate_iter=[0, 1],
            watershed_connectivity=[1, 2],
            min_area_px=[0, 64],
        ),
    )
    assert watershed_tune_shard_count(grid) == 2
    monolithic = list(iter_watershed_tune_param_sets(grid))
    assert len(monolithic) == watershed_tune_candidate_count(grid)
    assert _shard_param_sets_union(grid) == monolithic


def test_watershed_tune_shard_combo_count_matches_materialized_shard_length(
    tmp_path: Path,
) -> None:
    """INTENT: shard combo count helper matches iterator length without materializing twice."""
    grid = load_grid_from_axes(
        tmp_path,
        minimal_grid_axes(h_maxima=[0, 4], min_distance=[5, 9]),
    )
    for shard in iter_watershed_tune_shards(grid):
        assert watershed_tune_shard_combo_count(grid, shard) == len(
            list(iter_watershed_tune_param_sets_for_shard(grid, shard))
        )


def test_fixed_shard_axes_yield_one_combo_per_h_maxima_value(tmp_path: Path) -> None:
    """INTENT: fixed min_distance and boundary_dilate_iter yield one shard spanning h_maxima axis."""
    grid = load_grid_from_axes(
        tmp_path,
        minimal_grid_axes(h_maxima=[0, 4, 8]),
    )
    shard_sizes = [
        len(list(iter_watershed_tune_param_sets_for_shard(grid, shard)))
        for shard in iter_watershed_tune_shards(grid)
    ]
    assert shard_sizes == [len(grid.h_maxima)]
    assert sum(shard_sizes) == watershed_tune_candidate_count(grid)


def test_shard_param_order_matches_monolithic_subset_for_each_shard(
    tmp_path: Path,
) -> None:
    """INTENT: within-shard param order matches the monolithic tune-path ordering for that subset."""
    grid = load_grid_from_axes(
        tmp_path,
        minimal_grid_axes(
            min_distance=[5, 9],
            boundary_dilate_iter=[0, 1],
            h_maxima=[0, 4],
        ),
    )
    monolithic = list(iter_watershed_tune_param_sets(grid))
    for shard in iter_watershed_tune_shards(grid):
        shard_params = list(iter_watershed_tune_param_sets_for_shard(grid, shard))
        expected = [
            params
            for params in monolithic
            if params.min_distance == shard.min_distance
            and params.boundary_dilate_iter == shard.boundary_dilate_iter
        ]
        assert shard_params == expected
