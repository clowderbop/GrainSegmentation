"""Watershed tune grid axis-aligned shard partition contracts."""

from __future__ import annotations

from pathlib import Path

import yaml

from unet.watershed_tune_grid import (
    load_watershed_tune_grid,
    iter_watershed_tune_param_sets,
)
from unet.watershed_tune_grid_shard import (
    WatershedTuneShard,
    iter_watershed_tune_param_sets_for_shard,
    iter_watershed_tune_shards,
    watershed_tune_shard_combo_count,
    watershed_tune_shard_count,
)


def _write_grid_config(path: Path, grid: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump({"grid": grid}), encoding="utf-8")


def test_watershed_tune_shard_count_equals_min_distance_times_boundary_dilate_iter() -> (
    None
):
    """INTENT: shard count equals the product of min_distance and boundary_dilate_iter axis lengths."""
    grid = load_watershed_tune_grid().grid
    assert watershed_tune_shard_count(grid) == len(grid.min_distance) * len(
        grid.boundary_dilate_iter
    )


def test_watershed_tune_shard_count_generalizes_to_custom_grid_shapes(
    tmp_path: Path,
) -> None:
    """INTENT: shard count scales with arbitrary valid min_distance and boundary_dilate_iter axes."""
    grid_path = tmp_path / "grid.yaml"
    _write_grid_config(
        grid_path,
        {
            "min_distance": [5, 9],
            "h_maxima": [0],
            "boundary_dilate_iter": [0, 1, 2],
            "watershed_connectivity": [1],
            "min_area_px": [0],
            "exclude_border": [0],
            "ridge_level": [None],
        },
    )
    grid = load_watershed_tune_grid(grid_path).grid
    assert watershed_tune_shard_count(grid) == 2 * 3


def test_iter_watershed_tune_shards_yields_one_based_axis_aligned_descriptors() -> None:
    """INTENT: shard descriptors use 1-based indices and fix min_distance with boundary_dilate_iter."""
    grid = load_watershed_tune_grid().grid
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


def test_default_grid_shards_partition_full_grid_without_gaps_or_duplicates() -> None:
    """INTENT: default grid shards cover all tune combos exactly once."""
    grid = load_watershed_tune_grid().grid
    assert watershed_tune_shard_count(grid) == 6
    monolithic = list(iter_watershed_tune_param_sets(grid))
    assert len(monolithic) == 504
    assert _shard_param_sets_union(grid) == monolithic


def test_custom_grid_shards_partition_full_grid_without_gaps_or_duplicates(
    tmp_path: Path,
) -> None:
    """INTENT: shard partition generalizes to smaller custom grids without gaps or duplicates."""
    grid_path = tmp_path / "grid.yaml"
    _write_grid_config(
        grid_path,
        {
            "min_distance": [5],
            "h_maxima": [0],
            "boundary_dilate_iter": [0, 1],
            "watershed_connectivity": [1, 2],
            "min_area_px": [0, 64],
            "exclude_border": [0],
            "ridge_level": [None],
        },
    )
    grid = load_watershed_tune_grid(grid_path).grid
    assert watershed_tune_shard_count(grid) == 2
    monolithic = list(iter_watershed_tune_param_sets(grid))
    assert len(monolithic) == 1 * 1 * 2 * 2 * 2 * 1 * 1
    assert _shard_param_sets_union(grid) == monolithic


def test_watershed_tune_shard_combo_count_matches_materialized_shard_length() -> None:
    """INTENT: shard combo count helper matches iterator length without materializing twice."""
    grid = load_watershed_tune_grid().grid
    for shard in iter_watershed_tune_shards(grid):
        assert watershed_tune_shard_combo_count(grid, shard) == len(
            list(iter_watershed_tune_param_sets_for_shard(grid, shard))
        )


def test_default_grid_yields_six_shards_of_eighty_four_combinations_each() -> None:
    """INTENT: default grid shards align with six min_distance x boundary_dilate_iter pairs."""
    grid = load_watershed_tune_grid().grid
    shard_sizes = [
        len(list(iter_watershed_tune_param_sets_for_shard(grid, shard)))
        for shard in iter_watershed_tune_shards(grid)
    ]
    assert shard_sizes == [84] * 6


def test_shard_param_order_matches_monolithic_subset_for_each_shard() -> None:
    """INTENT: within-shard param order matches the monolithic tune-path ordering for that subset."""
    grid = load_watershed_tune_grid().grid
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
