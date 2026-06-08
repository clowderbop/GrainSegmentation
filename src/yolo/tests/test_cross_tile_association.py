"""Behavior tests for cross-tile instance association (fixture-only slice)."""

from __future__ import annotations

import numpy as np

from common.prediction_set import (
    assert_yolo_grains_non_overlapping,
    prediction_set_to_merged_instance_view,
    yolo_detection_mask_in_section,
)
from common.tests.prediction_set_fixtures import assert_instance_map_partitions_equal
from yolo.cross_tile_association import associate_tiled_proposals
from yolo.tests.cross_tile_association_fixtures import (
    adjacent_distinct_grains,
    complementary_border_partials,
    near_boundary_low_overlap_pair,
    overlapping_tile_central_vs_border,
    slice_boundary_duplicate_pair,
)


def _foreground_mask_count(prediction_set) -> int:
    height, width = prediction_set.height, prediction_set.width
    return sum(
        1
        for det in prediction_set.detections
        if yolo_detection_mask_in_section(det, height=height, width=width).any()
    )


def test_slice_boundary_duplicates_merge_to_one_grain() -> None:
    """INTENT: slice-boundary duplicate proposals associate into one non-overlapping grain."""
    proposals, height, width = slice_boundary_duplicate_pair()
    result = associate_tiled_proposals(proposals, height=height, width=width)
    assert_yolo_grains_non_overlapping(result)
    assert _foreground_mask_count(result) == 1


def test_adjacent_distinct_grains_remain_separate() -> None:
    """INTENT: adjacent distinct grains stay separate after cross-tile association."""
    proposals, height, width = adjacent_distinct_grains()
    result = associate_tiled_proposals(proposals, height=height, width=width)
    assert_yolo_grains_non_overlapping(result)
    assert _foreground_mask_count(result) == 2


def test_centrality_prefers_tile_central_mask_over_border_partial() -> None:
    """INTENT: association prefers the tile-central mask over a border partial on overlap."""
    proposals, height, width, expected_mask = overlapping_tile_central_vs_border()
    result = associate_tiled_proposals(proposals, height=height, width=width)
    assert_yolo_grains_non_overlapping(result)
    assert _foreground_mask_count(result) == 1
    merged = prediction_set_to_merged_instance_view(result)
    expected = np.zeros((height, width), dtype=np.int32)
    expected[expected_mask] = 1
    assert_instance_map_partitions_equal(merged, expected)


def test_complementary_border_partials_union_into_one_grain() -> None:
    """INTENT: complementary border partials union into one grain with the expected mask."""
    proposals, height, width, expected_mask = complementary_border_partials()
    result = associate_tiled_proposals(proposals, height=height, width=width)
    assert_yolo_grains_non_overlapping(result)
    assert _foreground_mask_count(result) == 1
    merged = prediction_set_to_merged_instance_view(result)
    expected = np.zeros((height, width), dtype=np.int32)
    expected[expected_mask] = 1
    assert_instance_map_partitions_equal(merged, expected)


def test_over_merge_guard_keeps_near_boundary_distinct_grains() -> None:
    """INTENT: over-merge guard keeps near-boundary low-overlap grains separate."""
    proposals, height, width = near_boundary_low_overlap_pair()
    result = associate_tiled_proposals(proposals, height=height, width=width)
    assert_yolo_grains_non_overlapping(result)
    assert _foreground_mask_count(result) == 2

