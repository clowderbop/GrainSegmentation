"""Focused behavior tests for sparse merged-view overlap statistics."""

from __future__ import annotations

import numpy as np

from common.instance_overlap import (
    GtOverlapPrep,
    OverlapStats,
    gt_overlap_prep,
    instance_overlap_stats,
)
from common.tests.merged_view_fixtures import (
    blank_map,
    bundle_fixture_duplicate_preds,
    bundle_fixture_empty_gt,
    bundle_fixture_empty_pred,
    bundle_fixture_split_merge,
    get_bundle_fixture,
    paint_box,
)


def _pair_triples(stats: OverlapStats) -> list[tuple[int, int, int]]:
    return [
        (int(gt), int(pred), int(inter))
        for gt, pred, inter in zip(
            stats.pair_gt_ids,
            stats.pair_pred_ids,
            stats.pair_intersections,
            strict=True,
        )
    ]


def test_instance_overlap_stats_pair_order_is_lexicographic_by_gt_then_pred() -> None:
    """INTENT: instance_overlap_stats returns co-occurring pairs sorted by gt id then pred id."""
    gt, pred = bundle_fixture_split_merge()
    stats = instance_overlap_stats(gt, pred)

    pairs = list(zip(stats.pair_gt_ids, stats.pair_pred_ids, strict=True))
    assert pairs == sorted(pairs, key=lambda p: (int(p[0]), int(p[1])))


def test_instance_overlap_stats_split_merge_reports_ordered_pairs_and_areas() -> None:
    """INTENT: instance_overlap_stats reports correct areas and intersections for a split-merge fixture."""
    gt, pred = bundle_fixture_split_merge()
    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == [1, 2]
    assert stats.pred_ids == [1]
    assert _pair_triples(stats) == [
        (1, 1, 10 * 10),
        (2, 1, 4 * 4),
    ]
    assert stats.gt_areas[1] == 10 * 10
    assert stats.gt_areas[2] == 10 * 10
    assert stats.pred_areas[1] == 10 * 10 + 8 * 8 + 4 * 4


def test_instance_overlap_stats_empty_prediction_maps_have_gt_areas_only() -> None:
    """INTENT: instance_overlap_stats lists gt areas and no pairs when the prediction map is empty."""
    gt, pred = bundle_fixture_empty_pred()
    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == [1, 2]
    assert stats.pred_ids == []
    assert stats.gt_areas == {1: 12 * 12, 2: 6 * 6}
    assert stats.pred_areas == {}
    assert _pair_triples(stats) == []


def test_instance_overlap_stats_empty_ground_truth_maps_have_pred_areas_only() -> None:
    """INTENT: instance_overlap_stats lists pred areas and no pairs when the ground-truth map is empty."""
    gt, pred = bundle_fixture_empty_gt()
    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == []
    assert stats.pred_ids == [1]
    assert stats.gt_areas == {}
    assert stats.pred_areas[1] == 12 * 12
    assert _pair_triples(stats) == []


def test_instance_overlap_stats_duplicate_predictions_report_all_co_occurring_pairs() -> None:
    """INTENT: instance_overlap_stats records separate intersections for each overlapping prediction."""
    gt, pred = bundle_fixture_duplicate_preds()
    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == [1]
    assert stats.pred_ids == [1, 2]
    inner = 12 * 12
    outer = 16 * 16
    ring = outer - inner
    assert stats.gt_areas[1] == outer
    assert stats.pred_areas[1] == ring
    assert stats.pred_areas[2] == inner
    assert _pair_triples(stats) == [
        (1, 1, ring),
        (1, 2, inner),
    ]


def test_instance_overlap_stats_both_empty_maps() -> None:
    """INTENT: instance_overlap_stats returns empty id lists and no pairs for blank maps."""
    gt, pred = get_bundle_fixture("both_empty")
    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == []
    assert stats.pred_ids == []
    assert stats.gt_areas == {}
    assert stats.pred_areas == {}
    assert _pair_triples(stats) == []


def test_instance_overlap_stats_perfect_single_instance_pair() -> None:
    """INTENT: instance_overlap_stats reports a full-area intersection for a perfect single-grain match."""
    gt, pred = get_bundle_fixture("perfect_single")
    stats = instance_overlap_stats(gt, pred)

    area = 16 * 16
    assert stats.gt_ids == [1]
    assert stats.pred_ids == [1]
    assert stats.gt_areas == {1: area}
    assert stats.pred_areas == {1: area}
    assert _pair_triples(stats) == [(1, 1, area)]


def test_instance_overlap_stats_gapped_label_ids_remain_sorted() -> None:
    """INTENT: instance_overlap_stats preserves non-contiguous label ids in sorted order."""
    gt = blank_map(24, 24)
    pred = blank_map(24, 24)
    paint_box(gt, 7, 2, 2, 10, 10)
    paint_box(pred, 42, 2, 2, 10, 10)

    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == [7]
    assert stats.pred_ids == [42]
    assert _pair_triples(stats) == [(7, 42, 8 * 8)]


def test_instance_overlap_stats_reports_disjoint_pred_instances_separately() -> None:
    """INTENT: instance_overlap_stats omits pairs for predictions that do not overlap any ground truth."""
    gt = blank_map(32, 32)
    pred = blank_map(32, 32)
    paint_box(gt, 1, 4, 4, 20, 20)
    paint_box(pred, 1, 4, 4, 20, 20)
    paint_box(pred, 2, 22, 22, 30, 30)

    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == [1]
    assert stats.pred_ids == [1, 2]
    assert stats.gt_areas[1] == 16 * 16
    assert stats.pred_areas[1] == 16 * 16
    assert stats.pred_areas[2] == 8 * 8
    assert _pair_triples(stats) == [(1, 1, 16 * 16)]


def test_gt_overlap_prep_captures_sorted_ids_and_areas() -> None:
    """INTENT: gt_overlap_prep caches sorted ground-truth ids and per-label areas from a label map."""
    gt = blank_map(32, 32)
    paint_box(gt, 3, 4, 4, 12, 12)
    paint_box(gt, 7, 20, 20, 28, 28)

    prep = gt_overlap_prep(gt)

    assert isinstance(prep, GtOverlapPrep)
    assert prep.gt_ids == [3, 7]
    assert prep.gt_areas == {3: 8 * 8, 7: 8 * 8}


def test_instance_overlap_stats_with_gt_prep_matches_full_extraction() -> None:
    """INTENT: instance_overlap_stats with gt_prep matches the uncached full extraction path."""
    gt, pred = bundle_fixture_split_merge()
    prep = gt_overlap_prep(gt)

    full = instance_overlap_stats(gt, pred)
    cached_gt = instance_overlap_stats(gt, pred, gt_prep=prep)

    assert cached_gt.gt_ids == full.gt_ids
    assert cached_gt.pred_ids == full.pred_ids
    assert cached_gt.gt_areas == full.gt_areas
    assert cached_gt.pred_areas == full.pred_areas
    assert _pair_triples(cached_gt) == _pair_triples(full)


def test_instance_overlap_stats_non_overlapping_instances_omit_zero_intersection_pairs() -> None:
    """INTENT: instance_overlap_stats excludes zero-intersection gt-pred pairs from the sparse pair list."""
    gt = blank_map(32, 32)
    pred = blank_map(32, 32)
    paint_box(gt, 1, 4, 4, 12, 12)
    paint_box(pred, 2, 20, 20, 28, 28)

    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == [1]
    assert stats.pred_ids == [2]
    assert _pair_triples(stats) == []
