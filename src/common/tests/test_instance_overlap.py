"""Focused behavior tests for sparse merged-view overlap statistics."""

from __future__ import annotations

import numpy as np

from common.instance_overlap import OverlapStats, instance_overlap_stats
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
    gt, pred = bundle_fixture_split_merge()
    stats = instance_overlap_stats(gt, pred)

    pairs = list(zip(stats.pair_gt_ids, stats.pair_pred_ids, strict=True))
    assert pairs == sorted(pairs, key=lambda p: (int(p[0]), int(p[1])))


def test_instance_overlap_stats_split_merge_reports_ordered_pairs_and_areas() -> None:
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
    gt, pred = bundle_fixture_empty_pred()
    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == [1, 2]
    assert stats.pred_ids == []
    assert stats.gt_areas == {1: 12 * 12, 2: 6 * 6}
    assert stats.pred_areas == {}
    assert _pair_triples(stats) == []


def test_instance_overlap_stats_empty_ground_truth_maps_have_pred_areas_only() -> None:
    gt, pred = bundle_fixture_empty_gt()
    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == []
    assert stats.pred_ids == [1]
    assert stats.gt_areas == {}
    assert stats.pred_areas[1] == 12 * 12
    assert _pair_triples(stats) == []


def test_instance_overlap_stats_duplicate_predictions_report_all_co_occurring_pairs() -> None:
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
    gt, pred = get_bundle_fixture("both_empty")
    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == []
    assert stats.pred_ids == []
    assert stats.gt_areas == {}
    assert stats.pred_areas == {}
    assert _pair_triples(stats) == []


def test_instance_overlap_stats_perfect_single_instance_pair() -> None:
    gt, pred = get_bundle_fixture("perfect_single")
    stats = instance_overlap_stats(gt, pred)

    area = 16 * 16
    assert stats.gt_ids == [1]
    assert stats.pred_ids == [1]
    assert stats.gt_areas == {1: area}
    assert stats.pred_areas == {1: area}
    assert _pair_triples(stats) == [(1, 1, area)]


def test_instance_overlap_stats_gapped_label_ids_remain_sorted() -> None:
    gt = blank_map(24, 24)
    pred = blank_map(24, 24)
    paint_box(gt, 7, 2, 2, 10, 10)
    paint_box(pred, 42, 2, 2, 10, 10)

    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == [7]
    assert stats.pred_ids == [42]
    assert _pair_triples(stats) == [(7, 42, 8 * 8)]


def test_instance_overlap_stats_reports_disjoint_pred_instances_separately() -> None:
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


def test_instance_overlap_stats_non_overlapping_instances_omit_zero_intersection_pairs() -> None:
    gt = blank_map(32, 32)
    pred = blank_map(32, 32)
    paint_box(gt, 1, 4, 4, 12, 12)
    paint_box(pred, 2, 20, 20, 28, 28)

    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == [1]
    assert stats.pred_ids == [2]
    assert _pair_triples(stats) == []
