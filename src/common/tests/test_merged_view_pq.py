"""Behavior tests for tune-path merged-view PQ scoring primitives."""

from __future__ import annotations

import numpy as np
import pytest

from common.instance_metric_bundle import compute_instance_metric_bundle
from common.metrics import PQ_MATCH_IOU
from common.merged_view_pq import (
    MERGED_VIEW_PQ_COUNT_KEYS,
    MERGED_VIEW_PQ_RESULT_KEYS,
    compute_merged_view_pq,
    format_merged_view_pq_value,
    instance_overlap_stats,
)
from common.tests.instance_map_fixtures import blank_map, paint_box


def test_merged_view_pq_count_keys_are_subset_of_result_keys() -> None:
    assert MERGED_VIEW_PQ_COUNT_KEYS <= frozenset(MERGED_VIEW_PQ_RESULT_KEYS)


def test_format_merged_view_pq_value_formats_counts_as_int_strings() -> None:
    assert format_merged_view_pq_value("tp", 2) == "2"
    assert format_merged_view_pq_value("pq", 0.82) == "0.82000000"


def test_instance_overlap_stats_reports_co_occurring_pairs_and_areas() -> None:
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
    assert list(zip(stats.pair_gt_ids, stats.pair_pred_ids, stats.pair_intersections)) == [
        (1, 1, 16 * 16),
    ]


def test_instance_overlap_stats_empty_maps_have_no_pairs() -> None:
    gt = blank_map(8, 8)
    pred = blank_map(8, 8)

    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == []
    assert stats.pred_ids == []
    assert stats.gt_areas == {}
    assert stats.pred_areas == {}
    assert len(stats.pair_gt_ids) == 0


def test_instance_overlap_stats_single_instance_pair() -> None:
    gt = blank_map(16, 16)
    pred = blank_map(16, 16)
    paint_box(gt, 1, 2, 2, 10, 10)
    paint_box(pred, 1, 2, 2, 10, 10)

    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == [1]
    assert stats.pred_ids == [1]
    assert len(stats.pair_gt_ids) == 1
    assert int(stats.pair_intersections[0]) == stats.gt_areas[1] == stats.pred_areas[1]


def test_instance_overlap_stats_split_merge_one_pred_two_gt() -> None:
    """One predicted instance overlaps two GT grains; sparse pairs list both intersections."""
    gt = blank_map(48, 48)
    pred = blank_map(48, 48)
    paint_box(gt, 1, 10, 10, 20, 20)
    paint_box(gt, 2, 28, 28, 38, 38)
    paint_box(pred, 1, 10, 10, 20, 20)
    paint_box(pred, 1, 20, 20, 28, 28)
    paint_box(pred, 1, 28, 28, 32, 32)

    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == [1, 2]
    assert stats.pred_ids == [1]
    pairs = sorted(
        zip(stats.pair_gt_ids, stats.pair_pred_ids, stats.pair_intersections)
    )
    assert pairs[0][0] == 1 and pairs[0][1] == 1
    assert pairs[1][0] == 2 and pairs[1][1] == 1
    assert int(pairs[0][2]) > 0 and int(pairs[1][2]) > 0
    assert stats.gt_areas[1] + stats.gt_areas[2] > int(stats.pred_areas[1])


def test_instance_overlap_stats_handles_gapped_label_ids() -> None:
    gt = blank_map(24, 24)
    pred = blank_map(24, 24)
    paint_box(gt, 7, 2, 2, 10, 10)
    paint_box(pred, 42, 2, 2, 10, 10)

    stats = instance_overlap_stats(gt, pred)

    assert stats.gt_ids == [7]
    assert stats.pred_ids == [42]
    assert stats.gt_areas[7] == 8 * 8
    assert stats.pred_areas[42] == 8 * 8
    assert list(zip(stats.pair_gt_ids, stats.pair_pred_ids, stats.pair_intersections)) == [
        (7, 42, 8 * 8),
    ]


_BUNDLE_PQ_FIELDS = (
    "pq",
    "dq",
    "sq",
    "precision_iou50",
    "recall_iou50",
    "f1_iou50",
    "gt_instance_count",
    "pred_instance_count",
    "pred_gt_instance_ratio",
)


def _assert_pq_matches_bundle(gt: np.ndarray, pred: np.ndarray) -> None:
    bundle = compute_instance_metric_bundle(gt, pred)
    result = compute_merged_view_pq(gt, pred)
    assert tuple(result.keys()) == MERGED_VIEW_PQ_RESULT_KEYS
    for key in _BUNDLE_PQ_FIELDS:
        assert result[key] == pytest.approx(bundle[key]), key
    tp = result["tp"]
    fp = result["fp"]
    fn = result["fn"]
    assert tp + fp == result["pred_instance_count"]
    assert tp + fn == result["gt_instance_count"]
    assert result["pq"] == pytest.approx(result["dq"] * result["sq"])


def test_compute_merged_view_pq_matches_bundle_on_perfect_single_grain() -> None:
    gt = blank_map(32, 32)
    pred = blank_map(32, 32)
    paint_box(gt, 1, 4, 4, 20, 20)
    paint_box(pred, 1, 4, 4, 20, 20)
    _assert_pq_matches_bundle(gt, pred)
    result = compute_merged_view_pq(gt, pred)
    assert result["tp"] == 1
    assert result["num_cooccurring_pairs"] == 1
    assert result["num_pairs_above_pq_threshold"] == 1
    assert result["min_matched_iou"] == pytest.approx(1.0)


def test_compute_merged_view_pq_matches_bundle_on_both_empty() -> None:
    _assert_pq_matches_bundle(blank_map(16, 16), blank_map(16, 16))


def test_compute_merged_view_pq_matches_bundle_on_empty_prediction() -> None:
    gt = blank_map(24, 24)
    pred = blank_map(24, 24)
    paint_box(gt, 1, 2, 2, 14, 14)
    paint_box(gt, 2, 16, 16, 22, 22)
    _assert_pq_matches_bundle(gt, pred)


def test_compute_merged_view_pq_matches_bundle_on_missed_grain() -> None:
    gt = blank_map(32, 32)
    pred = blank_map(32, 32)
    paint_box(gt, 1, 4, 4, 14, 14)
    paint_box(gt, 2, 18, 18, 28, 28)
    paint_box(pred, 1, 4, 4, 14, 14)
    _assert_pq_matches_bundle(gt, pred)


def test_compute_merged_view_pq_matches_bundle_on_empty_ground_truth() -> None:
    gt = blank_map(20, 20)
    pred = blank_map(20, 20)
    paint_box(pred, 1, 4, 4, 16, 16)
    _assert_pq_matches_bundle(gt, pred)


def test_compute_merged_view_pq_matches_bundle_on_pq_decomposition() -> None:
    gt = blank_map(48, 48)
    pred = blank_map(48, 48)
    paint_box(gt, 1, 4, 4, 18, 18)
    paint_box(gt, 2, 28, 28, 44, 44)
    paint_box(pred, 1, 4, 4, 18, 18)
    paint_box(pred, 2, 28, 28, 44, 44)
    paint_box(pred, 3, 4, 28, 18, 44)
    _assert_pq_matches_bundle(gt, pred)
    result = compute_merged_view_pq(gt, pred)
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 0


def test_compute_merged_view_pq_matches_bundle_on_duplicate_predictions() -> None:
    gt = blank_map(32, 32)
    pred = blank_map(32, 32)
    paint_box(gt, 1, 6, 6, 22, 22)
    paint_box(pred, 1, 6, 6, 22, 22)
    paint_box(pred, 2, 8, 8, 20, 20)
    _assert_pq_matches_bundle(gt, pred)
    result = compute_merged_view_pq(gt, pred)
    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["near_miss_pred_count"] >= 0


def test_compute_merged_view_pq_matches_bundle_on_split_merge_overlap() -> None:
    gt = blank_map(48, 48)
    pred = blank_map(48, 48)
    paint_box(gt, 1, 10, 10, 20, 20)
    paint_box(gt, 2, 28, 28, 38, 38)
    paint_box(pred, 1, 10, 10, 20, 20)
    paint_box(pred, 1, 20, 20, 28, 28)
    paint_box(pred, 1, 28, 28, 32, 32)
    _assert_pq_matches_bundle(gt, pred)
    result = compute_merged_view_pq(gt, pred)
    assert result["num_cooccurring_pairs"] == 2
    assert result["tp"] == 1
    assert result["min_matched_iou"] == result["max_matched_iou"] == result["median_matched_iou"]
    assert result["min_matched_iou"] == pytest.approx(5 / 9)
    assert result["sq"] == pytest.approx(5 / 9)


def test_near_miss_counts_when_best_iou_is_positive_but_not_a_match() -> None:
    gt = blank_map(32, 32)
    pred = blank_map(32, 32)
    paint_box(gt, 1, 8, 8, 24, 24)
    paint_box(pred, 1, 8, 8, 18, 18)

    result = compute_merged_view_pq(gt, pred)

    assert result["tp"] == 0
    assert result["num_cooccurring_pairs"] == 1
    assert result["num_pairs_above_pq_threshold"] == 0
    assert result["near_miss_pred_count"] == 1
    assert result["near_miss_gt_count"] == 1
    assert 0.0 < result["avg_best_iou_unmatched_pred"] < PQ_MATCH_IOU


def test_merged_view_pq_from_instance_metric_bundle_derives_match_counts() -> None:
    from common.merged_view_pq import merged_view_pq_from_instance_metric_bundle

    bundle = {
        "pq": 0.8,
        "dq": 0.9,
        "sq": 0.7,
        "precision_iou50": 0.5,
        "recall_iou50": 1.0,
        "f1_iou50": 2 / 3,
        "gt_instance_count": 2,
        "pred_instance_count": 4,
        "pred_gt_instance_ratio": 2.0,
    }
    result = merged_view_pq_from_instance_metric_bundle(bundle)

    assert result["tp"] == 2
    assert result["fp"] == 2
    assert result["fn"] == 0
    assert result["pq"] == pytest.approx(0.8)
    assert result["num_cooccurring_pairs"] == 0


def test_instance_metric_bundle_keys_unchanged() -> None:
    from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS

    assert INSTANCE_METRIC_BUNDLE_KEYS == (
        "pq",
        "dq",
        "sq",
        "precision_iou50",
        "recall_iou50",
        "f1_iou50",
        "precision_iou75",
        "recall_iou75",
        "f1_iou75",
        "mP_iou50_95",
        "mR_iou50_95",
        "mF1_iou50_95",
        "gt_instance_count",
        "pred_instance_count",
        "pred_gt_instance_ratio",
        "aji_plus",
    )
