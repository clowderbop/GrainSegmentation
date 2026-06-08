"""Behavior tests for the shared instance metric bundle (merged instance views)."""

from __future__ import annotations

import numpy as np
import pytest

from common.instance_metric_bundle import (
    INSTANCE_METRIC_BUNDLE_KEYS,
    compute_instance_metric_bundle,
)
from common.tests.merged_view_fixtures import (
    bundle_fixture_aji_plus_duplicates,
    bundle_fixture_both_empty,
    bundle_fixture_duplicate_preds,
    bundle_fixture_empty_gt,
    bundle_fixture_empty_pred,
    bundle_fixture_missed_grain,
    bundle_fixture_perfect_single,
    bundle_fixture_poor_mask,
    bundle_fixture_pq_decomposition,
    bundle_fixture_split_merge,
)


def test_perfect_single_grain_match_has_unit_pq() -> None:
    """INTENT: compute_instance_metric_bundle scores a perfect single-grain match with unit PQ and AJI+."""
    gt, pred = bundle_fixture_perfect_single()

    bundle = compute_instance_metric_bundle(gt, pred)

    assert bundle["tp"] == 1
    assert bundle["fp"] == 0
    assert bundle["fn"] == 0
    assert bundle["pq"] == pytest.approx(1.0)
    assert bundle["dq"] == pytest.approx(1.0)
    assert bundle["sq"] == pytest.approx(1.0)
    assert bundle["pq"] == pytest.approx(bundle["dq"] * bundle["sq"])
    assert bundle["gt_instance_count"] == 1
    assert bundle["pred_instance_count"] == 1
    assert bundle["pred_gt_instance_ratio"] == pytest.approx(1.0)
    assert bundle["f1_iou50"] == pytest.approx(1.0)
    assert bundle["aji_plus"] == pytest.approx(1.0)
    assert tuple(bundle.keys()) == INSTANCE_METRIC_BUNDLE_KEYS


def test_both_empty_maps_score_perfectly() -> None:
    """INTENT: compute_instance_metric_bundle treats mutually empty maps as perfect PQ and AJI+."""
    gt, pred = bundle_fixture_both_empty()

    bundle = compute_instance_metric_bundle(gt, pred)

    assert bundle["tp"] == 0
    assert bundle["fp"] == 0
    assert bundle["fn"] == 0
    assert bundle["pq"] == pytest.approx(1.0)
    assert bundle["dq"] == pytest.approx(1.0)
    assert bundle["sq"] == pytest.approx(1.0)
    assert bundle["gt_instance_count"] == 0
    assert bundle["pred_instance_count"] == 0
    assert bundle["pred_gt_instance_ratio"] == pytest.approx(1.0)
    assert bundle["f1_iou50"] == pytest.approx(1.0)
    assert bundle["aji_plus"] == pytest.approx(1.0)


def test_empty_prediction_yields_zero_pq_and_recall() -> None:
    """INTENT: compute_instance_metric_bundle yields zero PQ and recall when predictions are empty."""
    gt, pred = bundle_fixture_empty_pred()

    bundle = compute_instance_metric_bundle(gt, pred)

    assert bundle["pq"] == pytest.approx(0.0)
    assert bundle["dq"] == pytest.approx(0.0)
    assert bundle["sq"] == pytest.approx(0.0)
    assert bundle["gt_instance_count"] == 2
    assert bundle["pred_instance_count"] == 0
    assert bundle["pred_gt_instance_ratio"] == pytest.approx(0.0)
    assert bundle["recall_iou50"] == pytest.approx(0.0)
    assert bundle["aji_plus"] == pytest.approx(0.0)


def test_empty_ground_truth_with_predictions_scores_zero_pq() -> None:
    """INTENT: compute_instance_metric_bundle yields zero PQ and infinite pred_gt ratio for empty GT."""
    gt, pred = bundle_fixture_empty_gt()

    bundle = compute_instance_metric_bundle(gt, pred)

    assert bundle["pq"] == pytest.approx(0.0)
    assert bundle["gt_instance_count"] == 0
    assert bundle["pred_instance_count"] == 1
    assert bundle["pred_gt_instance_ratio"] == float("inf")
    assert bundle["precision_iou50"] == pytest.approx(0.0)
    assert bundle["aji_plus"] == pytest.approx(0.0)


def test_missed_grain_lowers_dq_and_recall() -> None:
    """INTENT: compute_instance_metric_bundle penalizes detection quality and recall for a missed grain."""
    gt, pred = bundle_fixture_missed_grain()

    bundle = compute_instance_metric_bundle(gt, pred)

    assert bundle["gt_instance_count"] == 2
    assert bundle["pred_instance_count"] == 1
    assert bundle["pred_gt_instance_ratio"] == pytest.approx(0.5)
    assert bundle["dq"] == pytest.approx(1 / (1 + 0.5 * 1))
    assert bundle["sq"] == pytest.approx(1.0)
    assert bundle["pq"] == pytest.approx(bundle["dq"] * bundle["sq"])
    assert bundle["recall_iou50"] == pytest.approx(0.5)
    assert 0.0 < bundle["pq"] < 1.0


def test_duplicate_predictions_penalize_precision() -> None:
    """INTENT: compute_instance_metric_bundle penalizes precision and DQ for duplicate predictions."""
    gt, pred = bundle_fixture_duplicate_preds()

    bundle = compute_instance_metric_bundle(gt, pred)

    assert bundle["gt_instance_count"] == 1
    assert bundle["pred_instance_count"] == 2
    assert bundle["pred_gt_instance_ratio"] == pytest.approx(2.0)
    assert bundle["precision_iou50"] == pytest.approx(0.5)
    assert bundle["recall_iou50"] == pytest.approx(1.0)
    assert bundle["dq"] == pytest.approx(1 / (1 + 0.5 * 1))
    assert bundle["sq"] == pytest.approx(0.5625)


def test_one_prediction_covering_two_grains_counts_one_match() -> None:
    """INTENT: compute_instance_metric_bundle counts only one IoU>0.5 match when one pred spans two grains."""
    gt, pred = bundle_fixture_split_merge()

    assert np.any((pred > 0) & (gt == 1))
    assert np.any((pred > 0) & (gt == 2))

    bundle = compute_instance_metric_bundle(gt, pred)

    assert bundle["gt_instance_count"] == 2
    assert bundle["pred_instance_count"] == 1
    assert bundle["pred_gt_instance_ratio"] == pytest.approx(0.5)
    assert bundle["recall_iou50"] == pytest.approx(0.5)
    assert bundle["precision_iou50"] == pytest.approx(1.0)
    assert bundle["dq"] == pytest.approx(1 / (1 + 0.5 * 1))
    assert bundle["sq"] == pytest.approx(5 / 9)
    assert 0.0 < bundle["pq"] < bundle["dq"]


def test_poor_mask_match_below_iou50_is_false_negative() -> None:
    """INTENT: compute_instance_metric_bundle scores zero PQ when overlap is below the IoU 0.5 threshold."""
    gt, pred = bundle_fixture_poor_mask()

    bundle = compute_instance_metric_bundle(gt, pred)

    assert bundle["pq"] == pytest.approx(0.0)
    assert bundle["dq"] == pytest.approx(0.0)
    assert bundle["sq"] == pytest.approx(0.0)
    assert bundle["precision_iou50"] == pytest.approx(0.0)
    assert bundle["recall_iou50"] == pytest.approx(0.0)


def test_pq_decomposition_matches_hand_computed_tp_fp_fn() -> None:
    """INTENT: compute_instance_metric_bundle DQ and PQ match hand-computed tp/fp/fn decomposition."""
    gt, pred = bundle_fixture_pq_decomposition()

    bundle = compute_instance_metric_bundle(gt, pred)

    tp, fp, fn = 2, 1, 0
    expected_dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    expected_sq = 1.0
    assert bundle["dq"] == pytest.approx(expected_dq)
    assert bundle["sq"] == pytest.approx(expected_sq)
    assert bundle["pq"] == pytest.approx(expected_dq * expected_sq)


def test_aji_plus_pairs_at_most_one_prediction_per_ground_truth() -> None:
    """INTENT: compute_instance_metric_bundle AJI+ pairs at most one prediction per ground-truth instance."""
    gt, pred = bundle_fixture_aji_plus_duplicates()

    bundle = compute_instance_metric_bundle(gt, pred)

    assert bundle["gt_instance_count"] == 1
    assert bundle["pred_instance_count"] == 2
    assert bundle["aji_plus"] == pytest.approx(0.6)
    assert bundle["aji_plus"] < 1.0
