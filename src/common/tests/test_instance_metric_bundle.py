"""Behavior tests for the shared instance metric bundle (merged instance views)."""

from __future__ import annotations

import numpy as np
import pytest

from common.instance_metric_bundle import (
    INSTANCE_METRIC_BUNDLE_KEYS,
    compute_instance_metric_bundle,
)


def _blank_map(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.int32)


def _paint_box(
    instance_map: np.ndarray, label: int, y0: int, x0: int, y1: int, x1: int
) -> None:
    instance_map[y0:y1, x0:x1] = label


def test_perfect_single_grain_match_has_unit_pq() -> None:
    gt = _blank_map(32, 32)
    pred = _blank_map(32, 32)
    _paint_box(gt, 1, 4, 4, 20, 20)
    _paint_box(pred, 1, 4, 4, 20, 20)

    bundle = compute_instance_metric_bundle(gt, pred)

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
    gt = _blank_map(16, 16)
    pred = _blank_map(16, 16)

    bundle = compute_instance_metric_bundle(gt, pred)

    assert bundle["pq"] == pytest.approx(1.0)
    assert bundle["dq"] == pytest.approx(1.0)
    assert bundle["sq"] == pytest.approx(1.0)
    assert bundle["gt_instance_count"] == 0
    assert bundle["pred_instance_count"] == 0
    assert bundle["pred_gt_instance_ratio"] == pytest.approx(1.0)
    assert bundle["f1_iou50"] == pytest.approx(1.0)
    assert bundle["aji_plus"] == pytest.approx(1.0)


def test_empty_prediction_yields_zero_pq_and_recall() -> None:
    gt = _blank_map(24, 24)
    pred = _blank_map(24, 24)
    _paint_box(gt, 1, 2, 2, 14, 14)
    _paint_box(gt, 2, 16, 16, 22, 22)

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
    gt = _blank_map(20, 20)
    pred = _blank_map(20, 20)
    _paint_box(pred, 1, 4, 4, 16, 16)

    bundle = compute_instance_metric_bundle(gt, pred)

    assert bundle["pq"] == pytest.approx(0.0)
    assert bundle["gt_instance_count"] == 0
    assert bundle["pred_instance_count"] == 1
    assert bundle["pred_gt_instance_ratio"] == float("inf")
    assert bundle["precision_iou50"] == pytest.approx(0.0)
    assert bundle["aji_plus"] == pytest.approx(0.0)


def test_missed_grain_lowers_dq_and_recall() -> None:
    gt = _blank_map(32, 32)
    pred = _blank_map(32, 32)
    _paint_box(gt, 1, 4, 4, 14, 14)
    _paint_box(gt, 2, 18, 18, 28, 28)
    _paint_box(pred, 1, 4, 4, 14, 14)

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
    gt = _blank_map(32, 32)
    pred = _blank_map(32, 32)
    _paint_box(gt, 1, 6, 6, 22, 22)
    _paint_box(pred, 1, 6, 6, 22, 22)
    _paint_box(pred, 2, 8, 8, 20, 20)

    bundle = compute_instance_metric_bundle(gt, pred)

    assert bundle["gt_instance_count"] == 1
    assert bundle["pred_instance_count"] == 2
    assert bundle["pred_gt_instance_ratio"] == pytest.approx(2.0)
    assert bundle["precision_iou50"] == pytest.approx(0.5)
    assert bundle["recall_iou50"] == pytest.approx(1.0)
    assert bundle["dq"] == pytest.approx(1 / (1 + 0.5 * 1))
    assert bundle["sq"] == pytest.approx(0.5625)


def test_one_prediction_covering_two_grains_counts_one_match() -> None:
    """One predicted instance overlaps both GT grains; only one strict IoU>0.5 match."""
    gt = _blank_map(48, 48)
    pred = _blank_map(48, 48)
    _paint_box(gt, 1, 10, 10, 20, 20)
    _paint_box(gt, 2, 28, 28, 38, 38)
    _paint_box(pred, 1, 10, 10, 20, 20)
    _paint_box(pred, 1, 20, 20, 28, 28)
    _paint_box(pred, 1, 28, 28, 32, 32)

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
    gt = _blank_map(32, 32)
    pred = _blank_map(32, 32)
    _paint_box(gt, 1, 8, 8, 24, 24)
    _paint_box(pred, 1, 8, 8, 18, 18)

    bundle = compute_instance_metric_bundle(gt, pred)

    assert bundle["pq"] == pytest.approx(0.0)
    assert bundle["dq"] == pytest.approx(0.0)
    assert bundle["sq"] == pytest.approx(0.0)
    assert bundle["precision_iou50"] == pytest.approx(0.0)
    assert bundle["recall_iou50"] == pytest.approx(0.0)


def test_iou_exactly_equal_to_threshold_does_not_match() -> None:
    """ADR 0003: strict IoU > threshold, not >=."""
    from common.metrics import greedy_one_to_one_matches

    iou_matrix = np.array([[0.5]], dtype=np.float64)
    assert greedy_one_to_one_matches(iou_matrix, 0.5) == []

    iou_matrix[0, 0] = np.nextafter(0.5, 1.0)
    assert greedy_one_to_one_matches(iou_matrix, 0.5) == [(0, 0)]


def test_pq_decomposition_matches_hand_computed_tp_fp_fn() -> None:
    gt = _blank_map(48, 48)
    pred = _blank_map(48, 48)
    _paint_box(gt, 1, 4, 4, 18, 18)
    _paint_box(gt, 2, 28, 28, 44, 44)
    _paint_box(pred, 1, 4, 4, 18, 18)
    _paint_box(pred, 2, 28, 28, 44, 44)
    _paint_box(pred, 3, 4, 28, 18, 44)

    bundle = compute_instance_metric_bundle(gt, pred)

    tp, fp, fn = 2, 1, 0
    expected_dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    expected_sq = 1.0
    assert bundle["dq"] == pytest.approx(expected_dq)
    assert bundle["sq"] == pytest.approx(expected_sq)
    assert bundle["pq"] == pytest.approx(expected_dq * expected_sq)


def test_aji_plus_pairs_at_most_one_prediction_per_ground_truth() -> None:
    """Duplicate predictions: only one pred pairs; the other counts in the union."""
    gt = _blank_map(24, 24)
    pred = _blank_map(24, 24)
    _paint_box(gt, 1, 4, 4, 16, 16)
    _paint_box(pred, 1, 4, 4, 16, 16)
    _paint_box(pred, 2, 6, 6, 12, 12)

    bundle = compute_instance_metric_bundle(gt, pred)

    assert bundle["gt_instance_count"] == 1
    assert bundle["pred_instance_count"] == 2
    assert bundle["aji_plus"] == pytest.approx(0.6)
    assert bundle["aji_plus"] < 1.0
