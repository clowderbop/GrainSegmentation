"""Behavior tests for tune-path merged-view PQ scoring primitives."""

from __future__ import annotations

import numpy as np
import pytest

from common.instance_metric_bundle import compute_instance_metric_bundle
from common.instance_overlap import (
    gt_overlap_prep,
    instance_overlap_stats,
    iou_matrix_from_overlap,
)
from common.metrics import PQ_MATCH_IOU, greedy_one_to_one_matches
from common.merged_view_pq import (
    MERGED_VIEW_PQ_COUNT_KEYS,
    MERGED_VIEW_PQ_RESULT_KEYS,
    coerce_merged_view_pq_value,
    compute_merged_view_pq,
    flatten_merged_view_pq_results_by_suffix,
    format_merged_view_pq_value,
    merged_view_pq_result_from_prefixed_columns,
)
from common.tests.merged_view_fixtures import (
    BUNDLE_FIXTURE_BUILDERS,
    blank_map,
    paint_box,
    scale_fixture_many_ids_few_co_occurring_pairs,
)


def test_merged_view_pq_count_keys_are_subset_of_result_keys() -> None:
    assert MERGED_VIEW_PQ_COUNT_KEYS <= frozenset(MERGED_VIEW_PQ_RESULT_KEYS)


def test_format_merged_view_pq_value_formats_counts_as_int_strings() -> None:
    assert format_merged_view_pq_value("tp", 2) == "2"
    assert format_merged_view_pq_value("pq", 0.82) == "0.82000000"


def test_instance_overlap_stats_reports_co_occurring_pairs_and_areas() -> None:
    """Regression import path: overlap primitive stays on merged_view_pq exports."""
    from common.merged_view_pq import instance_overlap_stats

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
    assert list(
        zip(stats.pair_gt_ids, stats.pair_pred_ids, stats.pair_intersections, strict=True)
    ) == [(1, 1, 16 * 16)]


_BUNDLE_PQ_FIELDS = (
    "pq",
    "dq",
    "sq",
    "tp",
    "fp",
    "fn",
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
        assert result[key] == bundle[key], key
    assert result["pq"] == pytest.approx(result["dq"] * result["sq"])


def test_compute_merged_view_pq_with_gt_prep_matches_uncached_path() -> None:
    gt, pred = BUNDLE_FIXTURE_BUILDERS["split_merge"]()
    prep = gt_overlap_prep(gt)

    reference = compute_merged_view_pq(gt, pred)
    cached_gt = compute_merged_view_pq(gt, pred, gt_prep=prep)

    assert cached_gt == reference


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


def test_compute_merged_view_pq_matches_bundle_on_gapped_label_ids() -> None:
    """Non-contiguous instance ids must not skew PQ, counts, or IoU50 P/R/F1 vs bundle."""
    gt = blank_map(48, 48)
    pred = blank_map(48, 48)
    paint_box(gt, 10, 4, 4, 18, 18)
    paint_box(gt, 50, 28, 28, 44, 44)
    paint_box(pred, 10, 4, 4, 18, 18)
    paint_box(pred, 50, 28, 28, 44, 44)
    paint_box(pred, 200, 4, 28, 18, 44)

    _assert_pq_matches_bundle(gt, pred)
    result = compute_merged_view_pq(gt, pred)
    assert result["tp"] == 2
    assert result["fp"] == 1
    assert result["fn"] == 0


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


_TUNE_SELECTION_PQ_FIELDS = (
    "pq",
    "tp",
    "fp",
    "fn",
    "gt_instance_count",
    "pred_instance_count",
    "num_cooccurring_pairs",
    "num_pairs_above_pq_threshold",
)


def test_compute_merged_view_pq_on_many_ids_few_co_occurring_pairs() -> None:
    """Scale fixture: many ids, sparse overlaps, tune-path selection fields."""
    num_gt = 120
    num_pred = 1500
    num_matched = 4
    gt, pred = scale_fixture_many_ids_few_co_occurring_pairs(
        num_gt=num_gt,
        num_pred=num_pred,
        num_matched=num_matched,
    )

    result = compute_merged_view_pq(gt, pred)

    assert tuple(result.keys()) == MERGED_VIEW_PQ_RESULT_KEYS
    assert result["gt_instance_count"] == num_gt
    assert result["pred_instance_count"] == num_pred
    assert result["num_cooccurring_pairs"] == num_matched
    assert result["num_pairs_above_pq_threshold"] == num_matched
    assert result["tp"] == num_matched
    assert result["fp"] == num_pred - num_matched
    assert result["fn"] == num_gt - num_matched
    for key in _TUNE_SELECTION_PQ_FIELDS:
        assert key in result
    assert result["pq"] == pytest.approx(result["dq"] * result["sq"])
    _assert_pq_matches_bundle(gt, pred)


def test_compute_merged_view_pq_avoids_dense_gt_by_pred_matrix_on_scale_fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: tune-path PQ must not allocate or scan nt x np_ dense IoU work."""
    import common.instance_overlap as overlap_mod
    import common.metrics as metrics_mod

    gt, pred = scale_fixture_many_ids_few_co_occurring_pairs()
    stats = instance_overlap_stats(gt, pred)
    nt, np_ = len(stats.gt_ids), len(stats.pred_ids)
    assert nt * np_ > 10_000

    def forbid_dense_iou_matrix(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError(
            "tune-path PQ must not build a dense GT-by-prediction IoU matrix"
        )

    def forbid_dense_greedy_scan(*_args: object, **_kwargs: object) -> list[tuple[int, int]]:
        raise AssertionError(
            "tune-path PQ must not scan a dense GT-by-prediction IoU matrix"
        )

    monkeypatch.setattr(overlap_mod, "iou_matrix_from_overlap", forbid_dense_iou_matrix)
    monkeypatch.setattr(metrics_mod, "greedy_one_to_one_matches", forbid_dense_greedy_scan)

    result = compute_merged_view_pq(gt, pred)

    assert result["num_cooccurring_pairs"] == 4
    assert result["tp"] == 4


@pytest.mark.parametrize("fixture_name", tuple(BUNDLE_FIXTURE_BUILDERS))
def test_sparse_greedy_matching_matches_dense_semantics(fixture_name: str) -> None:
    from common.metrics import greedy_one_to_one_matches_from_overlap

    gt, pred = BUNDLE_FIXTURE_BUILDERS[fixture_name]()
    stats = instance_overlap_stats(gt, pred)
    dense_matrix = iou_matrix_from_overlap(stats)
    dense_matches = greedy_one_to_one_matches(dense_matrix, PQ_MATCH_IOU)
    sparse_matches = greedy_one_to_one_matches_from_overlap(stats, PQ_MATCH_IOU)
    assert sparse_matches == dense_matches


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


def test_zero_overlap_prediction_does_not_count_as_near_miss() -> None:
    gt = blank_map(32, 32)
    pred = blank_map(32, 32)
    paint_box(gt, 1, 4, 4, 14, 14)
    paint_box(pred, 1, 4, 4, 14, 14)
    paint_box(pred, 2, 20, 20, 28, 28)

    result = compute_merged_view_pq(gt, pred)

    assert result["tp"] == 1
    assert result["fp"] == 1
    assert result["num_cooccurring_pairs"] == 1
    assert result["num_pairs_above_pq_threshold"] == 1
    assert result["near_miss_pred_count"] == 0
    assert result["near_miss_gt_count"] == 0
    assert result["avg_best_iou_unmatched_pred"] == 0.0


def test_matched_prediction_forensics_exclude_matched_instances() -> None:
    gt = blank_map(32, 32)
    pred = blank_map(32, 32)
    paint_box(gt, 1, 4, 4, 14, 14)
    paint_box(gt, 2, 18, 18, 28, 28)
    paint_box(pred, 1, 4, 4, 14, 14)
    paint_box(pred, 2, 18, 18, 28, 28)

    result = compute_merged_view_pq(gt, pred)

    assert result["tp"] == 2
    assert result["fp"] == 0
    assert result["fn"] == 0
    assert result["num_cooccurring_pairs"] == 2
    assert result["num_pairs_above_pq_threshold"] == 2
    assert result["near_miss_pred_count"] == 0
    assert result["near_miss_gt_count"] == 0
    assert result["avg_best_iou_unmatched_pred"] == 0.0
    assert result["min_matched_iou"] == result["max_matched_iou"] == pytest.approx(1.0)


def test_pairs_above_pq_threshold_uses_strict_iou_gt_half() -> None:
    gt = blank_map(16, 16)
    pred = blank_map(16, 16)
    paint_box(gt, 1, 0, 0, 10, 10)
    paint_box(pred, 1, 5, 0, 10, 10)

    result = compute_merged_view_pq(gt, pred)

    assert result["tp"] == 0
    assert result["num_cooccurring_pairs"] == 1
    assert result["num_pairs_above_pq_threshold"] == 0
    assert result["near_miss_pred_count"] == 1
    assert result["near_miss_gt_count"] == 1


def test_coerce_and_format_forensics_fields() -> None:
    forensics_counts = (
        "num_cooccurring_pairs",
        "num_pairs_above_pq_threshold",
        "near_miss_pred_count",
        "near_miss_gt_count",
    )
    for key in forensics_counts:
        assert coerce_merged_view_pq_value(key, 3.4) == 3
        assert format_merged_view_pq_value(key, 3) == "3"

    assert coerce_merged_view_pq_value("avg_best_iou_unmatched_pred", "0.41") == pytest.approx(
        0.41
    )
    assert format_merged_view_pq_value("avg_best_iou_unmatched_pred", 0.41) == "0.41000000"


def test_flatten_and_parse_forensics_fields_roundtrip() -> None:
    gt = blank_map(32, 32)
    pred = blank_map(32, 32)
    paint_box(gt, 1, 8, 8, 24, 24)
    paint_box(pred, 1, 8, 8, 18, 18)

    result = compute_merged_view_pq(gt, pred)
    flat = flatten_merged_view_pq_results_by_suffix({"variant_a": result})
    parsed = merged_view_pq_result_from_prefixed_columns(flat, suffix="variant_a")

    assert parsed == result
    assert parsed["num_cooccurring_pairs"] == 1
    assert parsed["near_miss_pred_count"] == 1
    assert parsed["avg_best_iou_unmatched_pred"] == pytest.approx(
        result["avg_best_iou_unmatched_pred"]
    )


def test_instance_metric_bundle_keys_unchanged() -> None:
    from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS

    assert INSTANCE_METRIC_BUNDLE_KEYS == (
        "pq",
        "dq",
        "sq",
        "tp",
        "fp",
        "fn",
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
