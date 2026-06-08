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
    """INTENT: compute_merged_view_pq with gt_prep matches the uncached PQ computation."""
    gt, pred = BUNDLE_FIXTURE_BUILDERS["split_merge"]()
    prep = gt_overlap_prep(gt)

    reference = compute_merged_view_pq(gt, pred)
    cached_gt = compute_merged_view_pq(gt, pred, gt_prep=prep)

    assert cached_gt == reference


@pytest.mark.parametrize("fixture_name", tuple(BUNDLE_FIXTURE_BUILDERS))
def test_compute_merged_view_pq_matches_bundle(fixture_name: str) -> None:
    """INTENT: compute_merged_view_pq agrees with compute_instance_metric_bundle across fixture maps."""
    gt, pred = BUNDLE_FIXTURE_BUILDERS[fixture_name]()
    _assert_pq_matches_bundle(gt, pred)
    result = compute_merged_view_pq(gt, pred)
    if fixture_name == "perfect_single":
        assert result["tp"] == 1
        assert result["num_cooccurring_pairs"] == 1
        assert result["num_pairs_above_pq_threshold"] == 1
        assert result["min_matched_iou"] == pytest.approx(1.0)
    elif fixture_name == "pq_decomposition":
        assert result["tp"] == 2
        assert result["fp"] == 1
        assert result["fn"] == 0
    elif fixture_name == "duplicate_preds":
        assert result["tp"] == 1
        assert result["fp"] == 1
    elif fixture_name == "gapped_label_ids":
        assert result["tp"] == 2
        assert result["fp"] == 1
        assert result["fn"] == 0
    elif fixture_name == "split_merge":
        assert result["num_cooccurring_pairs"] == 2
        assert result["tp"] == 1
        assert (
            result["min_matched_iou"]
            == result["max_matched_iou"]
            == result["median_matched_iou"]
        )
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
    """INTENT: compute_merged_view_pq reports correct sparse-overlap counts on a large-id scale fixture."""
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
    """INTENT: compute_merged_view_pq avoids dense GT-by-prediction IoU matrix work on scale fixtures."""
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

    def forbid_dense_greedy_scan(
        *_args: object, **_kwargs: object
    ) -> list[tuple[int, int]]:
        raise AssertionError(
            "tune-path PQ must not scan a dense GT-by-prediction IoU matrix"
        )

    monkeypatch.setattr(overlap_mod, "iou_matrix_from_overlap", forbid_dense_iou_matrix)
    monkeypatch.setattr(
        metrics_mod, "greedy_one_to_one_matches", forbid_dense_greedy_scan
    )

    result = compute_merged_view_pq(gt, pred)

    assert result["num_cooccurring_pairs"] == 4
    assert result["tp"] == 4


@pytest.mark.parametrize("fixture_name", tuple(BUNDLE_FIXTURE_BUILDERS))
def test_sparse_greedy_matching_matches_dense_semantics(fixture_name: str) -> None:
    """INTENT: greedy_one_to_one_matches_from_overlap matches dense greedy matching across bundle fixtures."""
    from common.metrics import greedy_one_to_one_matches_from_overlap

    gt, pred = BUNDLE_FIXTURE_BUILDERS[fixture_name]()
    stats = instance_overlap_stats(gt, pred)
    dense_matrix = iou_matrix_from_overlap(stats)
    dense_matches = greedy_one_to_one_matches(dense_matrix, PQ_MATCH_IOU)
    sparse_matches = greedy_one_to_one_matches_from_overlap(stats, PQ_MATCH_IOU)
    assert sparse_matches == dense_matches


def test_near_miss_counts_when_best_iou_is_positive_but_not_a_match() -> None:
    """INTENT: compute_merged_view_pq counts near-miss forensics when best IoU is positive but below threshold."""
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
    """INTENT: compute_merged_view_pq does not count disjoint false-positive predictions as near misses."""
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
    """INTENT: compute_merged_view_pq reports zero near-miss forensics when all instances match."""
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
    """INTENT: compute_merged_view_pq requires strict IoU greater than 0.5 for pairs_above_pq_threshold."""
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
    """INTENT: coerce_merged_view_pq_value and format_merged_view_pq_value handle count and float fields."""
    assert format_merged_view_pq_value("tp", 2) == "2"
    assert format_merged_view_pq_value("pq", 0.82) == "0.82000000"
    forensics_counts = (
        "num_cooccurring_pairs",
        "num_pairs_above_pq_threshold",
        "near_miss_pred_count",
        "near_miss_gt_count",
    )
    for key in forensics_counts:
        assert coerce_merged_view_pq_value(key, 3.4) == 3
        assert format_merged_view_pq_value(key, 3) == "3"

    assert coerce_merged_view_pq_value(
        "avg_best_iou_unmatched_pred", "0.41"
    ) == pytest.approx(0.41)
    assert (
        format_merged_view_pq_value("avg_best_iou_unmatched_pred", 0.41) == "0.41000000"
    )


def test_flatten_and_parse_forensics_fields_roundtrip() -> None:
    """INTENT: flatten_merged_view_pq_results_by_suffix and merged_view_pq_result_from_prefixed_columns round-trip."""
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
