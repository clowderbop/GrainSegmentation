"""Tune-path PQ scoring for merged instance views (sparse overlap + greedy match)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

import numpy as np

from common.instance_overlap import (
    OverlapStats,
    instance_overlap_stats,
    iou_from_intersection,
    iou_matrix_from_overlap,
)
from common.instance_pq_core import pq_from_match_counts, pred_gt_instance_ratio
from common.metrics import (
    PQ_MATCH_IOU,
    greedy_one_to_one_matches,
    precision_recall_f1_from_iou_matrix,
)

# Re-export overlap primitives for callers that import from this module.
__all__ = [
    "MERGED_VIEW_PQ_COUNT_KEYS",
    "MERGED_VIEW_PQ_RESULT_KEYS",
    "MergedViewPqResult",
    "OverlapStats",
    "compute_merged_view_pq",
    "format_merged_view_pq_value",
    "instance_overlap_stats",
    "merged_view_pq_from_instance_metric_bundle",
]

MERGED_VIEW_PQ_RESULT_KEYS: tuple[str, ...] = (
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
    "min_matched_iou",
    "max_matched_iou",
    "median_matched_iou",
    "num_cooccurring_pairs",
    "num_pairs_above_pq_threshold",
    "near_miss_pred_count",
    "near_miss_gt_count",
    "avg_best_iou_unmatched_pred",
)

MERGED_VIEW_PQ_COUNT_KEYS: frozenset[str] = frozenset(
    {
        "tp",
        "fp",
        "fn",
        "gt_instance_count",
        "pred_instance_count",
        "num_cooccurring_pairs",
        "num_pairs_above_pq_threshold",
        "near_miss_pred_count",
        "near_miss_gt_count",
    }
)
assert MERGED_VIEW_PQ_COUNT_KEYS <= frozenset(MERGED_VIEW_PQ_RESULT_KEYS)


def format_merged_view_pq_value(key: str, value: float | int) -> str:
    if key in MERGED_VIEW_PQ_COUNT_KEYS:
        return str(int(value))
    return f"{float(value):.8f}"


class MergedViewPqResult(TypedDict):
    pq: float
    dq: float
    sq: float
    tp: int
    fp: int
    fn: int
    precision_iou50: float
    recall_iou50: float
    f1_iou50: float
    gt_instance_count: int
    pred_instance_count: int
    pred_gt_instance_ratio: float
    min_matched_iou: float
    max_matched_iou: float
    median_matched_iou: float
    num_cooccurring_pairs: int
    num_pairs_above_pq_threshold: int
    near_miss_pred_count: int
    near_miss_gt_count: int
    avg_best_iou_unmatched_pred: float


def _matched_iou_spread(matched_ious: list[float]) -> tuple[float, float, float]:
    if not matched_ious:
        return 0.0, 0.0, 0.0
    arr = np.asarray(matched_ious, dtype=np.float64)
    return float(arr.min()), float(arr.max()), float(np.median(arr))


def _overlap_forensics(
    stats: OverlapStats,
    iou_matrix: np.ndarray,
    matched: list[tuple[int, int]],
) -> dict[str, int | float]:
    num_cooccurring_pairs = int(len(stats.pair_intersections))
    num_pairs_above_pq_threshold = 0
    for tid, pid, inter in zip(
        stats.pair_gt_ids,
        stats.pair_pred_ids,
        stats.pair_intersections,
        strict=True,
    ):
        iou = iou_from_intersection(
            float(inter), stats.gt_areas[int(tid)], stats.pred_areas[int(pid)]
        )
        if iou > PQ_MATCH_IOU:
            num_pairs_above_pq_threshold += 1

    nt, np_ = iou_matrix.shape
    matched_rows = {i for i, _ in matched}
    matched_cols = {j for _, j in matched}

    def _best_iou_per_row() -> list[float]:
        if nt == 0:
            return []
        return [float(iou_matrix[i, :].max()) if np_ > 0 else 0.0 for i in range(nt)]

    def _best_iou_per_col() -> list[float]:
        if np_ == 0:
            return []
        return [float(iou_matrix[:, j].max()) if nt > 0 else 0.0 for j in range(np_)]

    best_gt = _best_iou_per_row()
    best_pred = _best_iou_per_col()

    near_miss_gt_count = sum(
        1
        for i, v in enumerate(best_gt)
        if i not in matched_rows and 0.0 < v <= PQ_MATCH_IOU
    )
    near_miss_pred_count = sum(
        1
        for j, v in enumerate(best_pred)
        if j not in matched_cols and 0.0 < v <= PQ_MATCH_IOU
    )

    unmatched_pred_ious = [
        best_pred[j] for j in range(np_) if j not in matched_cols and best_pred
    ]
    avg_best_iou_unmatched_pred = (
        float(np.mean(unmatched_pred_ious)) if unmatched_pred_ious else 0.0
    )

    return {
        "num_cooccurring_pairs": num_cooccurring_pairs,
        "num_pairs_above_pq_threshold": num_pairs_above_pq_threshold,
        "near_miss_pred_count": near_miss_pred_count,
        "near_miss_gt_count": near_miss_gt_count,
        "avg_best_iou_unmatched_pred": avg_best_iou_unmatched_pred,
    }


def _tp_fp_fn_from_bundle_counts(
    bundle: Mapping[str, float | int],
) -> tuple[int, int, int]:
    gt = int(bundle["gt_instance_count"])
    pred = int(bundle["pred_instance_count"])
    if gt == 0 and pred == 0:
        return 0, 0, 0
    if gt == 0:
        return 0, pred, 0
    if pred == 0:
        return 0, 0, gt
    recall = float(bundle["recall_iou50"])
    precision = float(bundle["precision_iou50"])
    tp_from_recall = int(round(recall * gt))
    tp_from_precision = int(round(precision * pred))
    tp = tp_from_recall
    if tp_from_recall != tp_from_precision:
        tp = min(tp_from_recall, tp_from_precision, gt, pred)
    fp = pred - tp
    fn = gt - tp
    return tp, fp, fn


def merged_view_pq_from_instance_metric_bundle(
    bundle: Mapping[str, float | int],
) -> MergedViewPqResult:
    """Project a full instance metric bundle onto ``MergedViewPqResult``.

    For legacy ``run_grid_search`` rows sourced from ``evaluate_instances`` reports.
    Shared PQ/count/IoU50 fields copy verbatim; ``tp``/``fp``/``fn`` are derived from
    precision/recall and instance counts when absent. Overlap forensics and matched-IoU
    spread are zero-filled (or ``sq`` when ``tp > 0``) because reports do not carry them.
    """
    tp, fp, fn = _tp_fp_fn_from_bundle_counts(bundle)
    sq = float(bundle["sq"])
    result: MergedViewPqResult = {
        "pq": float(bundle["pq"]),
        "dq": float(bundle["dq"]),
        "sq": sq,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_iou50": float(bundle["precision_iou50"]),
        "recall_iou50": float(bundle["recall_iou50"]),
        "f1_iou50": float(bundle["f1_iou50"]),
        "gt_instance_count": int(bundle["gt_instance_count"]),
        "pred_instance_count": int(bundle["pred_instance_count"]),
        "pred_gt_instance_ratio": float(bundle["pred_gt_instance_ratio"]),
        "min_matched_iou": sq if tp > 0 else 0.0,
        "max_matched_iou": sq if tp > 0 else 0.0,
        "median_matched_iou": sq if tp > 0 else 0.0,
        "num_cooccurring_pairs": 0,
        "num_pairs_above_pq_threshold": 0,
        "near_miss_pred_count": 0,
        "near_miss_gt_count": 0,
        "avg_best_iou_unmatched_pred": 0.0,
    }
    assert tuple(result.keys()) == MERGED_VIEW_PQ_RESULT_KEYS
    return result


def compute_merged_view_pq(
    true_instances: np.ndarray, pred_instances: np.ndarray
) -> MergedViewPqResult:
    """Tune-path whole-section PQ for merged instance views.

    Use this entry point for watershed/YOLO tune scoring. For eval and reporting
    (multi-threshold PR/F1, AJI+), use ``compute_instance_metric_bundle`` instead.
    """
    stats = instance_overlap_stats(true_instances, pred_instances)
    nt, np_ = len(stats.gt_ids), len(stats.pred_ids)
    gt_instance_count = nt
    pred_instance_count = np_
    ratio = pred_gt_instance_ratio(nt, np_)

    if nt == 0 and np_ == 0:
        pq, dq, sq = 1.0, 1.0, 1.0
        tp, fp, fn = 0, 0, 0
        precision, recall, f1 = 1.0, 1.0, 1.0
        min_iou, max_iou, median_iou = 0.0, 0.0, 0.0
        forensics = {
            "num_cooccurring_pairs": 0,
            "num_pairs_above_pq_threshold": 0,
            "near_miss_pred_count": 0,
            "near_miss_gt_count": 0,
            "avg_best_iou_unmatched_pred": 0.0,
        }
    else:
        iou_matrix = iou_matrix_from_overlap(stats)
        precision, recall, f1 = precision_recall_f1_from_iou_matrix(
            iou_matrix, PQ_MATCH_IOU
        )
        matched = greedy_one_to_one_matches(iou_matrix, PQ_MATCH_IOU)
        tp = len(matched)
        fp = np_ - tp
        fn = nt - tp
        matched_ious = [float(iou_matrix[i, j]) for i, j in matched]
        pq, dq, sq = pq_from_match_counts(tp, fp, fn, matched_ious)
        min_iou, max_iou, median_iou = _matched_iou_spread(matched_ious)
        forensics = _overlap_forensics(stats, iou_matrix, matched)

    result: MergedViewPqResult = {
        "pq": float(pq),
        "dq": float(dq),
        "sq": float(sq),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision_iou50": float(precision),
        "recall_iou50": float(recall),
        "f1_iou50": float(f1),
        "gt_instance_count": gt_instance_count,
        "pred_instance_count": pred_instance_count,
        "pred_gt_instance_ratio": float(ratio),
        "min_matched_iou": min_iou,
        "max_matched_iou": max_iou,
        "median_matched_iou": median_iou,
        "num_cooccurring_pairs": int(forensics["num_cooccurring_pairs"]),
        "num_pairs_above_pq_threshold": int(forensics["num_pairs_above_pq_threshold"]),
        "near_miss_pred_count": int(forensics["near_miss_pred_count"]),
        "near_miss_gt_count": int(forensics["near_miss_gt_count"]),
        "avg_best_iou_unmatched_pred": float(forensics["avg_best_iou_unmatched_pred"]),
    }
    assert tuple(result.keys()) == MERGED_VIEW_PQ_RESULT_KEYS
    return result
