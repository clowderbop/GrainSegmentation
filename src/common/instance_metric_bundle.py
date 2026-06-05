"""Shared instance metric bundle for merged instance view evaluation."""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from common.instance_pq_core import (
    pq_from_match_counts,
    pred_gt_instance_ratio as compute_pred_gt_instance_ratio,
)
from common.metrics import (
    IOU_THRESHOLDS_50_95,
    PQ_MATCH_IOU,
    _index_for_reported_threshold,
    _instance_ids,
    build_instance_iou_matrix,
    compute_aji_plus,
    greedy_one_to_one_matches,
    precision_recall_f1_from_iou_matrix,
)

INSTANCE_METRIC_BUNDLE_KEYS: tuple[str, ...] = (
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


class InstanceMetricBundle(TypedDict):
    pq: float
    dq: float
    sq: float
    precision_iou50: float
    recall_iou50: float
    f1_iou50: float
    precision_iou75: float
    recall_iou75: float
    f1_iou75: float
    mP_iou50_95: float
    mR_iou50_95: float
    mF1_iou50_95: float
    gt_instance_count: int
    pred_instance_count: int
    pred_gt_instance_ratio: float
    aji_plus: float


def _thresholded_prf_bundle(
    iou_matrix: np.ndarray, nt: int, np_: int
) -> dict[str, float]:
    if nt == 0 and np_ == 0:
        one = 1.0
        return {
            "precision_iou50": one,
            "recall_iou50": one,
            "f1_iou50": one,
            "precision_iou75": one,
            "recall_iou75": one,
            "f1_iou75": one,
            "mP_iou50_95": one,
            "mR_iou50_95": one,
            "mF1_iou50_95": one,
        }
    if nt == 0 or np_ == 0:
        zero = 0.0
        return {
            "precision_iou50": zero,
            "recall_iou50": zero,
            "f1_iou50": zero,
            "precision_iou75": zero,
            "recall_iou75": zero,
            "f1_iou75": zero,
            "mP_iou50_95": zero,
            "mR_iou50_95": zero,
            "mF1_iou50_95": zero,
        }

    ps: list[float] = []
    rs: list[float] = []
    fs: list[float] = []
    for threshold in IOU_THRESHOLDS_50_95:
        p, r, f = precision_recall_f1_from_iou_matrix(iou_matrix, threshold)
        ps.append(p)
        rs.append(r)
        fs.append(f)

    idx75 = _index_for_reported_threshold(0.75)
    return {
        "precision_iou50": ps[0],
        "recall_iou50": rs[0],
        "f1_iou50": fs[0],
        "precision_iou75": ps[idx75],
        "recall_iou75": rs[idx75],
        "f1_iou75": fs[idx75],
        "mP_iou50_95": float(np.mean(ps)),
        "mR_iou50_95": float(np.mean(rs)),
        "mF1_iou50_95": float(np.mean(fs)),
    }


def compute_instance_metric_bundle(
    true_instances: np.ndarray, pred_instances: np.ndarray
) -> InstanceMetricBundle:
    """Compute the standard instance metric bundle for merged instance views.

    Use this for eval and reporting (multi-threshold PR/F1, AJI+). For tune-path
    whole-section PQ scoring only, use ``compute_merged_view_pq`` instead.
    """
    true_ids = _instance_ids(true_instances)
    pred_ids = _instance_ids(pred_instances)
    nt, np_ = len(true_ids), len(pred_ids)

    gt_instance_count = nt
    pred_instance_count = np_
    pred_gt_instance_ratio = compute_pred_gt_instance_ratio(nt, np_)

    if nt == 0 and np_ == 0:
        pq, dq, sq = 1.0, 1.0, 1.0
        prf = _thresholded_prf_bundle(np.zeros((0, 0)), 0, 0)
        aji_plus = 1.0
    else:
        iou_matrix, _, _ = build_instance_iou_matrix(true_instances, pred_instances)
        prf = _thresholded_prf_bundle(iou_matrix, nt, np_)

        matched = greedy_one_to_one_matches(iou_matrix, PQ_MATCH_IOU)
        tp = len(matched)
        fp = np_ - tp
        fn = nt - tp
        matched_ious = [float(iou_matrix[i, j]) for i, j in matched]
        pq, dq, sq = pq_from_match_counts(tp, fp, fn, matched_ious)
        aji_plus = float(compute_aji_plus(true_instances, pred_instances))

    bundle: InstanceMetricBundle = {
        "pq": float(pq),
        "dq": float(dq),
        "sq": float(sq),
        **prf,
        "gt_instance_count": gt_instance_count,
        "pred_instance_count": pred_instance_count,
        "pred_gt_instance_ratio": float(pred_gt_instance_ratio),
        "aji_plus": float(aji_plus),
    }
    assert tuple(bundle.keys()) == INSTANCE_METRIC_BUNDLE_KEYS
    return bundle
