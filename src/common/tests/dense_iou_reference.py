"""Pre-sparse histogram2d reference paths for parity tests."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.instance_metric_bundle import (
    INSTANCE_METRIC_BUNDLE_KEYS,
    InstanceMetricBundle,
)
from common.instance_overlap import instance_ids
from common.instance_pq_core import (
    pq_from_match_counts,
    pred_gt_instance_ratio as compute_pred_gt_instance_ratio,
)
from common.metrics import (
    IOU_THRESHOLDS_50_95,
    PQ_MATCH_IOU,
    _index_for_reported_threshold,
    greedy_one_to_one_matches,
    precision_recall_f1_from_iou_matrix,
)


def dense_build_instance_iou_matrix(
    true_instances: np.ndarray, pred_instances: np.ndarray
) -> tuple[np.ndarray, list[int], list[int]]:
    """Histogram2d + O(n_gt × n_pred) pair loop (pre-sparse reference)."""
    true_ids = instance_ids(true_instances)
    pred_ids = instance_ids(pred_instances)
    nt, np_ = len(true_ids), len(pred_ids)
    mat = np.zeros((nt, np_), dtype=np.float64)
    if nt == 0 or np_ == 0:
        return mat, true_ids, pred_ids

    max_true = int(true_instances.max())
    max_pred = int(pred_instances.max())
    intersection_matrix = np.histogram2d(
        true_instances.flatten(),
        pred_instances.flatten(),
        bins=(max_true + 1, max_pred + 1),
        range=((0, max_true + 1), (0, max_pred + 1)),
    )[0]
    true_areas = intersection_matrix.sum(axis=1)
    pred_areas = intersection_matrix.sum(axis=0)

    for i, tid in enumerate(true_ids):
        for j, pid in enumerate(pred_ids):
            inter = float(intersection_matrix[tid, pid])
            union = float(true_areas[tid] + pred_areas[pid] - inter)
            mat[i, j] = inter / union if union > 0 else 0.0
    return mat, true_ids, pred_ids


def dense_compute_aji(true_instances: np.ndarray, pred_instances: np.ndarray) -> float:
    """Legacy AJI reference for PQ-vs-AJI selection regression tests."""
    true_id_list = instance_ids(true_instances)
    pred_id_list = instance_ids(pred_instances)
    if not true_id_list and not pred_id_list:
        return 1.0
    if not true_id_list or not pred_id_list:
        return 0.0

    max_true = int(true_instances.max())
    max_pred = int(pred_instances.max())
    intersection_matrix = np.histogram2d(
        true_instances.flatten(),
        pred_instances.flatten(),
        bins=(max_true + 1, max_pred + 1),
        range=((0, max_true + 1), (0, max_pred + 1)),
    )[0]

    true_areas = intersection_matrix.sum(axis=1)
    pred_areas = intersection_matrix.sum(axis=0)

    overall_intersection = 0.0
    overall_union = 0.0
    unassigned_pred_ids = set(pred_id_list)

    for true_id in true_id_list:
        candidate_pred_ids = sorted(unassigned_pred_ids)
        if not candidate_pred_ids:
            overall_union += true_areas[true_id]
            continue

        intersections = np.array(
            [intersection_matrix[true_id, pred_id] for pred_id in candidate_pred_ids]
        )
        if intersections.sum() == 0:
            overall_union += true_areas[true_id]
            continue

        pred_areas_subset = np.array(
            [pred_areas[pred_id] for pred_id in candidate_pred_ids]
        )
        unions = true_areas[true_id] + pred_areas_subset - intersections

        ious = intersections / np.maximum(unions, 1)
        best_idx = int(np.argmax(ious))
        best_pred_id = candidate_pred_ids[best_idx]

        if ious[best_idx] > 0:
            overall_intersection += intersections[best_idx]
            overall_union += unions[best_idx]
            unassigned_pred_ids.remove(best_pred_id)
        else:
            overall_union += true_areas[true_id]

    for pred_id in unassigned_pred_ids:
        overall_union += pred_areas[pred_id]

    return float(overall_intersection / overall_union)


def dense_compute_aji_plus(
    true_instances: np.ndarray, pred_instances: np.ndarray
) -> float:
    """AJI+ reference using histogram2d intersection lookup."""
    true_id_list = instance_ids(true_instances)
    pred_id_list = instance_ids(pred_instances)
    if not true_id_list and not pred_id_list:
        return 1.0
    if not true_id_list or not pred_id_list:
        return 0.0

    iou_matrix, _, _ = dense_build_instance_iou_matrix(true_instances, pred_instances)
    nt, np_ = iou_matrix.shape
    if nt == 0 or np_ == 0:
        return 0.0

    max_true = int(true_instances.max())
    max_pred = int(pred_instances.max())
    intersection_matrix = np.histogram2d(
        true_instances.flatten(),
        pred_instances.flatten(),
        bins=(max_true + 1, max_pred + 1),
        range=((0, max_true + 1), (0, max_pred + 1)),
    )[0]

    paired_true, paired_pred = linear_sum_assignment(-iou_matrix)
    paired_iou = iou_matrix[paired_true, paired_pred]
    keep = paired_iou > 0.0
    paired_true = paired_true[keep]
    paired_pred = paired_pred[keep]

    overall_inter = 0.0
    overall_union = 0.0
    for i, j in zip(paired_true, paired_pred, strict=True):
        tid, pid = true_id_list[i], pred_id_list[j]
        inter = float(intersection_matrix[tid, pid])
        true_area = float((true_instances == tid).sum())
        pred_area = float((pred_instances == pid).sum())
        overall_inter += inter
        overall_union += true_area + pred_area - inter

    paired_true_ids = {true_id_list[i] for i in paired_true}
    paired_pred_ids = {pred_id_list[j] for j in paired_pred}
    for tid in true_id_list:
        if tid not in paired_true_ids:
            overall_union += float((true_instances == tid).sum())
    for pid in pred_id_list:
        if pid not in paired_pred_ids:
            overall_union += float((pred_instances == pid).sum())

    if overall_union <= 0:
        return 0.0
    return float(overall_inter / overall_union)


def _thresholded_prf_bundle_dense(
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


def compute_instance_metric_bundle_dense_reference(
    true_instances: np.ndarray, pred_instances: np.ndarray
) -> InstanceMetricBundle:
    """Bundle via dense IoU matrix + histogram AJI+ (pre-sparse reference)."""
    true_ids = instance_ids(true_instances)
    pred_ids = instance_ids(pred_instances)
    nt, np_ = len(true_ids), len(pred_ids)

    gt_instance_count = nt
    pred_instance_count = np_
    pred_gt_instance_ratio = compute_pred_gt_instance_ratio(nt, np_)

    if nt == 0 and np_ == 0:
        pq, dq, sq = 1.0, 1.0, 1.0
        tp, fp, fn = 0, 0, 0
        prf = _thresholded_prf_bundle_dense(np.zeros((0, 0)), 0, 0)
        aji_plus = 1.0
    else:
        iou_matrix, _, _ = dense_build_instance_iou_matrix(
            true_instances, pred_instances
        )
        prf = _thresholded_prf_bundle_dense(iou_matrix, nt, np_)

        matched = greedy_one_to_one_matches(iou_matrix, PQ_MATCH_IOU)
        tp = len(matched)
        fp = np_ - tp
        fn = nt - tp
        matched_ious = [float(iou_matrix[i, j]) for i, j in matched]
        pq, dq, sq = pq_from_match_counts(tp, fp, fn, matched_ious)
        aji_plus = float(dense_compute_aji_plus(true_instances, pred_instances))

    bundle: InstanceMetricBundle = {
        "pq": float(pq),
        "dq": float(dq),
        "sq": float(sq),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        **prf,
        "gt_instance_count": gt_instance_count,
        "pred_instance_count": pred_instance_count,
        "pred_gt_instance_ratio": float(pred_gt_instance_ratio),
        "aji_plus": float(aji_plus),
    }
    assert tuple(bundle.keys()) == INSTANCE_METRIC_BUNDLE_KEYS
    return bundle
