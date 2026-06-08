import numpy as np
from scipy.optimize import linear_sum_assignment

from common.instance_overlap import (
    OverlapStats,
    instance_ids as _instance_ids,
    instance_overlap_stats,
    iou_from_intersection,
    iou_matrix_from_overlap,
    pair_intersection_lookup,
)

IOU_THRESHOLDS_50_95 = tuple(np.arange(0.50, 1.0, 0.05))
PQ_MATCH_IOU = 0.5


def _index_for_reported_threshold(threshold: float) -> int:
    for i, t in enumerate(IOU_THRESHOLDS_50_95):
        if np.isclose(t, threshold, rtol=0.0, atol=1e-9):
            return i
    raise ValueError(
        f"threshold {threshold} not found in IOU_THRESHOLDS_50_95: {IOU_THRESHOLDS_50_95!r}"
    )


def build_instance_iou_matrix(
    true_instances: np.ndarray, pred_instances: np.ndarray
) -> tuple[np.ndarray, list[int], list[int]]:
    stats = instance_overlap_stats(true_instances, pred_instances)
    return iou_matrix_from_overlap(stats), stats.gt_ids, stats.pred_ids


def _strict_iou_exceeds(value: float, iou_threshold: float) -> bool:
    return value > iou_threshold


def _greedy_select_from_candidates(
    candidates: list[tuple[float, int, int]],
) -> list[tuple[int, int]]:
    candidates.sort(key=lambda x: -x[0])
    used_row: set[int] = set()
    used_col: set[int] = set()
    matched: list[tuple[int, int]] = []
    for _, i, j in candidates:
        if i in used_row or j in used_col:
            continue
        used_row.add(i)
        used_col.add(j)
        matched.append((i, j))
    return matched


def greedy_one_to_one_matches(
    iou_matrix: np.ndarray, iou_threshold: float
) -> list[tuple[int, int]]:
    """Greedy one-to-one matches with strict IoU > threshold (ADR 0003)."""
    if iou_matrix.size == 0:
        return []
    nt, np_ = iou_matrix.shape
    candidates: list[tuple[float, int, int]] = []
    for i in range(nt):
        for j in range(np_):
            v = float(iou_matrix[i, j])
            if _strict_iou_exceeds(v, iou_threshold):
                candidates.append((v, i, j))
    return _greedy_select_from_candidates(candidates)


def greedy_one_to_one_matches_from_overlap(
    stats: OverlapStats, iou_threshold: float
) -> list[tuple[int, int]]:
    """Greedy one-to-one matches from sparse co-occurring pairs only."""
    nt, np_ = len(stats.gt_ids), len(stats.pred_ids)
    if nt == 0 or np_ == 0:
        return []

    gt_index = {tid: i for i, tid in enumerate(stats.gt_ids)}
    pred_index = {pid: j for j, pid in enumerate(stats.pred_ids)}
    candidates: list[tuple[float, int, int]] = []
    for tid, pid, inter in zip(
        stats.pair_gt_ids,
        stats.pair_pred_ids,
        stats.pair_intersections,
        strict=True,
    ):
        gt_id, pred_id = int(tid), int(pid)
        iou = iou_from_intersection(
            float(inter), stats.gt_areas[gt_id], stats.pred_areas[pred_id]
        )
        if _strict_iou_exceeds(iou, iou_threshold):
            candidates.append((iou, gt_index[gt_id], pred_index[pred_id]))
    return _greedy_select_from_candidates(candidates)


def best_iou_per_instance_from_overlap(
    stats: OverlapStats,
) -> tuple[list[float], list[float]]:
    """Per-instance best IoU from co-occurring pairs only (zero when no overlap)."""
    nt, np_ = len(stats.gt_ids), len(stats.pred_ids)
    best_gt = [0.0] * nt
    best_pred = [0.0] * np_
    if nt == 0 or np_ == 0:
        return best_gt, best_pred

    gt_index = {tid: i for i, tid in enumerate(stats.gt_ids)}
    pred_index = {pid: j for j, pid in enumerate(stats.pred_ids)}
    for tid, pid, inter in zip(
        stats.pair_gt_ids,
        stats.pair_pred_ids,
        stats.pair_intersections,
        strict=True,
    ):
        gt_id, pred_id = int(tid), int(pid)
        iou = iou_from_intersection(
            float(inter), stats.gt_areas[gt_id], stats.pred_areas[pred_id]
        )
        i = gt_index[gt_id]
        j = pred_index[pred_id]
        if iou > best_gt[i]:
            best_gt[i] = iou
        if iou > best_pred[j]:
            best_pred[j] = iou
    return best_gt, best_pred


def precision_recall_f1_from_overlap(
    stats: OverlapStats, iou_threshold: float
) -> tuple[float, float, float]:
    nt, np_ = len(stats.gt_ids), len(stats.pred_ids)
    if nt == 0 and np_ == 0:
        return 1.0, 1.0, 1.0
    if nt == 0 or np_ == 0:
        return 0.0, 0.0, 0.0

    tp = len(greedy_one_to_one_matches_from_overlap(stats, iou_threshold))
    fp = np_ - tp
    fn = nt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def greedy_one_to_one_tp_count(iou_matrix: np.ndarray, iou_threshold: float) -> int:
    return len(greedy_one_to_one_matches(iou_matrix, iou_threshold))


def precision_recall_f1_from_iou_matrix(
    iou_matrix: np.ndarray, iou_threshold: float
) -> tuple[float, float, float]:
    nt, np_ = iou_matrix.shape
    if nt == 0 and np_ == 0:
        return 1.0, 1.0, 1.0
    if nt == 0:
        return 0.0, 0.0, 0.0
    if np_ == 0:
        return 0.0, 0.0, 0.0

    tp = greedy_one_to_one_tp_count(iou_matrix, iou_threshold)
    fp = np_ - tp
    fn = nt - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2.0 * precision * recall / (precision + recall)
    return float(precision), float(recall), float(f1)


def _prf_series_over_thresholds(
    iou_matrix: np.ndarray, thresholds: tuple[float, ...]
) -> tuple[list[float], list[float], list[float]]:
    ps: list[float] = []
    rs: list[float] = []
    fs: list[float] = []
    for t in thresholds:
        p, r, f = precision_recall_f1_from_iou_matrix(iou_matrix, t)
        ps.append(p)
        rs.append(r)
        fs.append(f)
    return ps, rs, fs


def compute_aji_plus(
    true_instances: np.ndarray,
    pred_instances: np.ndarray,
    *,
    overlap_stats: OverlapStats | None = None,
) -> float:
    """AJI+ with maximal unique GT/pred pairing (HoVer-Net-style)."""
    stats = overlap_stats or instance_overlap_stats(true_instances, pred_instances)
    true_ids = stats.gt_ids
    pred_ids = stats.pred_ids
    if not true_ids and not pred_ids:
        return 1.0
    if not true_ids or not pred_ids:
        return 0.0

    iou_matrix = iou_matrix_from_overlap(stats)
    nt, np_ = iou_matrix.shape
    if nt == 0 or np_ == 0:
        return 0.0

    intersections = pair_intersection_lookup(stats)

    paired_true, paired_pred = linear_sum_assignment(-iou_matrix)
    paired_iou = iou_matrix[paired_true, paired_pred]
    keep = paired_iou > 0.0
    paired_true = paired_true[keep]
    paired_pred = paired_pred[keep]

    overall_inter = 0.0
    overall_union = 0.0
    for i, j in zip(paired_true, paired_pred, strict=True):
        tid, pid = true_ids[i], pred_ids[j]
        inter = intersections.get((tid, pid), 0.0)
        true_area = float(stats.gt_areas[tid])
        pred_area = float(stats.pred_areas[pid])
        overall_inter += inter
        overall_union += true_area + pred_area - inter

    paired_true_ids = {true_ids[i] for i in paired_true}
    paired_pred_ids = {pred_ids[j] for j in paired_pred}
    for tid in true_ids:
        if tid not in paired_true_ids:
            overall_union += float(stats.gt_areas[tid])
    for pid in pred_ids:
        if pid not in paired_pred_ids:
            overall_union += float(stats.pred_areas[pid])

    if overall_union <= 0:
        return 0.0
    return float(overall_inter / overall_union)
