import numpy as np

from evaluation.instance_masks import semantic_to_instance_label_map


IOU_THRESHOLDS_50_95 = tuple(np.arange(0.50, 1.0, 0.05))


def _index_for_reported_threshold(threshold: float) -> int:
    for i, t in enumerate(IOU_THRESHOLDS_50_95):
        if np.isclose(t, threshold, rtol=0.0, atol=1e-9):
            return i
    raise ValueError(
        f"threshold {threshold} not found in IOU_THRESHOLDS_50_95: {IOU_THRESHOLDS_50_95!r}"
    )


def get_instances(semantic_mask: np.ndarray, interior_class: int = 1):
    return semantic_to_instance_label_map(
        semantic_mask, interior_class=interior_class, connectivity=1, min_area_px=0
    )


def _instance_ids(instance_map: np.ndarray) -> list[int]:
    return sorted(int(x) for x in np.unique(instance_map) if x != 0)


def build_instance_iou_matrix(
    true_instances: np.ndarray, pred_instances: np.ndarray
) -> tuple[np.ndarray, list[int], list[int]]:
    true_ids = _instance_ids(true_instances)
    pred_ids = _instance_ids(pred_instances)
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


def greedy_one_to_one_tp_count(iou_matrix: np.ndarray, iou_threshold: float) -> int:
    if iou_matrix.size == 0:
        return 0
    nt, np_ = iou_matrix.shape
    candidates: list[tuple[float, int, int]] = []
    for i in range(nt):
        for j in range(np_):
            v = float(iou_matrix[i, j])
            if v >= iou_threshold:
                candidates.append((v, i, j))
    candidates.sort(key=lambda x: -x[0])
    used_row: set[int] = set()
    used_col: set[int] = set()
    tp = 0
    for _, i, j in candidates:
        if i in used_row or j in used_col:
            continue
        used_row.add(i)
        used_col.add(j)
        tp += 1
    return tp


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


def compute_instance_precision_recall_f1(
    true_instances: np.ndarray,
    pred_instances: np.ndarray,
    iou_threshold: float,
) -> tuple[float, float, float]:
    true_ids = _instance_ids(true_instances)
    pred_ids = _instance_ids(pred_instances)
    nt, np_ = len(true_ids), len(pred_ids)

    if nt == 0 and np_ == 0:
        return 1.0, 1.0, 1.0
    if nt == 0:
        return 0.0, 0.0, 0.0
    if np_ == 0:
        return 0.0, 0.0, 0.0

    iou_matrix, _, _ = build_instance_iou_matrix(true_instances, pred_instances)
    return precision_recall_f1_from_iou_matrix(iou_matrix, iou_threshold)


def compute_instance_prf_mean_iou_sweep(
    true_instances: np.ndarray,
    pred_instances: np.ndarray,
    thresholds: tuple[float, ...] = IOU_THRESHOLDS_50_95,
) -> tuple[float, float, float]:
    if not thresholds:
        return float("nan"), float("nan"), float("nan")

    true_ids = _instance_ids(true_instances)
    pred_ids = _instance_ids(pred_instances)
    nt, np_ = len(true_ids), len(pred_ids)

    if nt == 0 and np_ == 0:
        return 1.0, 1.0, 1.0
    if nt == 0:
        return 0.0, 0.0, 0.0
    if np_ == 0:
        return 0.0, 0.0, 0.0

    iou_matrix, _, _ = build_instance_iou_matrix(true_instances, pred_instances)
    ps: list[float] = []
    rs: list[float] = []
    fs: list[float] = []
    for t in thresholds:
        p, r, f = precision_recall_f1_from_iou_matrix(iou_matrix, t)
        ps.append(p)
        rs.append(r)
        fs.append(f)
    return float(np.mean(ps)), float(np.mean(rs)), float(np.mean(fs))


def compute_instance_metrics_dict(
    true_instances: np.ndarray, pred_instances: np.ndarray
) -> dict[str, float]:
    true_ids = _instance_ids(true_instances)
    pred_ids = _instance_ids(pred_instances)
    nt, np_ = len(true_ids), len(pred_ids)

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

    iou_matrix, _, _ = build_instance_iou_matrix(true_instances, pred_instances)
    ps: list[float] = []
    rs: list[float] = []
    fs: list[float] = []
    for t in IOU_THRESHOLDS_50_95:
        p, r, f = precision_recall_f1_from_iou_matrix(iou_matrix, t)
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


def compute_aji(true_instances: np.ndarray, pred_instances: np.ndarray):
    true_id_list = list(np.unique(true_instances))
    pred_id_list = list(np.unique(pred_instances))

    if 0 in true_id_list:
        true_id_list.remove(0)
    if 0 in pred_id_list:
        pred_id_list.remove(0)

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

    overall_intersection = 0
    overall_union = 0

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
        best_idx = np.argmax(ious)
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
