"""Sparse overlap extraction and IoU matrix construction for merged instance views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def instance_ids(instance_map: np.ndarray) -> list[int]:
    return sorted(int(x) for x in np.unique(instance_map) if x != 0)


@dataclass(frozen=True)
class GtOverlapPrep:
    """Precomputed ground-truth ids and areas for repeated PQ scoring.

    Must be built from the same merged instance view raster passed as
    ``true_instances`` to overlap extraction; the API does not verify this.
    """

    gt_ids: list[int]
    gt_areas: dict[int, int]


def gt_overlap_prep(true_instances: np.ndarray) -> GtOverlapPrep:
    gt_ids = instance_ids(true_instances)
    return GtOverlapPrep(gt_ids=gt_ids, gt_areas=_areas_by_id(true_instances, gt_ids))


@dataclass(frozen=True)
class OverlapStats:
    """Sparse overlap between merged instance views."""

    gt_ids: list[int]
    pred_ids: list[int]
    gt_areas: dict[int, int]
    pred_areas: dict[int, int]
    pair_gt_ids: np.ndarray
    pair_pred_ids: np.ndarray
    pair_intersections: np.ndarray


def _areas_by_id(instance_map: np.ndarray, ids: list[int]) -> dict[int, int]:
    if not ids:
        return {}
    max_id = max(ids)
    counts = np.bincount(instance_map.ravel(), minlength=max_id + 1)
    return {int(i): int(counts[i]) for i in ids}


def _sort_co_occurring_pairs(
    pair_gt_ids: np.ndarray,
    pair_pred_ids: np.ndarray,
    pair_intersections: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(pair_gt_ids) == 0:
        return pair_gt_ids, pair_pred_ids, pair_intersections
    order = np.lexsort((pair_pred_ids, pair_gt_ids))
    return (
        pair_gt_ids[order],
        pair_pred_ids[order],
        pair_intersections[order],
    )


def instance_overlap_stats(
    true_instances: np.ndarray,
    pred_instances: np.ndarray,
    *,
    gt_prep: GtOverlapPrep | None = None,
) -> OverlapStats:
    """Extract co-occurring (gt_id, pred_id) intersections and per-id areas in O(pixels)."""
    if gt_prep is not None:
        gt_ids = gt_prep.gt_ids
        gt_areas = gt_prep.gt_areas
    else:
        gt_ids = instance_ids(true_instances)
        gt_areas = _areas_by_id(true_instances, gt_ids)
    pred_ids = instance_ids(pred_instances)
    pred_areas = _areas_by_id(pred_instances, pred_ids)

    overlap_mask = (true_instances != 0) & (pred_instances != 0)
    if not np.any(overlap_mask):
        return OverlapStats(
            gt_ids=gt_ids,
            pred_ids=pred_ids,
            gt_areas=gt_areas,
            pred_areas=pred_areas,
            pair_gt_ids=np.array([], dtype=np.int32),
            pair_pred_ids=np.array([], dtype=np.int32),
            pair_intersections=np.array([], dtype=np.int64),
        )

    stacked = np.column_stack(
        [true_instances[overlap_mask], pred_instances[overlap_mask]]
    )
    pairs, counts = np.unique(stacked, axis=0, return_counts=True)
    pair_gt_ids = pairs[:, 0].astype(np.int32, copy=False)
    pair_pred_ids = pairs[:, 1].astype(np.int32, copy=False)
    pair_intersections = counts.astype(np.int64, copy=False)
    pair_gt_ids, pair_pred_ids, pair_intersections = _sort_co_occurring_pairs(
        pair_gt_ids, pair_pred_ids, pair_intersections
    )
    return OverlapStats(
        gt_ids=gt_ids,
        pred_ids=pred_ids,
        gt_areas=gt_areas,
        pred_areas=pred_areas,
        pair_gt_ids=pair_gt_ids,
        pair_pred_ids=pair_pred_ids,
        pair_intersections=pair_intersections,
    )


def iou_from_intersection(
    intersection: float, gt_area: int, pred_area: int
) -> float:
    union = float(gt_area + pred_area - intersection)
    return intersection / union if union > 0 else 0.0


def iou_matrix_from_overlap(stats: OverlapStats) -> np.ndarray:
    """Fill a dense IoU matrix from sparse co-occurring pairs in O(n_pairs)."""
    nt, np_ = len(stats.gt_ids), len(stats.pred_ids)
    mat = np.zeros((nt, np_), dtype=np.float64)
    if nt == 0 or np_ == 0:
        return mat

    gt_index = {tid: i for i, tid in enumerate(stats.gt_ids)}
    pred_index = {pid: j for j, pid in enumerate(stats.pred_ids)}
    for tid, pid, inter in zip(
        stats.pair_gt_ids,
        stats.pair_pred_ids,
        stats.pair_intersections,
        strict=True,
    ):
        i = gt_index[int(tid)]
        j = pred_index[int(pid)]
        mat[i, j] = iou_from_intersection(
            float(inter), stats.gt_areas[int(tid)], stats.pred_areas[int(pid)]
        )
    return mat


def pair_intersection_lookup(stats: OverlapStats) -> dict[tuple[int, int], float]:
    return {
        (int(tid), int(pid)): float(inter)
        for tid, pid, inter in zip(
            stats.pair_gt_ids,
            stats.pair_pred_ids,
            stats.pair_intersections,
            strict=True,
        )
    }
