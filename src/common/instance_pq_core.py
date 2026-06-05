"""Shared PQ decomposition helpers for merged instance view scoring."""

from __future__ import annotations


def pred_gt_instance_ratio(gt_count: int, pred_count: int) -> float:
    if gt_count > 0:
        return float(pred_count) / float(gt_count)
    if pred_count == 0:
        return 1.0
    return float("inf")


def pq_from_match_counts(
    tp: int, fp: int, fn: int, matched_ious: list[float]
) -> tuple[float, float, float]:
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0, 1.0, 1.0
    denom = tp + 0.5 * fp + 0.5 * fn
    dq = float(tp) / float(denom) if denom > 0 else 0.0
    sq = float(sum(matched_ious)) / float(tp) if tp > 0 else 0.0
    return dq * sq, dq, sq
