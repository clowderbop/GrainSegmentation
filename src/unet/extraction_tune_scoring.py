"""Train whole-section PQ scoring for U-Net extraction profile tuning (ADR 0003)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from common.merged_view_pq import (
    MERGED_VIEW_PQ_COUNT_KEYS,
    MERGED_VIEW_PQ_RESULT_KEYS,
    MergedViewPqResult,
    compute_merged_view_pq,
    format_merged_view_pq_value,
)
from unet.instance_masks import semantic_to_instance_label_map_watershed

WATERSHED_SELECTION_OBJECTIVE = "pq"


@dataclass(frozen=True)
class WatershedParamSet:
    min_distance: int
    boundary_dilate_iter: int
    watershed_connectivity: int
    min_area_px: int
    exclude_border: bool
    ridge_level: float | None


def _watershed_kwargs(
    params: WatershedParamSet,
    *,
    interior_class: int = 1,
    boundary_class: int = 2,
) -> dict[str, Any]:
    kw: dict[str, Any] = dict(
        interior_class=interior_class,
        boundary_class=boundary_class,
        min_distance=params.min_distance,
        boundary_dilate_iter=params.boundary_dilate_iter,
        watershed_connectivity=params.watershed_connectivity,
        min_area_px=params.min_area_px,
        exclude_border=params.exclude_border,
    )
    if params.ridge_level is not None:
        kw["ridge_level"] = params.ridge_level
    return kw


def instance_map_for_watershed_params(
    pred_semantic: np.ndarray,
    params: WatershedParamSet,
) -> np.ndarray:
    return semantic_to_instance_label_map_watershed(
        pred_semantic, **_watershed_kwargs(params)
    )


def merged_view_pq_for_sample(
    true_instances: np.ndarray,
    pred_semantic: np.ndarray,
    params: WatershedParamSet,
) -> MergedViewPqResult:
    pred_instances = instance_map_for_watershed_params(pred_semantic, params)
    return compute_merged_view_pq(true_instances, pred_instances)


def _mean_merged_view_pq(
    results: Sequence[MergedViewPqResult],
) -> dict[str, float | int]:
    if not results:
        raise ValueError("results must not be empty")
    out: dict[str, float | int] = {}
    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        if key in MERGED_VIEW_PQ_COUNT_KEYS:
            out[key] = int(round(float(np.mean([r[key] for r in results]))))
        else:
            out[key] = float(np.mean([float(r[key]) for r in results]))
    return out


def mean_train_pq_for_watershed_params(
    true_instances_per_sample: Sequence[np.ndarray],
    pred_semantic_per_sample: Sequence[np.ndarray],
    params: WatershedParamSet,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    if len(true_instances_per_sample) != len(pred_semantic_per_sample):
        raise ValueError("true and pred lists must have the same length")
    per_sample: list[dict[str, float | int]] = []
    for true_instances, pred_semantic in zip(
        true_instances_per_sample, pred_semantic_per_sample, strict=True
    ):
        per_sample.append(dict(merged_view_pq_for_sample(true_instances, pred_semantic, params)))
    return _mean_merged_view_pq(per_sample), per_sample


def watershed_per_sample_columns(
    sample_ids: Sequence[str],
    per_sample: Sequence[dict[str, float | int]],
    *,
    sanitize_sample_id: Callable[[str], str],
) -> dict[str, str]:
    if len(sample_ids) != len(per_sample):
        raise ValueError("sample_ids and per_sample must have the same length")
    out: dict[str, str] = {}
    for sid, sample_result in zip(sample_ids, per_sample, strict=True):
        safe_sid = sanitize_sample_id(sid)
        for key in MERGED_VIEW_PQ_RESULT_KEYS:
            out[f"{key}__{safe_sid}"] = format_merged_view_pq_value(key, sample_result[key])
    return out


def watershed_tune_row(
    params: WatershedParamSet,
    mean_pq: dict[str, float | int],
    *,
    per_sample_pq: dict[str, str],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "min_distance": params.min_distance,
        "boundary_dilate_iter": params.boundary_dilate_iter,
        "watershed_connectivity": params.watershed_connectivity,
        "min_area_px": params.min_area_px,
        "exclude_border": int(params.exclude_border),
        "ridge_level": "" if params.ridge_level is None else f"{params.ridge_level:g}",
    }
    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        row[f"mean_{key}"] = format_merged_view_pq_value(key, mean_pq[key])
    row.update(per_sample_pq)
    return row


def watershed_tune_fieldnames(
    sample_ids: Sequence[str],
    *,
    sanitize_sample_id: Callable[[str], str],
) -> list[str]:
    param_fields = [
        "min_distance",
        "boundary_dilate_iter",
        "watershed_connectivity",
        "min_area_px",
        "exclude_border",
        "ridge_level",
    ]
    mean_fields = [f"mean_{key}" for key in MERGED_VIEW_PQ_RESULT_KEYS]
    per_sample_fields = [
        f"{key}__{sanitize_sample_id(sid)}"
        for sid in sample_ids
        for key in MERGED_VIEW_PQ_RESULT_KEYS
    ]
    return param_fields + mean_fields + per_sample_fields


def select_best_watershed_tune_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    return max(rows, key=lambda row: float(row["mean_pq"]))


def _json_scalar_for_pq_field(key: str, value: float | int) -> float | int:
    if key in MERGED_VIEW_PQ_COUNT_KEYS:
        return int(value)
    return float(value)


def watershed_best_json_summary(
    best_row: dict[str, Any],
    best_params: WatershedParamSet,
    sample_ids: Sequence[str],
    *,
    sanitize_sample_id: Callable[[str], str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "selection_objective": WATERSHED_SELECTION_OBJECTIVE,
        "best_params": {
            "min_distance": best_params.min_distance,
            "boundary_dilate_iter": best_params.boundary_dilate_iter,
            "watershed_connectivity": best_params.watershed_connectivity,
            "min_area_px": best_params.min_area_px,
            "exclude_border": best_params.exclude_border,
            "ridge_level": best_params.ridge_level,
        },
        "sample_ids": list(sample_ids),
    }
    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        summary[f"best_mean_{key}"] = _json_scalar_for_pq_field(key, float(best_row[f"mean_{key}"]))
        summary[f"best_per_sample_{key}"] = {
            sid: _json_scalar_for_pq_field(key, float(best_row[f"{key}__{sanitize_sample_id(sid)}"]))
            for sid in sample_ids
        }
    return summary
