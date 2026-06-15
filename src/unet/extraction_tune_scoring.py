"""Train whole-section PQ scoring for U-Net extraction profile tuning (ADR 0003)."""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from common.instance_overlap import GtOverlapPrep
from common.semantic_instance import SEMANTIC_BOUNDARY_CLASS, SEMANTIC_INTERIOR_CLASS
from common.merged_view_pq import (
    MERGED_VIEW_PQ_RESULT_KEYS,
    MergedViewPqResult,
    coerce_merged_view_pq_value,
    compute_merged_view_pq,
    format_merged_view_pq_value,
    mean_merged_view_pq_results,
)
from unet.instance_masks import (
    build_watershed_semantic_prep,
    watershed_area_filter,
    watershed_base_extraction,
)

WATERSHED_SELECTION_OBJECTIVE = "pq"


@dataclass
class WatershedScoringTimings:
    watershed_s: float = 0.0
    metrics_s: float = 0.0

    @property
    def total_s(self) -> float:
        return self.watershed_s + self.metrics_s


def log_scoring_phase_timing(phase: str, elapsed_s: float) -> None:
    print(f"    {phase} {elapsed_s:.1f}s", flush=True)


def log_scoring_phase_start(phase: str, *, prefix: str = "") -> None:
    line = f"running {phase} …"
    if prefix:
        print(f"  {prefix}{line}", flush=True)
    else:
        print(f"    {line}", flush=True)


def format_merged_view_pq_audit_line(
    result: MergedViewPqResult | dict[str, float | int],
) -> str:
    from common.merged_view_pq import _merged_view_pq_value

    return (
        f"PQ={float(_merged_view_pq_value(result, 'pq')):.6f} "
        f"DQ={float(_merged_view_pq_value(result, 'dq')):.6f} "
        f"SQ={float(_merged_view_pq_value(result, 'sq')):.6f} "
        f"tp={int(_merged_view_pq_value(result, 'tp'))} "
        f"fp={int(_merged_view_pq_value(result, 'fp'))} "
        f"fn={int(_merged_view_pq_value(result, 'fn'))} "
        f"pred={int(_merged_view_pq_value(result, 'pred_instance_count'))} "
        f"gt={int(_merged_view_pq_value(result, 'gt_instance_count'))} "
        f"ratio={float(_merged_view_pq_value(result, 'pred_gt_instance_ratio')):.3f}"
    )


@dataclass(frozen=True)
class WatershedParamSet:
    min_distance: int
    boundary_dilate_iter: int
    watershed_connectivity: int
    min_area_px: int
    exclude_border: bool
    ridge_level: float | None
    h_maxima: int = 0


def format_watershed_ridge_level(ridge_level: float | None) -> str:
    return "auto" if ridge_level is None else f"{ridge_level:g}"


def format_watershed_param_set(params: WatershedParamSet) -> str:
    return (
        f"min_dist={params.min_distance}, h_maxima={params.h_maxima}, "
        f"dilate={params.boundary_dilate_iter}, "
        f"conn={params.watershed_connectivity}, min_area={params.min_area_px}, "
        f"exclude_border={params.exclude_border}, "
        f"ridge={format_watershed_ridge_level(params.ridge_level)}"
    )


def watershed_param_set_from_tune_row(row: dict[str, Any]) -> WatershedParamSet:
    return WatershedParamSet(
        min_distance=int(row["min_distance"]),
        boundary_dilate_iter=int(row["boundary_dilate_iter"]),
        watershed_connectivity=int(row["watershed_connectivity"]),
        min_area_px=int(row["min_area_px"]),
        exclude_border=bool(int(row["exclude_border"])),
        ridge_level=None if row["ridge_level"] == "" else float(row["ridge_level"]),
        h_maxima=int(row["h_maxima"]),
    )


def sanitize_watershed_tune_csv_sample_id(sample_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", sample_id)


def _watershed_kwargs(
    params: WatershedParamSet,
    *,
    interior_class: int = SEMANTIC_INTERIOR_CLASS,
    boundary_class: int = SEMANTIC_BOUNDARY_CLASS,
) -> dict[str, Any]:
    kw: dict[str, Any] = dict(
        interior_class=interior_class,
        boundary_class=boundary_class,
        min_distance=params.min_distance,
        h_maxima=params.h_maxima,
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
    kw = _watershed_kwargs(params)
    prep = build_watershed_semantic_prep(
        pred_semantic,
        interior_class=kw["interior_class"],
        boundary_class=kw["boundary_class"],
    )
    base = watershed_base_extraction(
        prep,
        min_distance=kw["min_distance"],
        h_maxima=kw["h_maxima"],
        boundary_dilate_iter=kw["boundary_dilate_iter"],
        watershed_connectivity=kw["watershed_connectivity"],
        exclude_border=kw["exclude_border"],
        ridge_level=kw.get("ridge_level"),
    )
    return watershed_area_filter(base, kw["min_area_px"])


def merged_view_pq_for_sample(
    true_instances: np.ndarray,
    pred_semantic: np.ndarray | None,
    params: WatershedParamSet,
    *,
    pred_instances: np.ndarray | None = None,
    gt_prep: GtOverlapPrep | None = None,
    timings: WatershedScoringTimings | None = None,
    log_prefix: str = "",
) -> MergedViewPqResult:
    if pred_instances is None:
        if pred_semantic is None:
            raise ValueError(
                "pred_semantic is required when pred_instances is not provided"
            )
        if timings is not None:
            log_scoring_phase_start("watershed", prefix=log_prefix)
        t0 = time.perf_counter()
        pred_instances = instance_map_for_watershed_params(pred_semantic, params)
        if timings is not None:
            timings.watershed_s = time.perf_counter() - t0
            log_scoring_phase_timing("watershed", timings.watershed_s)
    if timings is not None:
        log_scoring_phase_start("metrics", prefix=log_prefix)
    t0 = time.perf_counter()
    result = compute_merged_view_pq(true_instances, pred_instances, gt_prep=gt_prep)
    if timings is not None:
        timings.metrics_s = time.perf_counter() - t0
        log_scoring_phase_timing("metrics", timings.metrics_s)
    return result


def watershed_tune_sample_prefix(
    idx: int,
    n_samples: int,
    sample_ids: Sequence[str] | None,
    *,
    log: bool,
) -> str:
    if not log or sample_ids is None:
        return ""
    sid = sample_ids[idx]
    return f"[{idx + 1}/{n_samples}] {sid}: " if n_samples > 1 else f"{sid}: "


def mean_train_pq_for_watershed_params(
    true_instances_per_sample: Sequence[np.ndarray],
    pred_semantic_per_sample: Sequence[np.ndarray] | None,
    params: WatershedParamSet,
    *,
    get_pred_instances: Callable[[int, WatershedParamSet], np.ndarray] | None = None,
    gt_overlap_preps: Sequence[GtOverlapPrep] | None = None,
    sample_ids: Sequence[str] | None = None,
    log: bool = False,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    if get_pred_instances is None and pred_semantic_per_sample is None:
        raise ValueError(
            "pred_semantic_per_sample is required when get_pred_instances is not set"
        )
    if pred_semantic_per_sample is not None and len(true_instances_per_sample) != len(
        pred_semantic_per_sample
    ):
        raise ValueError("true and pred lists must have the same length")
    if sample_ids is not None and len(sample_ids) != len(true_instances_per_sample):
        raise ValueError("sample_ids must match the number of samples")
    if gt_overlap_preps is not None and len(gt_overlap_preps) != len(
        true_instances_per_sample
    ):
        raise ValueError("gt_overlap_preps must match the number of samples")
    per_sample: list[dict[str, float | int]] = []
    n_samples = len(true_instances_per_sample)
    for idx in range(n_samples):
        true_instances = true_instances_per_sample[idx]
        pred_semantic = (
            pred_semantic_per_sample[idx]
            if pred_semantic_per_sample is not None
            else None
        )
        timings = WatershedScoringTimings() if log else None
        sample_prefix = watershed_tune_sample_prefix(
            idx, n_samples, sample_ids, log=log
        )
        pred_instances: np.ndarray | None = None
        if get_pred_instances is not None:
            if timings is not None:
                log_scoring_phase_start("watershed", prefix=sample_prefix)
            t0 = time.perf_counter()
            pred_instances = get_pred_instances(idx, params)
            if timings is not None:
                timings.watershed_s = time.perf_counter() - t0
                log_scoring_phase_timing("watershed", timings.watershed_s)
        sample_gt_prep = gt_overlap_preps[idx] if gt_overlap_preps is not None else None
        result = merged_view_pq_for_sample(
            true_instances,
            pred_semantic,
            params,
            pred_instances=pred_instances,
            gt_prep=sample_gt_prep,
            timings=timings,
            log_prefix=sample_prefix,
        )
        per_sample.append(cast(dict[str, float | int], dict(result)))
        if log and timings is not None:
            print(
                f"  {sample_prefix}{format_merged_view_pq_audit_line(result)} "
                f"({timings.total_s:.1f}s)",
                flush=True,
            )
    return mean_merged_view_pq_results(per_sample), per_sample


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
            out[f"{key}__{safe_sid}"] = format_merged_view_pq_value(
                key, sample_result[key]
            )
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
        "h_maxima": params.h_maxima,
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
        "h_maxima",
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
            "h_maxima": best_params.h_maxima,
            "watershed_connectivity": best_params.watershed_connectivity,
            "min_area_px": best_params.min_area_px,
            "exclude_border": best_params.exclude_border,
            "ridge_level": best_params.ridge_level,
        },
        "sample_ids": list(sample_ids),
    }
    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        summary[f"best_mean_{key}"] = coerce_merged_view_pq_value(
            key, float(best_row[f"mean_{key}"])
        )
        summary[f"best_per_sample_{key}"] = {
            sid: coerce_merged_view_pq_value(
                key, float(best_row[f"{key}__{sanitize_sample_id(sid)}"])
            )
            for sid in sample_ids
        }
    return summary
