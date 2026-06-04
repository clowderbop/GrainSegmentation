"""Train whole-section PQ scoring for U-Net extraction profile tuning (ADR 0003)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from common.instance_metric_bundle import (
    INSTANCE_METRIC_BUNDLE_KEYS,
    InstanceMetricBundle,
    compute_instance_metric_bundle,
)
from common.metrics import compute_aji
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


def instance_metric_bundle_for_sample(
    true_instances: np.ndarray,
    pred_semantic: np.ndarray,
    params: WatershedParamSet,
) -> InstanceMetricBundle:
    pred_instances = instance_map_for_watershed_params(pred_semantic, params)
    return compute_instance_metric_bundle(true_instances, pred_instances)


def _mean_bundle(bundles: Sequence[InstanceMetricBundle]) -> dict[str, float]:
    if not bundles:
        raise ValueError("bundles must not be empty")
    out: dict[str, float] = {}
    for key in INSTANCE_METRIC_BUNDLE_KEYS:
        if key.endswith("_count"):
            out[key] = int(round(float(np.mean([b[key] for b in bundles]))))
        else:
            out[key] = float(np.mean([float(b[key]) for b in bundles]))
    return out


def mean_train_bundle_for_watershed_params(
    true_instances_per_sample: Sequence[np.ndarray],
    pred_semantic_per_sample: Sequence[np.ndarray],
    params: WatershedParamSet,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    if len(true_instances_per_sample) != len(pred_semantic_per_sample):
        raise ValueError("true and pred lists must have the same length")
    per_sample: list[dict[str, float]] = []
    for true_instances, pred_semantic in zip(
        true_instances_per_sample, pred_semantic_per_sample, strict=True
    ):
        per_sample.append(
            dict(
                instance_metric_bundle_for_sample(
                    true_instances, pred_semantic, params
                )
            )
        )
    return _mean_bundle(per_sample), per_sample


def mean_aji_for_watershed_params(
    true_instances_per_sample: Sequence[np.ndarray],
    pred_semantic_per_sample: Sequence[np.ndarray],
    params: WatershedParamSet,
) -> tuple[float, list[float]]:
    """Legacy AJI audit field for watershed tuning CSV rows."""
    if len(true_instances_per_sample) != len(pred_semantic_per_sample):
        raise ValueError("true and pred lists must have the same length")
    ajis: list[float] = []
    for true_instances, pred_semantic in zip(
        true_instances_per_sample, pred_semantic_per_sample, strict=True
    ):
        pred_instances = instance_map_for_watershed_params(pred_semantic, params)
        ajis.append(float(compute_aji(true_instances, pred_instances)))
    return float(np.mean(ajis)), ajis


def watershed_tune_row(
    params: WatershedParamSet,
    mean_bundle: dict[str, float],
    *,
    mean_aji: float,
    per_sample_aji: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "min_distance": params.min_distance,
        "boundary_dilate_iter": params.boundary_dilate_iter,
        "watershed_connectivity": params.watershed_connectivity,
        "min_area_px": params.min_area_px,
        "exclude_border": int(params.exclude_border),
        "ridge_level": "" if params.ridge_level is None else f"{params.ridge_level:g}",
        "mean_pq": f"{mean_bundle['pq']:.8f}",
        "mean_aji": f"{mean_aji:.8f}",
    }
    for key in INSTANCE_METRIC_BUNDLE_KEYS:
        row[f"mean_{key}"] = f"{mean_bundle[key]:.8f}"
    row.update(per_sample_aji)
    return row


def watershed_tune_fieldnames(
    sample_ids: Sequence[str],
    *,
    sanitize_sample_id,
) -> list[str]:
    param_fields = [
        "min_distance",
        "boundary_dilate_iter",
        "watershed_connectivity",
        "min_area_px",
        "exclude_border",
        "ridge_level",
        "mean_pq",
        "mean_aji",
    ]
    bundle_fields = [f"mean_{key}" for key in INSTANCE_METRIC_BUNDLE_KEYS]
    per_sample_aji = [f"aji__{sanitize_sample_id(sid)}" for sid in sample_ids]
    return param_fields + bundle_fields + per_sample_aji


def select_best_watershed_tune_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("rows must not be empty")
    return max(rows, key=lambda row: float(row["mean_pq"]))
