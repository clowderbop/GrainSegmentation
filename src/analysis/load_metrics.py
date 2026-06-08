"""Load and normalize eval metric JSON for reporting tables."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.discover import EvalRunRef, UltralyticsValRef
from common.reporting import (
    INSTANCE_METRIC_KEYS,
    aggregate_mean_metrics,
    patch_aggregate_extra_keys,
)
from common.variants import get_variant


class IncompleteInstanceMetricBundleError(ValueError):
    """Raised when instance_metrics.json lacks the PQ-centered metric bundle."""


def _is_empty_gt_false_positive_ratio(mapping: dict[str, Any]) -> bool:
    """GT has no instances but predictions exist (ratio is +inf before JSON null)."""
    if "gt_instance_count" not in mapping or "pred_instance_count" not in mapping:
        return False
    gt = int(mapping["gt_instance_count"])
    pred = int(mapping["pred_instance_count"])
    return gt == 0 and pred > 0


def _coerce_pred_gt_instance_ratio(
    value: Any,
    mapping: dict[str, Any],
) -> float:
    if not _is_empty_gt_false_positive_ratio(mapping):
        if value is None or isinstance(value, bool):
            raise IncompleteInstanceMetricBundleError(
                "invalid value for 'pred_gt_instance_ratio' in instance metric bundle"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise IncompleteInstanceMetricBundleError(
                "non-finite value for 'pred_gt_instance_ratio' in instance metric bundle"
            )
        return numeric
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("inf")
    numeric = float(value)
    if math.isinf(numeric) and numeric > 0:
        return numeric
    if math.isfinite(numeric):
        raise IncompleteInstanceMetricBundleError(
            "pred_gt_instance_ratio must be +inf when gt_instance_count is 0 "
            "and pred_instance_count > 0"
        )
    raise IncompleteInstanceMetricBundleError(
        "invalid pred_gt_instance_ratio for empty-GT false-positive sample"
    )


def _normalize_serialized_sample_row(row: dict[str, Any]) -> dict[str, Any]:
    """Restore bundle fields dropped by json_safe_for_dump (e.g. +inf count ratio)."""
    normalized = dict(row)
    if _is_empty_gt_false_positive_ratio(normalized):
        if normalized.get("pred_gt_instance_ratio") is None:
            normalized["pred_gt_instance_ratio"] = float("inf")
    return normalized


def _reject_pre_policy_sample_rows(
    samples: list[dict[str, Any]],
    *,
    context: str = "",
) -> None:
    prefix = f"{context}: " if context else ""
    for row in samples:
        if not isinstance(row, dict):
            continue
        if "aji" in row and "pq" not in row:
            raise IncompleteInstanceMetricBundleError(
                f"{prefix}pre-policy instance_metrics.json (AJI/F1 headline fields only); "
                "regenerate eval artifacts under the PQ bundle policy"
            )


def _require_sample_bundle_rows(
    samples: list[dict[str, Any]],
    *,
    context: str = "",
) -> None:
    prefix = f"{context}: " if context else ""
    for row in samples:
        if not isinstance(row, dict):
            raise IncompleteInstanceMetricBundleError(
                f"{prefix}invalid sample row in instance_metrics.json"
            )
        missing = [
            key
            for key in INSTANCE_METRIC_KEYS
            if key not in row
            and not (
                key == "pred_gt_instance_ratio"
                and _is_empty_gt_false_positive_ratio(row)
            )
        ]
        if missing:
            raise IncompleteInstanceMetricBundleError(
                f"{prefix}incomplete instance metric bundle in sample "
                f"{row.get('sample_id', '<unknown>')!r}; missing fields: "
                + ", ".join(missing)
            )


def _bundle_metrics_from_mapping(
    mapping: dict[str, Any],
    *,
    context: str = "",
) -> dict[str, float]:
    prefix = f"{context}: " if context else ""
    if "aji" in mapping and "pq" not in mapping:
        raise IncompleteInstanceMetricBundleError(
            f"{prefix}pre-policy instance_metrics.json (AJI/F1 headline fields only); "
            "regenerate eval artifacts under the PQ bundle policy"
        )
    missing = [
        key
        for key in INSTANCE_METRIC_KEYS
        if key not in mapping
        and not (
            key == "pred_gt_instance_ratio"
            and _is_empty_gt_false_positive_ratio(mapping)
        )
    ]
    if missing:
        raise IncompleteInstanceMetricBundleError(
            f"{prefix}incomplete instance metric bundle; missing fields: "
            + ", ".join(missing)
        )

    out: dict[str, float] = {}
    for key in INSTANCE_METRIC_KEYS:
        if key == "pred_gt_instance_ratio":
            out[key] = _coerce_pred_gt_instance_ratio(mapping.get(key), mapping)
            continue
        value = mapping[key]
        if value is None or isinstance(value, bool):
            raise IncompleteInstanceMetricBundleError(
                f"{prefix}invalid value for {key!r} in instance metric bundle"
            )
        numeric = float(value)
        if not math.isfinite(numeric):
            raise IncompleteInstanceMetricBundleError(
                f"{prefix}non-finite value for {key!r} in instance metric bundle"
            )
        out[key] = numeric
    return out


def load_instance_metrics_json(
    report: dict[str, Any],
    *,
    context: str = "",
) -> dict[str, float]:
    """PQ-centered instance metric bundle from instance_metrics.json (no recomputation)."""
    mean = report.get("mean")
    if isinstance(mean, dict):
        if "aji" in mean and "pq" not in mean:
            raise IncompleteInstanceMetricBundleError(
                f"{context + ': ' if context else ''}"
                "pre-policy instance_metrics.json (AJI/F1 headline fields only); "
                "regenerate eval artifacts under the PQ bundle policy"
            )
        return _bundle_metrics_from_mapping(mean, context=context)

    samples = report.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(
            f"{context + ': ' if context else ''}"
            "instance_metrics.json has no mean or samples"
        )

    _reject_pre_policy_sample_rows(samples, context=context)
    normalized = [
        _normalize_serialized_sample_row(row)
        for row in samples
        if isinstance(row, dict)
    ]
    _require_sample_bundle_rows(normalized, context=context)
    if len(normalized) == 1:
        return _bundle_metrics_from_mapping(normalized[0], context=context)
    aggregated = aggregate_mean_metrics(normalized, keys=INSTANCE_METRIC_KEYS)
    return _bundle_metrics_from_mapping(aggregated, context=context)


def patch_supporting_metrics(
    report: dict[str, Any], *, context: str = ""
) -> dict[str, float]:
    extras = report.get("extras")
    if not isinstance(extras, dict):
        return {}
    prefix = f"{context}: " if context else ""
    out: dict[str, float] = {}
    for key in patch_aggregate_extra_keys():
        if key not in extras or extras[key] is None:
            continue
        value = extras[key]
        if isinstance(value, bool):
            raise IncompleteInstanceMetricBundleError(
                f"{prefix}invalid value for extras[{key!r}]"
            )
        if key in ("n_patches", "n_empty_gt"):
            out[key] = float(int(value))
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            raise IncompleteInstanceMetricBundleError(
                f"{prefix}non-finite value for extras[{key!r}]"
            )
        out[key] = numeric
    return out


def load_ultralytics_val_json(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    seg = report.get("seg") or report.get("mask") or {}
    if not isinstance(seg, dict):
        return {}
    out: dict[str, float] = {}
    for src_key, dst_key in (("map50", "val_seg_map50"), ("map", "val_seg_map")):
        if src_key in seg and seg[src_key] is not None:
            out[dst_key] = float(seg[src_key])
    return out


def instance_metric_row(
    *,
    producer: str,
    variant: str,
    unit: str,
    metrics: dict[str, float],
    supporting: dict[str, float] | None = None,
    source_path: Path | None = None,
) -> dict[str, Any]:
    spec = get_variant(variant)
    row: dict[str, Any] = {
        "producer": producer,
        "variant": variant,
        "display_name": spec.display_name,
        "input_image_count": spec.unet.num_inputs,
        "unit": unit,
        "source": "instance",
    }
    if source_path is not None:
        row["source_path"] = str(source_path)
    row.update(metrics)
    if supporting:
        row.update(supporting)
    return row


def metrics_table_from_runs(runs: list[EvalRunRef]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run in runs:
        path = run.instance_metrics_path
        context = str(path)
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = load_instance_metrics_json(report, context=context)
        supporting = (
            patch_supporting_metrics(report, context=context)
            if run.unit == "patch"
            else {}
        )
        rows.append(
            instance_metric_row(
                producer=run.producer,
                variant=run.variant,
                unit=run.unit,
                metrics=metrics,
                supporting=supporting,
                source_path=path,
            )
        )
    return pd.DataFrame(rows)


def ultralytics_val_table(refs: list[UltralyticsValRef]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ref in refs:
        spec = get_variant(ref.variant)
        metrics = load_ultralytics_val_json(ref.metrics_path)
        rows.append(
            {
                "producer": "yolo",
                "variant": ref.variant,
                "display_name": spec.display_name,
                "unit": "patch",
                "source": "ultralytics_val",
                "source_path": str(ref.metrics_path),
                **metrics,
            }
        )
    return pd.DataFrame(rows)
