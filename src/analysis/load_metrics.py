"""Load and normalize eval metric JSON for reporting tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.discover import EvalRunRef, UltralyticsValRef
from common.reporting import INSTANCE_METRIC_KEYS, aggregate_mean_metrics
from common.variants import get_variant

HEADLINE_KEYS = ("aji", "f1_iou50")


def load_instance_metrics_json(report: dict[str, Any]) -> dict[str, float]:
    """Headline instance metrics from a whole or patch instance_metrics.json."""
    mean = report.get("mean")
    if isinstance(mean, dict):
        return {key: float(mean[key]) for key in HEADLINE_KEYS if key in mean}

    samples = report.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("instance_metrics.json has no mean or samples")

    aggregated = aggregate_mean_metrics(samples, keys=INSTANCE_METRIC_KEYS)
    return {key: float(aggregated[key]) for key in HEADLINE_KEYS}


def patch_supporting_metrics(report: dict[str, Any]) -> dict[str, float]:
    extras = report.get("extras")
    if not isinstance(extras, dict):
        return {}
    out: dict[str, float] = {}
    for key in ("mean_aji_grainy", "mean_f1_iou50_grainy"):
        if key in extras and extras[key] is not None:
            out[key] = float(extras[key])
    return out


def load_mask_ap_json(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    mean = report.get("mean_coco_mask_ap")
    if not isinstance(mean, dict):
        return {}
    return {
        "mask_ap50": float(mean["mean_AP50"]),
        "mask_ap": float(mean["mean_AP"]),
    }


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
        report = json.loads(run.instance_metrics_path.read_text(encoding="utf-8"))
        metrics = load_instance_metrics_json(report)
        supporting = patch_supporting_metrics(report) if run.unit == "patch" else {}
        if run.mask_ap_metrics_path is not None:
            supporting = {**supporting, **load_mask_ap_json(run.mask_ap_metrics_path)}
        rows.append(
            instance_metric_row(
                producer=run.producer,
                variant=run.variant,
                unit=run.unit,
                metrics=metrics,
                supporting=supporting,
                source_path=run.instance_metrics_path,
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
