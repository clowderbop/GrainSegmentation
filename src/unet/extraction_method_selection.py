"""Train whole-section PQ selection for CC vs tuned watershed (ADR 0003)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

from common.instance_eval_report import (
    extract_instance_metric_bundle_from_report,
    load_instance_eval_report,
    load_train_whole_section_bundles_from_eval_dir,
    mean_bundle_across_variants,
    validate_train_whole_section_report,
)
from common.instance_metric_bundle import INSTANCE_METRIC_BUNDLE_KEYS

ExtractionMethod = Literal["cc", "watershed"]


@dataclass(frozen=True)
class ExtractionMethodAudit:
    method: ExtractionMethod
    bundle: dict[str, float]
    per_variant_bundles: dict[str, dict[str, float]]


@dataclass(frozen=True)
class ExtractionMethodSelection:
    selected_method: ExtractionMethod
    objective_pq: float
    cc: ExtractionMethodAudit
    watershed: ExtractionMethodAudit


def select_train_extraction_method(
    *,
    cc_bundle: dict[str, float],
    watershed_bundle: dict[str, float],
    cc_per_variant: dict[str, dict[str, float]] | None = None,
    watershed_per_variant: dict[str, dict[str, float]] | None = None,
) -> ExtractionMethodSelection:
    cc_pq = float(cc_bundle["pq"])
    watershed_pq = float(watershed_bundle["pq"])
    if watershed_pq > cc_pq:
        selected: ExtractionMethod = "watershed"
        objective_pq = watershed_pq
    elif cc_pq > watershed_pq:
        selected = "cc"
        objective_pq = cc_pq
    else:
        selected = "cc"
        objective_pq = cc_pq
    return ExtractionMethodSelection(
        selected_method=selected,
        objective_pq=objective_pq,
        cc=ExtractionMethodAudit(
            method="cc",
            bundle=dict(cc_bundle),
            per_variant_bundles=dict(cc_per_variant or {}),
        ),
        watershed=ExtractionMethodAudit(
            method="watershed",
            bundle=dict(watershed_bundle),
            per_variant_bundles=dict(watershed_per_variant or {}),
        ),
    )


def extraction_method_selection_to_json(
    selection: ExtractionMethodSelection,
) -> dict[str, Any]:
    def _method_payload(audit: ExtractionMethodAudit) -> dict[str, Any]:
        payload: dict[str, Any] = {"method": audit.method}
        for key in INSTANCE_METRIC_BUNDLE_KEYS:
            payload[key] = float(audit.bundle[key])
        if audit.per_variant_bundles:
            payload["per_variant"] = {
                variant: {key: float(bundle[key]) for key in INSTANCE_METRIC_BUNDLE_KEYS}
                for variant, bundle in audit.per_variant_bundles.items()
            }
        return payload

    return {
        "selection_objective": "pq",
        "manifest_split": "train",
        "unit": "whole",
        "selected_method": selection.selected_method,
        "objective_pq": selection.objective_pq,
        "cc": _method_payload(selection.cc),
        "watershed": _method_payload(selection.watershed),
    }


def _bundle_from_validated_report(path: Path, *, model_type: str) -> dict[str, float]:
    report = load_instance_eval_report(path)
    validate_train_whole_section_report(report, model_type=model_type)
    return extract_instance_metric_bundle_from_report(report)


def select_train_extraction_method_from_reports(
    *,
    cc_report_path: Path,
    watershed_report_path: Path,
    model_type: str = "unet",
) -> ExtractionMethodSelection:
    return select_train_extraction_method(
        cc_bundle=_bundle_from_validated_report(cc_report_path, model_type=model_type),
        watershed_bundle=_bundle_from_validated_report(
            watershed_report_path, model_type=model_type
        ),
    )


def select_train_extraction_method_from_eval_dirs(
    *,
    cc_eval_dir: Path,
    watershed_eval_dir: Path,
    variant_names: Sequence[str],
    model_type: str = "unet",
) -> ExtractionMethodSelection:
    """Aggregate train whole-section PQ across registry variants per method."""
    cc_per_variant = load_train_whole_section_bundles_from_eval_dir(
        cc_eval_dir, variant_names, model_type=model_type
    )
    watershed_per_variant = load_train_whole_section_bundles_from_eval_dir(
        watershed_eval_dir, variant_names, model_type=model_type
    )
    return select_train_extraction_method(
        cc_bundle=mean_bundle_across_variants(cc_per_variant),
        watershed_bundle=mean_bundle_across_variants(watershed_per_variant),
        cc_per_variant=cc_per_variant,
        watershed_per_variant=watershed_per_variant,
    )


def write_extraction_method_selection_json(
    path: Path,
    selection: ExtractionMethodSelection,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(extraction_method_selection_to_json(selection), indent=2),
        encoding="utf-8",
    )
