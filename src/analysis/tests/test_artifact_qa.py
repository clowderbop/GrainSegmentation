"""Artifact QA and narrative summary for post-eval reporting."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.artifact_qa import (
    PATCH_ARTIFACT,
    ULTRALYTICS_VAL_ARTIFACT,
    WHOLE_SECTION_ARTIFACT,
    anomaly_report_table,
    completeness_artifact_audit_table,
    narrative_summary_markdown,
)
from analysis.build_report import build_reporting_bundle
from analysis.discover import discover_eval_runs, discover_ultralytics_val
from analysis.tests.test_derived_tables import _whole_row
from analysis.tests.test_diagnostic_derivation import _patch_row
from analysis.tests.test_discover import MINIMAL_INSTANCE_METRICS, _write_json
from analysis.tests.test_load_metrics import PQ_SAMPLE_ROW


def _audit_status(
    audit: pd.DataFrame,
    *,
    variant: str,
    producer: str,
    artifact: str,
) -> str:
    return audit.loc[
        (audit["Variant"] == variant)
        & (audit["Producer"] == producer)
        & (audit["Artifact"] == artifact),
        "Status",
    ].iloc[0]


def test_completeness_audit_reports_whole_patch_optional_and_val(
    tmp_path: Path,
) -> None:
    root = tmp_path / "GrainSeg"
    _write_json(
        root / "eval/yolo_PPL/instance_metrics.json",
        {**MINIMAL_INSTANCE_METRICS, "variant": "PPL"},
    )
    _write_json(
        root / "eval/yolo_patches/PPL/100/instance_metrics.json",
        {**MINIMAL_INSTANCE_METRICS, "variant": "PPL", "unit": "patch"},
    )
    _write_json(
        root / "runs/yolo26-seg-val/PPL/test/metrics.json",
        {"seg": {"map50": 0.5}},
    )
    runs = discover_eval_runs(root, variants=("PPL",))
    val_refs = discover_ultralytics_val(root, variants=("PPL",))

    audit = completeness_artifact_audit_table(
        root, runs=runs, val_refs=val_refs, variants=("PPL",)
    )

    assert _audit_status(
        audit, variant="PPL", producer="yolo", artifact=WHOLE_SECTION_ARTIFACT
    ) == "found"
    assert _audit_status(
        audit, variant="PPL", producer="unet", artifact=WHOLE_SECTION_ARTIFACT
    ) == "missing"
    assert _audit_status(
        audit, variant="PPL", producer="yolo", artifact=PATCH_ARTIFACT
    ) == "found"
    assert _audit_status(
        audit, variant="PPL", producer="unet", artifact=PATCH_ARTIFACT
    ) == "missing"
    assert audit.loc[
        (audit["Variant"] == "PPL") & (audit["Artifact"] == ULTRALYTICS_VAL_ARTIFACT),
        "Status",
    ].iloc[0] == "found"

    optional_missing = audit.loc[
        (audit["Expected"] == "optional") & (audit["Status"] == "missing")
    ]
    assert optional_missing.empty


def test_completeness_audit_marks_optional_artifacts_missing_when_absent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "GrainSeg"
    _write_json(
        root / "eval/yolo_PPL/instance_metrics.json",
        {**MINIMAL_INSTANCE_METRICS, "variant": "PPL"},
    )
    audit = completeness_artifact_audit_table(root, variants=("PPL",))

    assert "mask_ap" not in " ".join(audit["Artifact"].astype(str)).lower()
    assert audit.loc[
        (audit["Variant"] == "PPL") & (audit["Artifact"] == ULTRALYTICS_VAL_ARTIFACT),
        "Status",
    ].iloc[0] == "missing"


def test_anomaly_report_flags_high_sq_low_dq_and_patch_good_whole_bad() -> None:
    df = pd.DataFrame(
        [
            _whole_row(
                producer="yolo",
                variant="PPL",
                display_name="PPL",
                pq=0.20,
            )
            | {"dq": 0.10, "sq": 0.80},
            _patch_row(
                producer="yolo",
                variant="PPL",
                display_name="PPL",
                weighted_overrides={"pq": 0.55},
            ),
            _whole_row(
                producer="unet",
                variant="PPL+AllPPX",
                display_name="FullStack",
                pq=0.25,
            )
            | {"dq": 0.40, "sq": 0.45},
        ]
    )

    report = anomaly_report_table(df)
    kinds = set(report["Anomaly"].tolist())

    assert "high SQ with low DQ" in kinds
    assert "patch-good / whole-bad (PQ)" in kinds


def test_anomaly_report_flags_strong_signed_count_bias() -> None:
    df = pd.DataFrame(
        [
            _whole_row(
                producer="yolo",
                variant="PPL",
                display_name="PPL",
                pq=0.30,
            )
            | {"pred_gt_instance_ratio": 1.40},
        ]
    )
    report = anomaly_report_table(df)
    assert "strong signed count bias" in set(report["Anomaly"].tolist())


def test_narrative_summary_lists_headline_findings() -> None:
    df = pd.DataFrame(
        [
            _whole_row(
                producer="yolo",
                variant="PPL",
                display_name="PPL",
                pq=0.20,
            )
            | {"pred_gt_instance_ratio": 1.0},
            _patch_row(
                producer="yolo",
                variant="PPL",
                display_name="PPL",
                weighted_overrides={"pq": 0.55},
            ),
            _whole_row(
                producer="yolo",
                variant="PPL+AllPPX",
                display_name="FullStack",
                pq=0.55,
            )
            | {"pred_gt_instance_ratio": 1.50},
            _whole_row(
                producer="unet",
                variant="PPL",
                display_name="PPL",
                pq=0.40,
            ),
            _whole_row(
                producer="unet",
                variant="PPL+AllPPX",
                display_name="FullStack",
                pq=0.45,
            ),
        ]
    )
    text = narrative_summary_markdown(df)
    lower = text.lower()
    assert "best overall whole-section pq" in lower
    assert "YOLO" in text
    assert "U-Net" in text
    assert "largest ppl-relative whole-section pq gain" in lower
    assert "biggest patch-to-whole pq drop" in lower
    assert "strongest signed count bias" in lower
    assert "fullstack" in lower
    assert "ppl" in lower


def test_build_reporting_bundle_writes_qa_outputs_and_registers_summary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "GrainSeg"
    _write_json(
        root / "eval/yolo_PPL/instance_metrics.json",
        {
            **MINIMAL_INSTANCE_METRICS,
            "schema_version": 2,
            "variant": "PPL",
            "samples": [{**PQ_SAMPLE_ROW, "pq": 0.30}],
        },
    )
    _write_json(
        root / "eval/unet_test/run_unet_finetuned_PPL/instance_metrics.json",
        {
            **MINIMAL_INSTANCE_METRICS,
            "schema_version": 2,
            "model_type": "unet",
            "variant": "PPL",
            "samples": [{**PQ_SAMPLE_ROW, "pq": 0.40}],
        },
    )
    out = tmp_path / "reporting"
    summary = build_reporting_bundle(root, out, render_figures=False)

    audits = summary["written"]["audits"]
    narratives = summary["written"]["narratives"]
    assert "completeness_artifact_audit.csv" in audits
    assert "outlier_anomaly_report.csv" in audits
    assert "narrative_summary.md" in narratives
    assert (out / "audits" / "completeness_artifact_audit.csv").is_file()
    assert (out / "audits" / "outlier_anomaly_report.csv").is_file()
    assert (out / "narratives" / "narrative_summary.md").is_file()

    payload = json.loads((out / "analysis_summary.json").read_text(encoding="utf-8"))
    skipped_ids = {item["id"] for item in payload["skipped"]}
    assert "completeness_artifact_audit" not in skipped_ids
    assert "outlier_anomaly_report" not in skipped_ids
    assert "narrative_summary_generator" not in skipped_ids
