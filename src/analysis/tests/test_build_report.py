"""Reporting bundle assembly (tables + summary without plotting)."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.build_report import build_reporting_bundle
from analysis.tests.test_discover import MINIMAL_INSTANCE_METRICS, _write_json


def test_build_reporting_bundle_writes_derived_and_summary(tmp_path: Path) -> None:
    root = tmp_path / "GrainSeg"
    _write_json(
        root / "eval/yolo_PPL/instance_metrics.json",
        {
            **MINIMAL_INSTANCE_METRICS,
            "variant": "PPL",
            "samples": [
                {
                    "sample_id": "test",
                    "aji": 0.2,
                    "f1_iou50": 0.3,
                    "precision_iou50": 0.0,
                    "recall_iou50": 0.0,
                    "precision_iou75": 0.0,
                    "recall_iou75": 0.0,
                    "f1_iou75": 0.0,
                    "mP_iou50_95": 0.0,
                    "mR_iou50_95": 0.0,
                    "mF1_iou50_95": 0.0,
                    "gt_instances": 1,
                    "predicted_grain_count": 1,
                    "empty_gt": False,
                }
            ],
        },
    )
    out = tmp_path / "reporting"
    summary = build_reporting_bundle(root, out, render_figures=False)
    assert summary["n_instance_rows"] == 1
    assert (out / "derived" / "instance_metrics.csv").is_file()
    assert (out / "analysis_summary.json").is_file()
    payload = json.loads((out / "analysis_summary.json").read_text(encoding="utf-8"))
    assert "scope_note" in payload
    assert payload["figures"] == []
