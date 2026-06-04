"""Reporting bundle assembly (tables + summary without plotting)."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.build_report import build_reporting_bundle
from analysis.tests.test_discover import MINIMAL_INSTANCE_METRICS, _write_json
from analysis.tests.test_load_metrics import PQ_SAMPLE_ROW


def test_build_reporting_bundle_writes_derived_and_summary(tmp_path: Path) -> None:
    root = tmp_path / "GrainSeg"
    _write_json(
        root / "eval/yolo_PPL/instance_metrics.json",
        {
            **MINIMAL_INSTANCE_METRICS,
            "schema_version": 2,
            "variant": "PPL",
            "samples": [PQ_SAMPLE_ROW],
        },
    )
    out = tmp_path / "reporting"
    summary = build_reporting_bundle(root, out, render_figures=False)
    assert summary["n_instance_rows"] == 1
    assert (out / "derived" / "instance_metrics.csv").is_file()
    assert (out / "analysis_summary.json").is_file()
    payload = json.loads((out / "analysis_summary.json").read_text(encoding="utf-8"))
    assert "Headline whole-section PQ" in payload["scope_note"]
    assert "Headline AJI" not in payload["scope_note"]
    assert payload["figures"] == []
