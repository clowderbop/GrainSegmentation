"""Reporting bundle assembly (tables + summary without plotting)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import pytest

from analysis.build_report import build_reporting_bundle
from analysis.derived_tables import WHOLE_SECTION_PQ_COL
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
    assert payload["written"]["figures"] == []
    assert payload["headline_policy"]
    assert payload["reporting_contract"]["wave1_approved"]


def _figure_ready_eval_tree(root: Path) -> None:
    for variant in ("PPL", "PPL+AllPPX"):
        for model_type, eval_path in (
            ("yolo", root / f"eval/yolo_{variant}/instance_metrics.json"),
            (
                "unet",
                root
                / f"eval/unet_test/run_unet_finetuned_{variant}/instance_metrics.json",
            ),
        ):
            _write_json(
                eval_path,
                {
                    **MINIMAL_INSTANCE_METRICS,
                    "schema_version": 2,
                    "model_type": model_type,
                    "variant": variant,
                    "samples": [
                        {
                            **PQ_SAMPLE_ROW,
                            "pq": 0.30 if variant == "PPL" else 0.50,
                        }
                    ],
                },
            )


def test_build_reporting_bundle_writes_thesis_core_tables(tmp_path: Path) -> None:
    root = tmp_path / "GrainSeg"
    _figure_ready_eval_tree(root)
    out = tmp_path / "reporting"
    summary = build_reporting_bundle(root, out, render_figures=False)

    derived = summary["written"]["derived_tables"]
    assert "headline_pq_ranking.csv" in derived
    assert "thesis_ready_results.csv" in derived
    assert "whole_section_pq_matrix.csv" in derived

    ranking = pd.read_csv(out / "derived" / "headline_pq_ranking.csv")
    assert WHOLE_SECTION_PQ_COL in ranking.columns
    assert ranking[WHOLE_SECTION_PQ_COL].is_monotonic_decreasing

    thesis = pd.read_csv(out / "derived" / "thesis_ready_results.csv")
    assert "AJI+" in thesis.columns
    assert not any("map" in col.lower() for col in thesis.columns)

    matrix = pd.read_csv(out / "derived" / "whole_section_pq_matrix.csv", index_col=0)
    assert matrix.index.name == "Model"


def test_build_reporting_bundle_writes_pq_matrix_and_heatmap_with_figures(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    root = tmp_path / "GrainSeg"
    _figure_ready_eval_tree(root)
    out = tmp_path / "reporting"
    summary = build_reporting_bundle(root, out, render_figures=True)

    assert "whole_section_pq_matrix.csv" in summary["written"]["derived_tables"]
    assert "headline_heatmap.png" in summary["written"]["figures"]
    assert (out / "derived" / "whole_section_pq_matrix.csv").is_file()
    assert (out / "figures" / "headline_heatmap.png").is_file()

    matrix = pd.read_csv(out / "derived" / "whole_section_pq_matrix.csv", index_col=0)
    assert matrix.loc["YOLO", "PPL"] == pytest.approx(0.30)
