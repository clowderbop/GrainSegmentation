"""Reporting bundle assembly (tables + summary without plotting)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import pytest

from analysis.build_report import (
    FIGURE_NOT_GENERATED_SKIP_REASON,
    FIGURES_DISABLED_SKIP_REASON,
    MISSING_DQ_SQ_SKIP_REASON,
    MODEL_COMPARISON_SKIP_REASON,
    build_reporting_bundle,
)
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


def _producer_comparison_eval_tree(root: Path) -> None:
    pq_by_variant = {
        "PPL": {"yolo": 0.30, "unet": 0.40},
        "PPL+AllPPX": {"yolo": 0.50, "unet": 0.45},
    }
    for variant, scores in pq_by_variant.items():
        for model_type, pq in scores.items():
            if model_type == "yolo":
                eval_path = root / f"eval/yolo_{variant}/instance_metrics.json"
            else:
                eval_path = (
                    root
                    / f"eval/unet_test/run_unet_finetuned_{variant}/instance_metrics.json"
                )
            _write_json(
                eval_path,
                {
                    **MINIMAL_INSTANCE_METRICS,
                    "schema_version": 2,
                    "model_type": model_type,
                    "variant": variant,
                    "samples": [{**PQ_SAMPLE_ROW, "pq": pq}],
                },
            )


def test_build_reporting_bundle_writes_producer_comparison_tables(tmp_path: Path) -> None:
    root = tmp_path / "GrainSeg"
    _producer_comparison_eval_tree(root)
    out = tmp_path / "reporting"
    summary = build_reporting_bundle(root, out, render_figures=False)

    derived = summary["written"]["derived_tables"]
    assert "per_variant_winner.csv" in derived
    assert "ppl_baseline_gain.csv" in derived
    assert "model_family_comparison_matrix.csv" in derived

    winner = pd.read_csv(out / "derived" / "per_variant_winner.csv")
    assert "Winner" in winner.columns
    assert winner.loc[winner["Input configuration"] == "FullStack", "Winner"].iloc[0] == "YOLO"

    comparison = pd.read_csv(
        out / "derived" / "model_family_comparison_matrix.csv", index_col=0
    )
    assert comparison.loc["Whole-section PQ", "PPL"] == pytest.approx(-0.10)


def test_build_reporting_bundle_skips_family_comparison_without_both_producers(
    tmp_path: Path,
) -> None:
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

    derived = summary["written"]["derived_tables"]
    assert "ppl_baseline_gain.csv" in derived
    assert "per_variant_winner.csv" not in derived
    assert "model_family_comparison_matrix.csv" not in derived

    skipped_by_id = {item["id"]: item["reason"] for item in summary["skipped"]}
    assert skipped_by_id["per_variant_winner_table"] == MODEL_COMPARISON_SKIP_REASON
    assert skipped_by_id["model_family_comparison_matrix"] == MODEL_COMPARISON_SKIP_REASON
    assert skipped_by_id["ppl_relative_diagnostic_heatmap"] == FIGURES_DISABLED_SKIP_REASON


def test_build_reporting_bundle_skips_model_comparison_for_mosaic_producers(
    tmp_path: Path,
) -> None:
    root = tmp_path / "GrainSeg"
    _write_json(
        root / "eval/yolo_PPL/instance_metrics.json",
        {
            **MINIMAL_INSTANCE_METRICS,
            "schema_version": 2,
            "model_type": "yolo",
            "variant": "PPL",
            "samples": [{**PQ_SAMPLE_ROW, "pq": 0.30}],
        },
    )
    _write_json(
        root
        / "eval/unet_test/run_unet_finetuned_PPL+AllPPX/instance_metrics.json",
        {
            **MINIMAL_INSTANCE_METRICS,
            "schema_version": 2,
            "model_type": "unet",
            "variant": "PPL+AllPPX",
            "samples": [{**PQ_SAMPLE_ROW, "pq": 0.45}],
        },
    )
    out = tmp_path / "reporting"
    summary = build_reporting_bundle(root, out, render_figures=False)

    derived = summary["written"]["derived_tables"]
    assert "per_variant_winner.csv" not in derived
    assert "model_family_comparison_matrix.csv" not in derived
    skipped_by_id = {item["id"]: item["reason"] for item in summary["skipped"]}
    assert skipped_by_id["per_variant_winner_table"] == MODEL_COMPARISON_SKIP_REASON
    assert skipped_by_id["model_family_comparison_matrix"] == MODEL_COMPARISON_SKIP_REASON
    assert skipped_by_id["ppl_relative_diagnostic_heatmap"] == FIGURES_DISABLED_SKIP_REASON


def test_build_reporting_bundle_skips_ppl_heatmap_when_renderer_writes_no_file(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

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
    summary = build_reporting_bundle(root, out, render_figures=True)

    assert "ppl_delta_heatmap.png" not in summary["written"]["figures"]
    assert "ppl_relative_diagnostic_heatmaps.png" not in summary["written"]["figures"]
    skipped_by_id = {item["id"]: item["reason"] for item in summary["skipped"]}
    assert (
        skipped_by_id["ppl_relative_diagnostic_heatmap"]
        == FIGURE_NOT_GENERATED_SKIP_REASON
    )


def test_build_reporting_bundle_registers_figure_outputs_when_rendering_disabled(
    tmp_path: Path,
) -> None:
    root = tmp_path / "GrainSeg"
    _producer_comparison_eval_tree(root)
    out = tmp_path / "reporting"
    summary = build_reporting_bundle(root, out, render_figures=False)

    skipped_by_id = {item["id"]: item["reason"] for item in summary["skipped"]}
    assert skipped_by_id["ppl_relative_diagnostic_heatmap"] == FIGURES_DISABLED_SKIP_REASON
    assert skipped_by_id["pq_decomposition_grouped_bars"] == FIGURES_DISABLED_SKIP_REASON
    assert "whole_section_pq_matrix_heatmap" not in skipped_by_id
    assert summary["written"]["figures"] == []


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
    assert "pq_decomposition_grouped_bars.png" in summary["written"]["figures"]
    assert "ppl_relative_diagnostic_heatmaps.png" in summary["written"]["figures"]
    assert (out / "derived" / "whole_section_pq_matrix.csv").is_file()
    assert (out / "figures" / "headline_heatmap.png").is_file()
    assert (out / "figures" / "pq_decomposition_grouped_bars.png").is_file()
    assert (out / "figures" / "ppl_relative_diagnostic_heatmaps.png").is_file()

    skipped_ids = {item["id"] for item in summary["skipped"]}
    assert "ppl_relative_diagnostic_heatmap" not in skipped_ids
    assert "whole_section_pq_matrix_heatmap" not in skipped_ids
    assert "pq_decomposition_grouped_bars" not in skipped_ids

    matrix = pd.read_csv(out / "derived" / "whole_section_pq_matrix.csv", index_col=0)
    assert matrix.loc["YOLO", "PPL"] == pytest.approx(0.30)


def test_build_reporting_bundle_writes_failure_mode_outputs(tmp_path: Path) -> None:
    root = tmp_path / "GrainSeg"
    _figure_ready_eval_tree(root)
    out = tmp_path / "reporting"
    summary = build_reporting_bundle(root, out, render_figures=False)

    derived = summary["written"]["derived_tables"]
    assert "failure_mode_classification.csv" in derived
    assert "failure_mode_classification_rules.md" in derived

    failure = pd.read_csv(out / "derived" / "failure_mode_classification.csv")
    assert "Failure mode labels" in failure.columns

    rules = (out / "derived" / "failure_mode_classification_rules.md").read_text(
        encoding="utf-8"
    )
    assert "detection-limited" in rules

    skipped_ids = {item["id"] for item in summary["skipped"]}
    assert "failure_mode_classification" not in skipped_ids
    skipped_by_id = {item["id"]: item["reason"] for item in summary["skipped"]}
    assert skipped_by_id["pq_decomposition_grouped_bars"] == FIGURES_DISABLED_SKIP_REASON


def test_build_reporting_bundle_skips_failure_mode_without_dq_sq(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "GrainSeg"
    _figure_ready_eval_tree(root)
    out = tmp_path / "reporting"

    monkeypatch.setattr(
        "analysis.build_report.failure_mode_metrics_available",
        lambda _df: False,
    )
    monkeypatch.setattr(
        "analysis.build_report.pq_decomposition_metrics_available",
        lambda _df: False,
    )

    summary = build_reporting_bundle(root, out, render_figures=False)

    derived = summary["written"]["derived_tables"]
    assert "failure_mode_classification.csv" not in derived
    assert "failure_mode_classification_rules.md" not in derived

    skipped_by_id = {item["id"]: item["reason"] for item in summary["skipped"]}
    assert skipped_by_id["failure_mode_classification"] == MISSING_DQ_SQ_SKIP_REASON
    assert skipped_by_id["pq_decomposition_grouped_bars"] == MISSING_DQ_SQ_SKIP_REASON
