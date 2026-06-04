"""Wave 1 reporting contract: approved/deferred/cut lists and bundle regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analysis.build_report import (
    PRECISION_RECALL_NOT_INFORMATIVE_SKIP_REASON,
    build_reporting_bundle,
)
from analysis.figures import FIGURE_BUNDLE_FILENAMES
from analysis.reporting_contract import (
    CUT_OUTPUTS,
    FORBIDDEN_REPORTING_OUTPUTS,
    HEADLINE_POLICY,
    WAVE1_APPROVED_OUTPUTS,
    WAVE2_DEFERRED_OUTPUTS,
)
from analysis.tests.test_discover import MINIMAL_INSTANCE_METRICS, _write_json
from analysis.tests.test_load_metrics import PQ_SAMPLE_ROW

PRD_THESIS_CORE_IDS = {
    "headline_ranking_table",
    "whole_section_pq_matrix_heatmap",
    "thesis_ready_results_table",
    "per_variant_winner_table",
    "ppl_baseline_gain_table",
    "pq_decomposition_grouped_bars",
    "model_variant_bars",
}

PRD_DISCUSSION_DIAGNOSTIC_IDS = {
    "ppl_relative_diagnostic_heatmap",
    "model_family_comparison_matrix",
    "patch_to_whole_gap_table",
    "patch_to_whole_diagnostic_heatmap",
    "count_error_bar_chart",
    "strictness_drop_plot",
    "precision_recall_diagnostic_map_iou75",
    "pareto_frontier_table",
    "pareto_plot",
    "failure_mode_classification",
}

PRD_QA_WRITING_IDS = {
    "completeness_artifact_audit",
    "outlier_anomaly_report",
    "narrative_summary_generator",
}


def _forbidden_filename_matches(written_names: set[str]) -> list[str]:
    violations: list[str] = []
    for item in FORBIDDEN_REPORTING_OUTPUTS:
        for pattern in item.get("filename_patterns", []):
            matches = [name for name in written_names if pattern in name]
            if matches:
                violations.append(f"{item['id']!r} matched {matches!r}")
    return violations


def test_wave1_approved_outputs_include_thesis_core_tables() -> None:
    ids = {item["id"] for item in WAVE1_APPROVED_OUTPUTS}
    assert PRD_THESIS_CORE_IDS <= ids


def test_wave1_approved_outputs_match_prd_tiers() -> None:
    by_tier: dict[str, set[str]] = {}
    for item in WAVE1_APPROVED_OUTPUTS:
        by_tier.setdefault(item["tier"], set()).add(item["id"])

    assert by_tier["thesis-core"] == PRD_THESIS_CORE_IDS
    assert by_tier["discussion-diagnostic"] == PRD_DISCUSSION_DIAGNOSTIC_IDS
    assert by_tier["qa-writing"] == PRD_QA_WRITING_IDS
    assert "ppl_delta_heatmap" not in by_tier["thesis-core"]


def test_wave2_deferred_outputs_include_strictness_curve() -> None:
    ids = {item["id"] for item in WAVE2_DEFERRED_OUTPUTS}
    assert "strictness_curve" in ids
    assert "error_composition_bars" in ids


def test_cut_outputs_include_ap_map_additions() -> None:
    ids = {item["id"] for item in CUT_OUTPUTS}
    assert "ap_map_additions" in ids
    assert "dq_vs_sq_scatter" in ids


def test_wave1_approved_outputs_exclude_whole_section_mask_ap() -> None:
    approved_text = json.dumps(WAVE1_APPROVED_OUTPUTS).lower()
    assert "mask_ap" not in approved_text


def test_forbidden_outputs_cover_cut_wave2_and_ap_map() -> None:
    forbidden_ids = {item["id"] for item in FORBIDDEN_REPORTING_OUTPUTS}
    assert "ap_map_additions" in forbidden_ids
    assert "strictness_curve" in forbidden_ids
    assert "dq_vs_sq_scatter" in forbidden_ids


def test_headline_policy_ranks_whole_section_pq() -> None:
    policy = HEADLINE_POLICY.lower()
    assert "whole-section pq" in policy
    assert "aji+" in policy
    assert "rank" in policy


def _minimal_eval_tree(root: Path) -> None:
    _write_json(
        root / "eval/yolo_PPL/instance_metrics.json",
        {
            **MINIMAL_INSTANCE_METRICS,
            "schema_version": 2,
            "variant": "PPL",
            "samples": [PQ_SAMPLE_ROW],
        },
    )


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
                    "samples": [PQ_SAMPLE_ROW],
                },
            )


def test_figure_bundle_filenames_are_not_forbidden() -> None:
    violations = _forbidden_filename_matches(set(FIGURE_BUNDLE_FILENAMES))
    assert not violations, violations


def test_build_reporting_bundle_records_contract_in_summary(tmp_path: Path) -> None:
    root = tmp_path / "GrainSeg"
    _minimal_eval_tree(root)
    out = tmp_path / "reporting"
    summary = build_reporting_bundle(root, out, render_figures=False)

    assert summary["headline_policy"] == HEADLINE_POLICY
    contract = summary["reporting_contract"]
    assert contract["wave1_approved"] == WAVE1_APPROVED_OUTPUTS
    assert contract["wave2_deferred"] == WAVE2_DEFERRED_OUTPUTS
    assert contract["cut_outputs"] == CUT_OUTPUTS

    payload = json.loads((out / "analysis_summary.json").read_text(encoding="utf-8"))
    written_tables = payload["written"]["derived_tables"]
    assert payload["written"]["derived_tables"] == summary["written"]["derived_tables"]
    assert "headline_pq_ranking.csv" in written_tables
    assert "thesis_ready_results.csv" in written_tables
    assert "whole_section_pq_matrix.csv" in written_tables
    assert "ppl_baseline_gain.csv" in written_tables
    assert "per_variant_winner.csv" not in written_tables
    assert payload["written"]["figures"] == []
    assert isinstance(payload["skipped"], list)
    assert isinstance(payload["skipped_optional"], list)
    required_skipped = {item["id"] for item in payload["skipped"]}
    assert "ppl_relative_diagnostic_heatmap" in required_skipped
    skipped_ids = {item["id"] for item in payload["skipped_optional"]}
    assert "precision_recall_diagnostic_map_iou75" in skipped_ids
    skipped_optional_by_id = {
        item["id"]: item["reason"] for item in payload["skipped_optional"]
    }
    assert (
        skipped_optional_by_id["precision_recall_diagnostic_map_iou75"]
        == PRECISION_RECALL_NOT_INFORMATIVE_SKIP_REASON
    )
    assert "pareto_plot" in skipped_ids


def test_build_reporting_bundle_does_not_write_forbidden_table_outputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "GrainSeg"
    _minimal_eval_tree(root)
    out = tmp_path / "reporting"
    build_reporting_bundle(root, out, render_figures=False)

    written_names = {
        path.name
        for path in out.rglob("*")
        if path.is_file() and path.name != "analysis_summary.json"
    }
    violations = _forbidden_filename_matches(written_names)
    assert not violations, violations


def test_build_reporting_bundle_does_not_write_forbidden_figure_outputs(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    root = tmp_path / "GrainSeg"
    _figure_ready_eval_tree(root)
    out = tmp_path / "reporting"
    build_reporting_bundle(root, out, render_figures=True)

    written_names = {
        path.name
        for path in out.rglob("*")
        if path.is_file() and path.name != "analysis_summary.json"
    }
    rendered_figures = written_names & set(FIGURE_BUNDLE_FILENAMES)
    assert rendered_figures, "expected headline figures from render path"
    violations = _forbidden_filename_matches(written_names)
    assert not violations, violations


@pytest.mark.parametrize(
    "forbidden_fragment",
    [
        "strictness_curve",
        "error_composition",
        "ap_map",
        "dq_vs_sq_scatter",
        "rank_stability",
    ],
)
def test_forbidden_filename_fragments_are_not_in_wave1_approved(
    forbidden_fragment: str,
) -> None:
    approved_text = json.dumps(WAVE1_APPROVED_OUTPUTS)
    assert forbidden_fragment not in approved_text.lower()
