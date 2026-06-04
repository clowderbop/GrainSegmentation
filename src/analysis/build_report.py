"""CLI: discover eval artifacts and build the reporting bundle on scratch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from analysis.derived_tables import (
    can_compare_yolo_and_unet_on_shared_inputs,
    headline_ranking_table,
    model_family_comparison_matrix_table,
    per_variant_winner_table,
    ppl_baseline_gain_table,
    thesis_ready_results_table,
    whole_section_pq_matrix_table,
)
from analysis.diagnostic_derivation import (
    failure_mode_classification_table,
    failure_mode_metrics_available,
    failure_mode_rules_markdown,
    pq_decomposition_metrics_available,
)
from analysis.discover import discover_eval_runs, discover_ultralytics_val
from analysis.figures import render_all_figures
from analysis.load_metrics import metrics_table_from_runs, ultralytics_val_table
from analysis.reporting_contract import (
    HEADLINE_POLICY,
    ReportingOutput,
    WAVE1_APPROVED_OUTPUTS,
    optional_wave1_outputs,
    reporting_contract_metadata,
    required_wave1_outputs,
)

MODEL_COMPARISON_OUTPUT_IDS = frozenset(
    {
        "per_variant_winner_table",
        "model_family_comparison_matrix",
    }
)
MODEL_COMPARISON_SKIP_REASON = (
    "YOLO and U-Net whole-section metrics are not both present for any "
    "shared input configuration"
)
NOT_IMPLEMENTED_SKIP_REASON = (
    "output not implemented in this Wave 1 reporting bundle run"
)
FIGURES_DISABLED_SKIP_REASON = "figure rendering disabled for this reporting run"
FIGURE_NOT_GENERATED_SKIP_REASON = (
    "figure not generated: required data missing or renderer produced no file"
)
NO_INSTANCE_ROWS_SKIP_REASON = "no instance metric rows were loaded"
MISSING_DQ_SQ_SKIP_REASON = (
    "required whole-section DQ and SQ fields missing or non-finite"
)

def _written_output_ids(
    derived_tables: list[str],
    figure_names: list[str],
) -> set[str]:
    """Map written bundle filenames back to contract output ids when known."""
    written_names = set(derived_tables) | set(figure_names)
    matched: set[str] = set()
    for item in WAVE1_APPROVED_OUTPUTS:
        patterns = item.get("filename_patterns", [])
        if any(pattern in name for name in written_names for pattern in patterns):
            matched.add(item["id"])
    return matched


def _figure_filename_patterns(item: ReportingOutput) -> list[str]:
    return [
        pattern
        for pattern in item.get("filename_patterns", [])
        if pattern.endswith(".png")
    ]


def _table_filename_patterns(item: ReportingOutput) -> list[str]:
    return [
        pattern
        for pattern in item.get("filename_patterns", [])
        if pattern.endswith(".csv")
    ]


def _skip_reason_for_required_output(
    output_id: str,
    item: ReportingOutput,
    *,
    render_figures: bool,
    has_instance_rows: bool,
    can_compare_models: bool,
    failure_mode_available: bool,
    pq_decomposition_available: bool,
) -> str:
    if output_id in MODEL_COMPARISON_OUTPUT_IDS and not can_compare_models:
        return MODEL_COMPARISON_SKIP_REASON

    if output_id == "failure_mode_classification" and has_instance_rows:
        if not failure_mode_available:
            return MISSING_DQ_SQ_SKIP_REASON

    if output_id == "pq_decomposition_grouped_bars" and has_instance_rows:
        if not pq_decomposition_available:
            return MISSING_DQ_SQ_SKIP_REASON

    patterns = item.get("filename_patterns", [])
    if not patterns:
        return NOT_IMPLEMENTED_SKIP_REASON

    figure_patterns = _figure_filename_patterns(item)
    table_patterns = _table_filename_patterns(item)

    if not has_instance_rows and (figure_patterns or table_patterns):
        return NO_INSTANCE_ROWS_SKIP_REASON

    if figure_patterns and not render_figures:
        return FIGURES_DISABLED_SKIP_REASON

    if figure_patterns and render_figures:
        return FIGURE_NOT_GENERATED_SKIP_REASON

    if table_patterns and output_id in MODEL_COMPARISON_OUTPUT_IDS:
        return MODEL_COMPARISON_SKIP_REASON

    return NOT_IMPLEMENTED_SKIP_REASON


def _skipped_required_outputs(
    written_ids: set[str],
    *,
    render_figures: bool,
    has_instance_rows: bool,
    can_compare_models: bool,
    failure_mode_available: bool,
    pq_decomposition_available: bool,
) -> list[dict[str, str]]:
    """Required contract outputs that were not written, with reasons."""
    skipped: list[dict[str, str]] = []
    for item in required_wave1_outputs():
        output_id = item["id"]
        if output_id in written_ids:
            continue
        skipped.append(
            {
                "id": output_id,
                "label": item["label"],
                "reason": _skip_reason_for_required_output(
                    output_id,
                    item,
                    render_figures=render_figures,
                    has_instance_rows=has_instance_rows,
                    can_compare_models=can_compare_models,
                    failure_mode_available=failure_mode_available,
                    pq_decomposition_available=pq_decomposition_available,
                ),
            }
        )
    return skipped


SCOPE_NOTE = (
    "Headline whole-section PQ ranks input configurations and compares YOLO vs U-Net "
    "on held-out test eval (sliding-window). DQ, SQ, thresholded F1, instance counts, "
    "and AJI+ in instance_metrics.csv are companion diagnostics for failure-mode review. "
    "Patch instance rows and the Ultralytics val panel are supporting metrics with "
    "different inference geometry; do not rank variants from patch metrics or AP/mAP alone."
)


def build_reporting_bundle(
    grainseg_root: Path,
    output_dir: Path,
    *,
    render_figures: bool = True,
    strict: bool = False,
) -> dict[str, Any]:
    grainseg_root = grainseg_root.resolve()
    output_dir = output_dir.resolve()
    derived_dir = output_dir / "derived"
    figures_dir = output_dir / "figures"
    derived_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_eval_runs(grainseg_root, strict=strict)
    missing: list[str] = []

    val_refs = discover_ultralytics_val(grainseg_root)
    instance_df = metrics_table_from_runs(runs)
    val_df = ultralytics_val_table(val_refs)

    instance_csv = derived_dir / "instance_metrics.csv"
    instance_df.to_csv(instance_csv, index=False)
    derived_tables = ["instance_metrics.csv"]

    if not instance_df.empty:
        ranking_csv = derived_dir / "headline_pq_ranking.csv"
        headline_ranking_table(instance_df).to_csv(ranking_csv, index=False)
        derived_tables.append("headline_pq_ranking.csv")

        thesis_csv = derived_dir / "thesis_ready_results.csv"
        thesis_ready_results_table(instance_df).to_csv(thesis_csv, index=False)
        derived_tables.append("thesis_ready_results.csv")

        matrix_csv = derived_dir / "whole_section_pq_matrix.csv"
        whole_section_pq_matrix_table(instance_df).to_csv(matrix_csv)
        derived_tables.append("whole_section_pq_matrix.csv")

        gain_csv = derived_dir / "ppl_baseline_gain.csv"
        ppl_baseline_gain_table(instance_df).to_csv(gain_csv, index=False)
        derived_tables.append("ppl_baseline_gain.csv")

        if can_compare_yolo_and_unet_on_shared_inputs(instance_df):
            winner_csv = derived_dir / "per_variant_winner.csv"
            per_variant_winner_table(instance_df).to_csv(winner_csv, index=False)
            derived_tables.append("per_variant_winner.csv")

            comparison_csv = derived_dir / "model_family_comparison_matrix.csv"
            model_family_comparison_matrix_table(instance_df).to_csv(comparison_csv)
            derived_tables.append("model_family_comparison_matrix.csv")

        if failure_mode_metrics_available(instance_df):
            failure_csv = derived_dir / "failure_mode_classification.csv"
            failure_mode_classification_table(instance_df).to_csv(
                failure_csv, index=False
            )
            derived_tables.append("failure_mode_classification.csv")

            rules_md = derived_dir / "failure_mode_classification_rules.md"
            rules_md.write_text(failure_mode_rules_markdown(), encoding="utf-8")
            derived_tables.append("failure_mode_classification_rules.md")

    if not val_df.empty:
        val_csv = derived_dir / "ultralytics_val.csv"
        val_df.to_csv(val_csv, index=False)
        derived_tables.append("ultralytics_val.csv")

    figure_names: list[str] = []
    if render_figures:
        if instance_df.empty:
            raise ValueError(
                "cannot render headline figures: no instance metric rows were loaded"
            )
        figure_names = render_all_figures(instance_df, val_df, figures_dir)

    written_figure_names = list(figure_names)
    written_ids = _written_output_ids(derived_tables, written_figure_names)
    can_compare_models = (
        not instance_df.empty
        and can_compare_yolo_and_unet_on_shared_inputs(instance_df)
    )
    skipped_required = _skipped_required_outputs(
        written_ids,
        render_figures=render_figures,
        has_instance_rows=not instance_df.empty,
        can_compare_models=can_compare_models,
        failure_mode_available=failure_mode_metrics_available(instance_df),
        pq_decomposition_available=pq_decomposition_metrics_available(instance_df),
    )
    skipped_optional: list[dict[str, str]] = []
    for item in optional_wave1_outputs():
        if item["id"] not in written_ids:
            skipped_optional.append(
                {
                    "id": item["id"],
                    "label": item["label"],
                    "reason": "optional output not generated in this reporting run",
                }
            )

    summary: dict[str, Any] = {
        "grainseg_root": str(grainseg_root),
        "output_dir": str(output_dir),
        "scope_note": SCOPE_NOTE,
        "headline_policy": HEADLINE_POLICY,
        "reporting_contract": reporting_contract_metadata(),
        "missing_artifacts": missing,
        "n_instance_rows": int(len(instance_df)),
        "n_ultralytics_val_rows": int(len(val_df)),
        "written": {
            "derived_tables": derived_tables,
            "figures": written_figure_names,
            "audits": [],
            "narratives": [],
        },
        "skipped": skipped_required,
        "skipped_optional": skipped_optional,
    }
    summary_path = output_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.build_report",
        description="Build post-eval reporting bundle from scratch eval artifacts.",
    )
    parser.add_argument(
        "--grainseg-root",
        type=Path,
        required=True,
        help="GrainSeg root on scratch (e.g. $SCRATCH/GrainSeg)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Reporting bundle output (default: {grainseg_root}/eval/reporting)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Write derived tables and summary only (no matplotlib)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected whole-section eval artifact is missing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    grainseg_root: Path = args.grainseg_root
    output_dir = args.output_dir or (grainseg_root / "eval" / "reporting")
    try:
        summary = build_reporting_bundle(
            grainseg_root,
            output_dir,
            render_figures=not args.no_figures,
            strict=args.strict,
        )
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
