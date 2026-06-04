"""CLI: discover eval artifacts and build the reporting bundle on scratch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from analysis.discover import discover_eval_runs, discover_ultralytics_val
from analysis.figures import render_all_figures
from analysis.load_metrics import metrics_table_from_runs, ultralytics_val_table
from analysis.reporting_contract import (
    HEADLINE_POLICY,
    WAVE1_APPROVED_OUTPUTS,
    optional_wave1_outputs,
    reporting_contract_metadata,
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
    skipped_optional: list[dict[str, str]] = []
    for item in optional_wave1_outputs():
        if item["id"] not in _written_output_ids(derived_tables, written_figure_names):
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
