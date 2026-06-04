"""CLI: choose CC vs tuned watershed on train by whole-section PQ (issue 04)."""

from __future__ import annotations

import argparse
from pathlib import Path

from common.variants import all_variant_names
from unet.extraction_method_selection import (
    select_train_extraction_method_from_eval_dirs,
    select_train_extraction_method_from_reports,
    write_extraction_method_selection_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select U-Net train extraction method (connected components vs watershed) "
            "by mean whole-section PQ across registry variants."
        )
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--cc-eval-dir",
        type=Path,
        help="Train eval output dir for connected-components (e.g. eval/instance_val_cc).",
    )
    src.add_argument(
        "--cc-report-json",
        type=Path,
        help="Single-variant train whole-section CC instance metrics report.",
    )
    parser.add_argument(
        "--watershed-eval-dir",
        type=Path,
        help="Train eval output dir for tuned watershed (required with --cc-eval-dir).",
    )
    parser.add_argument(
        "--watershed-report-json",
        type=Path,
        help="Single-variant train whole-section watershed instance metrics report.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Registry variants to aggregate (default: all from variants.yaml).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Write selection audit JSON (selected method + both method bundles).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    variants = tuple(args.variants) if args.variants else all_variant_names()

    if args.cc_eval_dir is not None:
        if args.watershed_eval_dir is None:
            raise SystemExit("--watershed-eval-dir is required with --cc-eval-dir")
        selection = select_train_extraction_method_from_eval_dirs(
            cc_eval_dir=args.cc_eval_dir,
            watershed_eval_dir=args.watershed_eval_dir,
            variant_names=variants,
        )
    else:
        if args.watershed_report_json is None:
            raise SystemExit(
                "--watershed-report-json is required with --cc-report-json"
            )
        selection = select_train_extraction_method_from_reports(
            cc_report_path=args.cc_report_json,
            watershed_report_path=args.watershed_report_json,
        )

    write_extraction_method_selection_json(args.output_json, selection)
    print(
        f"Selected {selection.selected_method} "
        f"(mean train whole-section PQ={selection.objective_pq:.6f})"
    )
    print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
