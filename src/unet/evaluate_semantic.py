"""Evaluate U-Net semantic predictions against raster ground-truth masks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tifffile

from common.arg_errors import raise_cli_argument_error
from common.reporting import json_safe_for_dump
from common.samples import load_raster_mask, mask_extensions
from unet.extract_instances import list_semantic_predictions
from unet.prediction_cache import prediction_tiff_path
from unet.semantic_metrics import (
    build_semantic_eval_report,
    build_semantic_sample_row,
    compute_semantic_metrics_dict,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute semantic metrics from cached U-Net predictions.",
    )
    parser.add_argument("--semantic-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--mask-ext", default=None)
    parser.add_argument("--mask-stem-suffix", default="")
    parser.add_argument("--variant", default=None)
    parser.add_argument(
        "--unit",
        choices=("patch", "whole"),
        default="whole",
    )
    return parser


def _mask_path_for_sample(
    mask_dir: Path,
    sample_id: str,
    *,
    mask_ext: str | None,
    mask_stem_suffix: str,
) -> Path | None:
    for ext in mask_extensions(mask_ext):
        candidate = mask_dir / f"{sample_id}{mask_stem_suffix}{ext}"
        if candidate.is_file():
            return candidate
    return None


def run_evaluate_semantic(args: argparse.Namespace) -> None:
    if not args.mask_dir.is_dir():
        raise_cli_argument_error(f"mask-dir is not a directory: {args.mask_dir}")

    sample_ids = list_semantic_predictions(args.semantic_dir)
    if not sample_ids:
        print(f"ERROR: No predictions in {args.semantic_dir}", file=sys.stderr)
        sys.exit(1)

    sample_rows: list[dict] = []
    skipped = 0
    for sample_id in sample_ids:
        mask_path = _mask_path_for_sample(
            args.mask_dir,
            sample_id,
            mask_ext=args.mask_ext,
            mask_stem_suffix=args.mask_stem_suffix,
        )
        if mask_path is None:
            print(f"Warning: no GT mask for {sample_id}; skipping.", file=sys.stderr)
            skipped += 1
            continue

        pred = tifffile.imread(prediction_tiff_path(args.semantic_dir, sample_id))
        if pred.ndim == 3:
            pred = pred[0] if pred.shape[0] == 1 else pred[:, :, 0]
        gt = load_raster_mask(str(mask_path))
        metrics = compute_semantic_metrics_dict(pred.astype(int), gt)
        sample_rows.append(build_semantic_sample_row(sample_id, metrics))
        print(
            f"{sample_id}: mIoU={metrics['mean_iou']:.4f} "
            f"acc={metrics['pixel_accuracy']:.4f}"
        )

    if not sample_rows:
        print("ERROR: No samples with raster GT masks.", file=sys.stderr)
        sys.exit(1)

    report = build_semantic_eval_report(
        variant=args.variant,
        unit=args.unit,
        samples=sample_rows,
        extras={
            "mask_dir": str(args.mask_dir.resolve()),
            "semantic_dir": str(args.semantic_dir.resolve()),
            "skipped_no_mask": skipped,
        },
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(json_safe_for_dump(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote semantic metrics to {args.output_json}")


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    run_evaluate_semantic(args)


if __name__ == "__main__":
    main()
