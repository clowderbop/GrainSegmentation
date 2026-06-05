
from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from common.geometry import load_image_space_polygons
from common.ground_truth import polygons_to_instance_map
from common.image_io import (
    load_tiff_single_channel_mask,
    validate_input_images,
    validate_semantic_labels,
)
from common.manifest_io import collect_manifest_unet_samples, load_dataset_manifest
from common.samples import load_rgb_image
from common.arg_errors import raise_cli_argument_error
from unet.extraction_tune_scoring import (
    WatershedParamSet,
    mean_train_pq_for_watershed_params,
    select_best_watershed_tune_row,
    watershed_best_json_summary,
    watershed_per_sample_columns,
    watershed_tune_fieldnames,
    watershed_tune_row,
)


def _sanitize_csv_key(sample_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", sample_id)


def _load_pred_tiff(path: Path) -> np.ndarray:
    arr = load_tiff_single_channel_mask(path)
    return validate_semantic_labels(arr, str(path))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gt-gpkg", required=True)
    parser.add_argument("--num-inputs", type=int, default=None)
    parser.add_argument(
        "--image-suffixes",
        nargs="+",
        default=["_PPL", "_PPX1", "_PPX2", "_PPX3", "_PPX4", "_PPX5", "_PPX6"],
    )

    parser.add_argument(
        "--min-distance",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        )
    parser.add_argument(
        "--boundary-dilate-iter",
        type=int,
        nargs="+",
        default=[0, 1],
        )
    parser.add_argument(
        "--watershed-connectivity",
        type=int,
        nargs="+",
        default=[1, 2],
        choices=[1, 2],
        )
    parser.add_argument(
        "--min-area-px",
        type=int,
        nargs="+",
        default=[0],
        )
    parser.add_argument(
        "--exclude-border",
        type=int,
        nargs="+",
        default=[0, 1],
        choices=[0, 1],
        )
    parser.add_argument(
        "--ridge-level",
        type=float,
        nargs="*",
        default=None,
        )

    parser.add_argument(
        "--output-csv",
        required=True,
        )
    parser.add_argument(
        "--output-json",
        default=None,
        )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        )

    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _validate_tune_args(args, parser)
    return args


def _validate_tune_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser | None = None
) -> None:
    if args.num_inputs is not None and args.num_inputs not in {1, 2, 7}:
        raise_cli_argument_error(
            "num_inputs must be one of: 1, 2, 7", parser=parser
        )
    if not Path(args.gt_gpkg).is_file():
        raise_cli_argument_error(f"gt-gpkg is not a file: {args.gt_gpkg}", parser=parser)
    if any(v < 1 for v in args.min_distance):
        raise_cli_argument_error(
            "min_distance values must be >= 1 (matches extract_instances watershed)",
            parser=parser,
        )
    for name, vals in (
        ("boundary_dilate_iter", args.boundary_dilate_iter),
        ("min_area_px", args.min_area_px),
    ):
        if any(v < 0 for v in vals):
            raise_cli_argument_error(f"{name} values must be >= 0", parser=parser)
    if args.max_samples is not None and args.max_samples <= 0:
        raise_cli_argument_error("max_samples must be positive", parser=parser)
    preds_dir = Path(args.preds_dir)
    if not preds_dir.is_dir():
        raise_cli_argument_error(
            f"preds-dir is not a directory: {preds_dir.resolve()}",
            parser=parser,
        )


def _ridge_level_grid(args: argparse.Namespace) -> list[float | None]:
    if args.ridge_level is None:
        return [None]
    if len(args.ridge_level) == 0:
        return [None]
    return list(args.ridge_level)


def _resolve_watershed_samples(args: argparse.Namespace) -> list[dict]:
    doc = load_dataset_manifest(args.manifest)
    if args.num_inputs is not None:
        expected = len(doc.samples[0].images or ())
        if args.num_inputs != expected:
            raise_cli_argument_error(
                f"--num-inputs {args.num_inputs} != manifest ({expected})"
            )
    args.num_inputs = len(doc.samples[0].images or ())
    return collect_manifest_unet_samples(doc)


def _collect_samples(
    args: argparse.Namespace,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    """Materialize GT instances in scene/full-raster coords only (never patch-stem shifted)."""
    samples = _resolve_watershed_samples(args)
    if not samples:
        raise SystemExit("No samples found.")
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    gpkg_path = Path(args.gt_gpkg).resolve()
    gt_scene_polygons = load_image_space_polygons(gpkg_path)
    sample_ids: list[str] = []
    true_instances: list[np.ndarray] = []
    pred_semantic: list[np.ndarray] = []

    preds_dir = Path(args.preds_dir).resolve()
    for sample in samples:
        sid = sample["id"]
        pred_path = preds_dir / f"{sid}_pred.tif"
        if not pred_path.is_file():
            raise SystemExit(f"Missing prediction file: {pred_path}")
        print(f"Loading pred: {pred_path}")
        images = [load_rgb_image(p) for p in sample["images"]]
        height, width = validate_input_images(images)
        pred_arr = _load_pred_tiff(pred_path)
        if pred_arr.shape != (height, width):
            raise ValueError(
                f"Pred shape {pred_arr.shape} != image shape {(height, width)} for {sid}"
            )
        sample_ids.append(sid)
        true_instances.append(
            polygons_to_instance_map(
                gt_scene_polygons,
                height=height,
                width=width,
            )
        )
        pred_semantic.append(pred_arr)

    return sample_ids, true_instances, pred_semantic


def _format_ridge_level(ridge_level: float | None) -> str:
    return "auto" if ridge_level is None else f"{ridge_level:g}"


def _format_param_set(params: WatershedParamSet) -> str:
    return (
        f"min_dist={params.min_distance}, dilate={params.boundary_dilate_iter}, "
        f"conn={params.watershed_connectivity}, min_area={params.min_area_px}, "
        f"exclude_border={params.exclude_border}, ridge={_format_ridge_level(params.ridge_level)}"
    )


def _iter_param_grid(args: argparse.Namespace) -> Iterable[WatershedParamSet]:
    ridge_levels = _ridge_level_grid(args)
    for tup in itertools.product(
        args.min_distance,
        args.boundary_dilate_iter,
        args.watershed_connectivity,
        args.min_area_px,
        args.exclude_border,
        ridge_levels,
    ):
        md, bdi, wsc, mapx, exb, ridge = tup
        yield WatershedParamSet(
            min_distance=int(md),
            boundary_dilate_iter=int(bdi),
            watershed_connectivity=int(wsc),
            min_area_px=int(mapx),
            exclude_border=bool(int(exb)),
            ridge_level=ridge,
        )


def main() -> None:
    args = _parse_args()
    sample_ids, true_instances, pred_semantic = _collect_samples(args)
    if args.num_inputs is None:
        raise_cli_argument_error("Could not determine num_inputs")

    ridge_levels = _ridge_level_grid(args)
    grid_size = (
        len(args.min_distance)
        * len(args.boundary_dilate_iter)
        * len(args.watershed_connectivity)
        * len(args.min_area_px)
        * len(args.exclude_border)
        * len(ridge_levels)
    )
    print(f"Grid size: {grid_size} combinations on {len(sample_ids)} sample(s).")

    grid_rows: list[dict[str, Any]] = []

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = watershed_tune_fieldnames(
        sample_ids, sanitize_sample_id=_sanitize_csv_key
    )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for combo_idx, params in enumerate(_iter_param_grid(args), start=1):
            print(
                f"[{combo_idx}/{grid_size}] scoring "
                f"({_format_param_set(params)}) ..."
            )
            t0 = time.perf_counter()
            mean_pq, per_sample_pq = mean_train_pq_for_watershed_params(
                true_instances, pred_semantic, params
            )
            elapsed = time.perf_counter() - t0
            mean_pq_value = float(mean_pq["pq"])
            print(
                f"[{combo_idx}/{grid_size}] mean_pq={mean_pq_value:.6f} "
                f"({_format_param_set(params)}) {elapsed:.2f}s"
            )
            row = watershed_tune_row(
                params,
                mean_pq,
                per_sample_pq=watershed_per_sample_columns(
                    sample_ids,
                    per_sample_pq,
                    sanitize_sample_id=_sanitize_csv_key,
                ),
            )
            writer.writerow(row)
            f.flush()
            grid_rows.append(row)

    best_row = select_best_watershed_tune_row(grid_rows)
    best_params = WatershedParamSet(
        min_distance=int(best_row["min_distance"]),
        boundary_dilate_iter=int(best_row["boundary_dilate_iter"]),
        watershed_connectivity=int(best_row["watershed_connectivity"]),
        min_area_px=int(best_row["min_area_px"]),
        exclude_border=bool(int(best_row["exclude_border"])),
        ridge_level=(
            None
            if best_row["ridge_level"] == ""
            else float(best_row["ridge_level"])
        ),
    )
    best_mean_pq = float(best_row["mean_pq"])

    print("\nBest watershed parameters (max mean train whole-section PQ):")
    print(f"  min_distance: {best_params.min_distance}")
    print(f"  boundary_dilate_iter: {best_params.boundary_dilate_iter}")
    print(f"  watershed_connectivity: {best_params.watershed_connectivity}")
    print(f"  min_area_px: {best_params.min_area_px}")
    print(f"  exclude_border: {best_params.exclude_border}")
    print(f"  ridge_level: {_format_ridge_level(best_params.ridge_level)}")
    print(f"  mean_pq: {best_mean_pq:.6f}")
    print(f"\nWrote grid results to {out_path}")

    if args.output_json:
        summary = watershed_best_json_summary(
            best_row,
            best_params,
            sample_ids,
            sanitize_sample_id=_sanitize_csv_key,
        )
        jp = Path(args.output_json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        with jp.open("w") as jf:
            json.dump(summary, jf, indent=2)
        print(f"Wrote summary to {jp}")


if __name__ == "__main__":
    main()
