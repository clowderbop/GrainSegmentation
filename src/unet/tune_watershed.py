
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
from common.reporting import count_instances
from common.image_io import load_tiff_single_channel_mask, validate_semantic_labels
from common.manifest_io import collect_manifest_unet_samples, load_dataset_manifest
from common.arg_errors import raise_cli_argument_error
from common.merged_view_pq import (
    MERGED_VIEW_PQ_RESULT_KEYS,
    coerce_merged_view_pq_value,
)
from unet.extraction_tune_scoring import (
    WatershedParamSet,
    format_merged_view_pq_audit_line,
    format_watershed_param_set,
    format_watershed_ridge_level,
    mean_train_pq_for_watershed_params,
    select_best_watershed_tune_row,
    watershed_best_json_summary,
    watershed_per_sample_columns,
    watershed_tune_fieldnames,
    watershed_tune_row,
)
from unet.watershed_tune_grid import (
    DEFAULT_BOUNDARY_DILATE_ITER,
    DEFAULT_EXCLUDE_BORDER,
    DEFAULT_MIN_AREA_PX,
    DEFAULT_MIN_DISTANCE,
    DEFAULT_WATERSHED_CONNECTIVITY,
)


def _log(*parts: object) -> None:
    print(*parts, flush=True)


def _mean_audit_line_from_tune_row(row: dict[str, Any]) -> str:
    return format_merged_view_pq_audit_line(
        {
            key: coerce_merged_view_pq_value(key, row[f"mean_{key}"])
            for key in MERGED_VIEW_PQ_RESULT_KEYS
        }
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
        default=list(DEFAULT_MIN_DISTANCE),
        )
    parser.add_argument(
        "--boundary-dilate-iter",
        type=int,
        nargs="+",
        default=list(DEFAULT_BOUNDARY_DILATE_ITER),
        )
    parser.add_argument(
        "--watershed-connectivity",
        type=int,
        nargs="+",
        default=list(DEFAULT_WATERSHED_CONNECTIVITY),
        choices=[1, 2],
        )
    parser.add_argument(
        "--min-area-px",
        type=int,
        nargs="+",
        default=list(DEFAULT_MIN_AREA_PX),
        )
    parser.add_argument(
        "--exclude-border",
        type=int,
        nargs="+",
        default=list(DEFAULT_EXCLUDE_BORDER),
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
        _log(f"Loading pred: {pred_path}")
        pred_arr = _load_pred_tiff(pred_path)
        height, width = pred_arr.shape
        gt_map = polygons_to_instance_map(
            gt_scene_polygons,
            height=height,
            width=width,
        )
        if gt_map.shape != pred_arr.shape:
            raise ValueError(
                f"GT shape {gt_map.shape} does not match prediction shape "
                f"{pred_arr.shape} for sample {sid!r}"
            )
        sample_ids.append(sid)
        true_instances.append(gt_map)
        pred_semantic.append(pred_arr)
        _log(
            f"  {sid}: {width}×{height}, GT={count_instances(gt_map)} instances"
        )

    return sample_ids, true_instances, pred_semantic


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
    _log(f"Grid size: {grid_size} combinations on {len(sample_ids)} sample(s).")
    _log(
        "Grid axes: "
        f"min_distance={list(args.min_distance)}, "
        f"boundary_dilate_iter={list(args.boundary_dilate_iter)}, "
        f"watershed_connectivity={list(args.watershed_connectivity)}, "
        f"min_area_px={list(args.min_area_px)}, "
        f"exclude_border={list(args.exclude_border)}, "
        f"ridge_level={list(ridge_levels)}"
    )

    grid_rows: list[dict[str, Any]] = []
    best_so_far_pq: float | None = None
    best_so_far_idx: int | None = None

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = watershed_tune_fieldnames(
        sample_ids, sanitize_sample_id=_sanitize_csv_key
    )

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for combo_idx, params in enumerate(_iter_param_grid(args), start=1):
            _log(
                f"[{combo_idx}/{grid_size}] scoring "
                f"({format_watershed_param_set(params)}) …"
            )
            t0 = time.perf_counter()
            mean_pq, per_sample_pq = mean_train_pq_for_watershed_params(
                true_instances,
                pred_semantic,
                params,
                sample_ids=sample_ids,
                log=True,
            )
            elapsed = time.perf_counter() - t0
            mean_pq_value = float(mean_pq["pq"])
            if best_so_far_pq is None or mean_pq_value > best_so_far_pq:
                best_so_far_pq = mean_pq_value
                best_so_far_idx = combo_idx
            best_note = (
                f" | best: {best_so_far_pq:.6f} @ #{best_so_far_idx}"
                if best_so_far_pq is not None and best_so_far_idx is not None
                else ""
            )
            _log(
                f"[{combo_idx}/{grid_size}] mean "
                f"{format_merged_view_pq_audit_line(mean_pq)} "
                f"({format_watershed_param_set(params)}) {elapsed:.1f}s{best_note}"
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
    _log("\nBest watershed parameters (max mean train whole-section PQ):")
    _log(f"  min_distance: {best_params.min_distance}")
    _log(f"  boundary_dilate_iter: {best_params.boundary_dilate_iter}")
    _log(f"  watershed_connectivity: {best_params.watershed_connectivity}")
    _log(f"  min_area_px: {best_params.min_area_px}")
    _log(f"  exclude_border: {best_params.exclude_border}")
    _log(f"  ridge_level: {format_watershed_ridge_level(best_params.ridge_level)}")
    _log(f"  {_mean_audit_line_from_tune_row(best_row)}")
    _log(f"\nWrote grid results to {out_path}")

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
        _log(f"Wrote summary to {jp}")


if __name__ == "__main__":
    main()
