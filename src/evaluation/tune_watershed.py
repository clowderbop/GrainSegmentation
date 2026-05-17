
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.geometry import load_image_space_polygons
from common.ground_truth import scene_polygons_to_patch_instance_map
from common.image_io import (
    load_tiff_single_channel_mask,
    validate_semantic_labels,
)
from common.samples import list_samples, load_rgb_image, load_raster_mask
from evaluation.arg_errors import raise_cli_argument_error
from evaluation.instance_masks import semantic_to_instance_label_map_watershed
from evaluation.metrics import compute_aji
from evaluation.sample_checks import semantic_mask_after_sample_validation


def _validate_pred_semantic(pred: np.ndarray, mask_path: str) -> np.ndarray:
    return validate_semantic_labels(pred, mask_path)


def _sanitize_csv_key(sample_id: str) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "_", sample_id)


def _load_pred_tiff(path: Path) -> np.ndarray:
    arr = load_tiff_single_channel_mask(path)
    return validate_semantic_labels(arr, str(path))


@dataclass(frozen=True)
class WatershedParamSet:
    min_distance: int
    boundary_dilate_iter: int
    watershed_connectivity: int
    min_area_px: int
    exclude_border: bool
    ridge_level: float | None


def mean_aji_for_watershed_params(
    true_instances_per_sample: Sequence[np.ndarray],
    pred_semantic_per_sample: Sequence[np.ndarray],
    params: WatershedParamSet,
    *,
    interior_class: int = 1,
    boundary_class: int = 2,
) -> tuple[float, list[float]]:
    if len(true_instances_per_sample) != len(pred_semantic_per_sample):
        raise ValueError("true and pred lists must have the same length")
    kw: dict[str, Any] = dict(
        interior_class=interior_class,
        boundary_class=boundary_class,
        min_distance=params.min_distance,
        boundary_dilate_iter=params.boundary_dilate_iter,
        watershed_connectivity=params.watershed_connectivity,
        min_area_px=params.min_area_px,
        exclude_border=params.exclude_border,
    )
    if params.ridge_level is not None:
        kw["ridge_level"] = params.ridge_level

    ajis: list[float] = []
    for ti, pred in zip(true_instances_per_sample, pred_semantic_per_sample):
        pi = semantic_to_instance_label_map_watershed(pred, **kw)
        ajis.append(float(compute_aji(ti, pi)))
    return float(np.mean(ajis)), ajis


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--model-path",
        default=None,
        )
    src.add_argument(
        "--preds-dir",
        default=None,
        )

    parser.add_argument(
        "--image-dir", required=True, )
    parser.add_argument(
        "--mask-dir", required=True, )
    parser.add_argument(
        "--gt-gpkg",
        required=True,
        )
    parser.add_argument(
        "--gt-origin",
        choices=("patch_stem", "whole_image"),
        default="whole_image",
        help="How to translate GPKG scene coordinates into patch image space "
        "(default whole_image preserves legacy tune_watershed behavior).",
    )
    parser.add_argument("--num-inputs", type=int, default=7)
    parser.add_argument(
        "--image-suffixes",
        nargs="+",
        default=["_PPL", "_PPX1", "_PPX2", "_PPX3", "_PPX4", "_PPX5", "_PPX6"],
    )
    parser.add_argument("--mask-ext", default=None)
    parser.add_argument("--mask-stem-suffix", default="")
    parser.add_argument("--patch-size", type=int, default=3008)
    parser.add_argument("--stride", type=int, default=1504)
    parser.add_argument("--batch-size", type=int, default=4)

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

    args = parser.parse_args()
    _validate_tune_args(args, parser)
    return args


def _validate_tune_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser | None = None
) -> None:
    if args.num_inputs not in {1, 2, 7}:
        raise_cli_argument_error(
            "num_inputs must be one of: 1, 2, 7", parser=parser
        )
    if len(args.image_suffixes) < args.num_inputs:
        raise_cli_argument_error(
            "image_suffixes must provide at least num_inputs suffixes",
            parser=parser,
        )
    if args.patch_size <= 0 or args.stride <= 0:
        raise_cli_argument_error("patch_size and stride must be > 0", parser=parser)
    if args.stride > args.patch_size:
        raise_cli_argument_error("stride must be <= patch_size", parser=parser)
    if args.batch_size <= 0:
        raise_cli_argument_error("batch_size must be > 0", parser=parser)
    if not Path(args.gt_gpkg).is_file():
        raise_cli_argument_error(f"gt-gpkg is not a file: {args.gt_gpkg}", parser=parser)
    for name, vals in (
        ("min_distance", args.min_distance),
        ("boundary_dilate_iter", args.boundary_dilate_iter),
        ("min_area_px", args.min_area_px),
    ):
        if any(v < 0 for v in vals):
            raise_cli_argument_error(f"{name} values must be >= 0", parser=parser)
    if args.max_samples is not None and args.max_samples <= 0:
        raise_cli_argument_error("max_samples must be positive", parser=parser)


def _ridge_level_grid(args: argparse.Namespace) -> list[float | None]:
    if args.ridge_level is None:
        return [None]
    if len(args.ridge_level) == 0:
        return [None]
    return list(args.ridge_level)


def _collect_samples(
    args: argparse.Namespace,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    samples = list_samples(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_suffixes=args.image_suffixes,
        mask_ext=args.mask_ext,
        mask_stem_suffix=args.mask_stem_suffix,
        num_inputs=args.num_inputs,
    )
    if not samples:
        raise SystemExit("No samples found.")
    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    gpkg_path = Path(args.gt_gpkg).resolve()
    gt_scene_polygons = load_image_space_polygons(gpkg_path)
    sample_ids: list[str] = []
    true_instances: list[np.ndarray] = []
    pred_semantic: list[np.ndarray] = []

    if args.model_path:
        import tensorflow as tf

        from training.model import weighted_crossentropy
        from evaluation.inference import predict_full_image

        print(f"Loading model from {args.model_path}...")
        model = tf.keras.models.load_model(
            args.model_path,
            custom_objects={"weighted_crossentropy": weighted_crossentropy},
        )

        for sample in samples:
            sid = sample["id"]
            print(f"Inference: {sid}")
            images = [load_rgb_image(p) for p in sample["images"]]
            if len(images) != args.num_inputs:
                raise ValueError("Mismatch between num_inputs and loaded images.")
            true_mask = semantic_mask_after_sample_validation(
                images, load_raster_mask(sample["mask"]), sample["mask"]
            )
            height, width = true_mask.shape
            pred_classes, _ = predict_full_image(
                model=model,
                inputs=tuple(images),
                patch_size=args.patch_size,
                stride=args.stride,
                batch_size=args.batch_size,
            )
            sample_ids.append(sid)
            true_instances.append(
                scene_polygons_to_patch_instance_map(
                    gt_scene_polygons,
                    sample_id=sid,
                    height=height,
                    width=width,
                    gt_origin=args.gt_origin,
                )
            )
            pred_semantic.append(_validate_pred_semantic(pred_classes, sid))
    else:
        preds_dir = Path(args.preds_dir).resolve()
        if not preds_dir.is_dir():
            raise SystemExit(f"preds-dir is not a directory: {preds_dir}")

        for sample in samples:
            sid = sample["id"]
            pred_path = preds_dir / f"{sid}_pred.tif"
            if not pred_path.is_file():
                raise SystemExit(f"Missing prediction file: {pred_path}")
            print(f"Loading pred: {pred_path}")
            img0 = load_rgb_image(sample["images"][0])
            images = [img0]
            true_mask = semantic_mask_after_sample_validation(
                images, load_raster_mask(sample["mask"]), sample["mask"]
            )
            height, width = true_mask.shape
            pred_arr = _load_pred_tiff(pred_path)
            if pred_arr.shape != true_mask.shape:
                raise ValueError(
                    f"Pred shape {pred_arr.shape} != mask shape {true_mask.shape} for {sid}"
                )
            sample_ids.append(sid)
            true_instances.append(
                scene_polygons_to_patch_instance_map(
                    gt_scene_polygons,
                    sample_id=sid,
                    height=height,
                    width=width,
                    gt_origin=args.gt_origin,
                )
            )
            pred_semantic.append(pred_arr)

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

    best_mean = -1.0
    best_params: WatershedParamSet | None = None
    best_per_sample: list[float] | None = None

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "min_distance",
        "boundary_dilate_iter",
        "watershed_connectivity",
        "min_area_px",
        "exclude_border",
        "ridge_level",
        "mean_aji",
    ] + [f"aji__{_sanitize_csv_key(sid)}" for sid in sample_ids]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for params in _iter_param_grid(args):
            mean_aji, per = mean_aji_for_watershed_params(
                true_instances, pred_semantic, params
            )
            row: dict[str, Any] = {
                "min_distance": params.min_distance,
                "boundary_dilate_iter": params.boundary_dilate_iter,
                "watershed_connectivity": params.watershed_connectivity,
                "min_area_px": params.min_area_px,
                "exclude_border": int(params.exclude_border),
                "ridge_level": (
                    "" if params.ridge_level is None else f"{params.ridge_level:g}"
                ),
                "mean_aji": f"{mean_aji:.8f}",
            }
            for sid, a in zip(sample_ids, per):
                row[f"aji__{_sanitize_csv_key(sid)}"] = f"{a:.8f}"
            writer.writerow(row)

            if mean_aji > best_mean:
                best_mean = mean_aji
                best_params = params
                best_per_sample = per

    assert best_params is not None and best_per_sample is not None
    print("\nBest watershed parameters (max mean AJI):")
    print(f"  min_distance: {best_params.min_distance}")
    print(f"  boundary_dilate_iter: {best_params.boundary_dilate_iter}")
    print(f"  watershed_connectivity: {best_params.watershed_connectivity}")
    print(f"  min_area_px: {best_params.min_area_px}")
    print(f"  exclude_border: {best_params.exclude_border}")
    print(
        f"  ridge_level: {'auto' if best_params.ridge_level is None else best_params.ridge_level}"
    )
    print(f"  mean_aji: {best_mean:.6f}")
    for sid, a in zip(sample_ids, best_per_sample):
        print(f"    {sid}: {a:.6f}")
    print(f"\nWrote grid results to {out_path}")

    if args.output_json:
        summary = {
            "best_mean_aji": best_mean,
            "best_params": {
                "min_distance": best_params.min_distance,
                "boundary_dilate_iter": best_params.boundary_dilate_iter,
                "watershed_connectivity": best_params.watershed_connectivity,
                "min_area_px": best_params.min_area_px,
                "exclude_border": best_params.exclude_border,
                "ridge_level": best_params.ridge_level,
            },
            "best_per_sample": {
                sid: float(a) for sid, a in zip(sample_ids, best_per_sample)
            },
            "sample_ids": sample_ids,
        }
        jp = Path(args.output_json)
        jp.parent.mkdir(parents=True, exist_ok=True)
        with jp.open("w") as jf:
            json.dump(summary, jf, indent=2)
        print(f"Wrote summary to {jp}")


if __name__ == "__main__":
    main()
