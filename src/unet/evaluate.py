"""Evaluate U-Net semantic predictions with instance-level metrics."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from common.arg_errors import raise_cli_argument_error
from common.geometry import load_image_space_polygons
from common.ground_truth import GtOriginMode, scene_polygons_to_patch_instance_map
from common.metrics import compute_aji, compute_instance_metrics_dict
from common.reporting import (
    build_instance_eval_report,
    build_sample_row,
    count_instances,
    json_safe_for_dump,
)
from common.samples import list_samples, load_rgb_image, load_raster_mask
from common.semantic_instance import semantic_to_instance_label_map
from unet.instance_masks import semantic_to_instance_label_map_watershed
from unet.sample_checks import semantic_mask_after_sample_validation

MODEL_TYPE = "unet"
PREDICTION_CACHE_SCHEMA_VERSION = 1


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run U-Net inference and compute instance segmentation metrics.",
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--mask-dir", default=None, type=Path)
    parser.add_argument("--gt-gpkg", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--save-predictions-dir", default=None, type=Path)
    parser.add_argument(
        "--gt-origin",
        choices=("patch_stem", "whole_image"),
        default=None,
        help="GPKG-to-raster alignment. Default: patch_stem for --unit patch, "
        "whole_image for --unit whole.",
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
        "--instance-method",
        choices=("cc", "watershed"),
        default="cc",
    )
    parser.add_argument("--watershed-min-distance", type=int, default=1)
    parser.add_argument("--watershed-boundary-dilate-iter", type=int, default=0)
    parser.add_argument("--watershed-connectivity", type=int, choices=(1, 2), default=1)
    parser.add_argument("--watershed-min-area-px", type=int, default=0)
    parser.add_argument(
        "--watershed-exclude-border",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--watershed-ridge-level", type=float, default=None)
    parser.add_argument(
        "--variant",
        default=None,
        help="Input variant label for the metrics JSON (e.g. PPL, PPL+AllPPX).",
    )
    parser.add_argument(
        "--unit",
        choices=("patch", "whole"),
        default="patch",
        help="Evaluation scope recorded in the metrics JSON.",
    )
    return parser


def _validate_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser | None = None
) -> None:
    if args.num_inputs not in {1, 2, 7}:
        raise_cli_argument_error("num_inputs must be one of: 1, 2, 7", parser=parser)
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
    if args.watershed_min_distance < 1:
        raise_cli_argument_error("watershed_min_distance must be >= 1", parser=parser)
    if args.watershed_boundary_dilate_iter < 0:
        raise_cli_argument_error(
            "watershed_boundary_dilate_iter must be >= 0", parser=parser
        )
    if args.watershed_min_area_px < 0:
        raise_cli_argument_error("watershed_min_area_px must be >= 0", parser=parser)
    if args.watershed_ridge_level is not None and not np.isfinite(
        args.watershed_ridge_level
    ):
        raise_cli_argument_error(
            "watershed_ridge_level must be finite when set", parser=parser
        )
    if not args.gt_gpkg.is_file():
        raise_cli_argument_error(
            f"gt-gpkg is not a file: {args.gt_gpkg}", parser=parser
        )
    if args.mask_dir is not None and not args.mask_dir.is_dir():
        raise_cli_argument_error(
            f"mask-dir is not a directory: {args.mask_dir}", parser=parser
        )


def _effective_gt_origin(args: argparse.Namespace) -> GtOriginMode:
    if args.gt_origin is not None:
        return args.gt_origin
    return "whole_image" if args.unit == "whole" else "patch_stem"


def _prediction_tiff_path(save_dir: Path, sample_id: str) -> Path:
    return save_dir / f"{sample_id}_pred.tif"


def _prediction_meta_path(save_dir: Path, sample_id: str) -> Path:
    return save_dir / f"{sample_id}_pred.meta.json"


def _prediction_cache_record(args: argparse.Namespace) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": PREDICTION_CACHE_SCHEMA_VERSION,
        "model_path": str(args.model_path.resolve()),
        "patch_size": args.patch_size,
        "stride": args.stride,
        "batch_size": args.batch_size,
        "num_inputs": args.num_inputs,
        "image_suffixes": list(args.image_suffixes),
        "instance_method": args.instance_method,
    }
    if args.instance_method == "watershed":
        record.update(
            {
                "watershed_min_distance": args.watershed_min_distance,
                "watershed_boundary_dilate_iter": args.watershed_boundary_dilate_iter,
                "watershed_connectivity": args.watershed_connectivity,
                "watershed_min_area_px": args.watershed_min_area_px,
                "watershed_exclude_border": args.watershed_exclude_border,
                "watershed_ridge_level": args.watershed_ridge_level,
            }
        )
    return record


def _validate_prediction_cache(meta: dict[str, Any], args: argparse.Namespace) -> None:
    expected = _prediction_cache_record(args)
    if meta.get("schema_version") != PREDICTION_CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Cache schema_version {meta.get('schema_version')!r} != "
            f"{PREDICTION_CACHE_SCHEMA_VERSION}"
        )
    for key, expected_value in expected.items():
        if meta.get(key) != expected_value:
            raise ValueError(
                f"Cache mismatch for {key!r}: cached {meta.get(key)!r} != "
                f"current {expected_value!r}"
            )


def _load_cached_prediction_tiff(
    path: Path, expected_hw: tuple[int, int]
) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            raise ValueError(f"Cached prediction must be single-channel TIFF: {path}")
    if arr.ndim != 2:
        raise ValueError(f"Cached prediction must be 2D: {path}")
    if arr.shape != expected_hw:
        raise ValueError(
            f"Cached prediction shape {arr.shape} does not match image shape "
            f"{expected_hw}: {path}"
        )
    return arr.astype(np.int32)


def _instances_for_metrics(
    semantic: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    if args.instance_method == "cc":
        return semantic_to_instance_label_map(semantic, min_area_px=0)
    return semantic_to_instance_label_map_watershed(
        semantic,
        min_distance=args.watershed_min_distance,
        boundary_dilate_iter=args.watershed_boundary_dilate_iter,
        watershed_connectivity=args.watershed_connectivity,
        min_area_px=args.watershed_min_area_px,
        exclude_border=args.watershed_exclude_border,
        ridge_level=args.watershed_ridge_level,
    )


def _print_summary(mean_metrics: dict[str, float] | None, sample_count: int) -> None:
    if sample_count == 1:
        print("\n--- Single-Sample Evaluation ---")
        print(
            "Descriptive only: one evaluation sample found; "
            "skipping aggregate mean metrics."
        )
        return
    if mean_metrics is None:
        return
    print("\n--- Mean Metrics ---")
    for key, value in mean_metrics.items():
        print(f"{key}: {value:.4f}")


def _load_or_predict_semantic(
    *,
    args: argparse.Namespace,
    sample_id: str,
    images: list[np.ndarray],
    model_loader: Any,
) -> np.ndarray:
    height, width = int(images[0].shape[0]), int(images[0].shape[1])
    expected_hw = (height, width)

    if args.save_predictions_dir is not None:
        cache_path = _prediction_tiff_path(args.save_predictions_dir, sample_id)
        meta_path = _prediction_meta_path(args.save_predictions_dir, sample_id)
        if cache_path.is_file():
            if not meta_path.is_file():
                print(
                    "Cached prediction TIFF exists but metadata sidecar is missing "
                    f"({meta_path}); recomputing.",
                    file=sys.stderr,
                )
            else:
                try:
                    with meta_path.open(encoding="utf-8") as meta_file:
                        meta = json.load(meta_file)
                    _validate_prediction_cache(meta, args)
                except (OSError, json.JSONDecodeError, ValueError) as exc:
                    print(
                        f"Invalid or incompatible prediction cache metadata "
                        f"({meta_path}): {exc}; recomputing.",
                        file=sys.stderr,
                    )
                else:
                    print(f"Reusing cached prediction: {cache_path}")
                    t_cache = time.perf_counter()
                    pred = _load_cached_prediction_tiff(cache_path, expected_hw)
                    print(
                        f"  Loaded cached prediction: "
                        f"{time.perf_counter() - t_cache:.2f}s"
                    )
                    return pred

    from unet.inference import predict_full_image

    t_inf = time.perf_counter()
    pred_classes, _ = predict_full_image(
        model=model_loader(),
        inputs=tuple(images),
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    print(f"  Inference: {time.perf_counter() - t_inf:.2f}s")

    if args.save_predictions_dir is not None:
        cache_path = _prediction_tiff_path(args.save_predictions_dir, sample_id)
        meta_path = _prediction_meta_path(args.save_predictions_dir, sample_id)
        tifffile.imwrite(
            cache_path,
            pred_classes.astype(np.uint8),
            compression="deflate",
        )
        with meta_path.open("w", encoding="utf-8") as meta_file:
            json.dump(
                _prediction_cache_record(args), meta_file, indent=2, sort_keys=True
            )
            meta_file.write("\n")

    return pred_classes


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)

    gt_origin = _effective_gt_origin(args)
    gt_gpkg_path = args.gt_gpkg.resolve()
    gt_scene_polygons = load_image_space_polygons(gt_gpkg_path)

    samples = list_samples(
        image_dir=str(args.image_dir),
        mask_dir=str(args.mask_dir) if args.mask_dir is not None else None,
        image_suffixes=args.image_suffixes,
        mask_ext=args.mask_ext,
        mask_stem_suffix=args.mask_stem_suffix,
        num_inputs=args.num_inputs,
    )
    if not samples:
        print(
            "ERROR: No samples matched the given image/mask directories; "
            "nothing to evaluate.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(samples)} samples to evaluate (gt_origin={gt_origin}).")
    if args.save_predictions_dir is not None:
        args.save_predictions_dir.mkdir(parents=True, exist_ok=True)

    model: Any | None = None

    def model_loader() -> Any:
        nonlocal model
        if model is None:
            import tensorflow as tf

            from unet.model import weighted_crossentropy

            print(f"Loading model from {args.model_path}...")
            model = tf.keras.models.load_model(
                str(args.model_path),
                custom_objects={"weighted_crossentropy": weighted_crossentropy},
            )
        return model

    sample_rows: list[dict[str, Any]] = []

    for sample in samples:
        sample_id = sample["id"]
        print(f"Evaluating sample: {sample_id}")
        t0 = time.perf_counter()

        images = [load_rgb_image(p) for p in sample["images"]]
        if len(images) != args.num_inputs:
            raise ValueError("Mismatch between num_inputs and loaded images.")
        if "mask" in sample:
            semantic_mask_after_sample_validation(
                images, load_raster_mask(sample["mask"]), sample["mask"]
            )
            print(
                f"  Loaded inputs + raster mask (validated): "
                f"{time.perf_counter() - t0:.2f}s"
            )
        else:
            print(f"  Loaded inputs: {time.perf_counter() - t0:.2f}s")

        pred_classes = _load_or_predict_semantic(
            args=args,
            sample_id=sample_id,
            images=images,
            model_loader=model_loader,
        )

        height, width = int(images[0].shape[0]), int(images[0].shape[1])
        t_inst = time.perf_counter()
        true_instances = scene_polygons_to_patch_instance_map(
            gt_scene_polygons,
            sample_id=sample_id,
            height=height,
            width=width,
            gt_origin=gt_origin,
        )
        pred_instances = _instances_for_metrics(pred_classes, args)
        print(f"  Instance maps (GT + pred): {time.perf_counter() - t_inst:.2f}s")

        metrics: dict[str, float] = {}
        t_aji = time.perf_counter()
        metrics["aji"] = float(compute_aji(true_instances, pred_instances))
        print(f"  AJI: {time.perf_counter() - t_aji:.2f}s")

        t_prf = time.perf_counter()
        metrics.update(compute_instance_metrics_dict(true_instances, pred_instances))
        print(f"  Instance PR/F1 (IoU sweep): {time.perf_counter() - t_prf:.2f}s")

        gt_n = count_instances(true_instances)
        pred_n = count_instances(pred_instances)
        sample_rows.append(
            build_sample_row(
                sample_id,
                metrics=metrics,
                gt_instances=gt_n,
                pred_instances=pred_n,
                empty_gt=gt_n == 0,
            )
        )

        print(
            f"Metrics for {sample_id}: AJI: {metrics['aji']:.4f}, "
            f"F1@0.5: {metrics['f1_iou50']:.4f}, F1@0.75: {metrics['f1_iou75']:.4f}, "
            f"mF1@0.5:0.95: {metrics['mF1_iou50_95']:.4f}"
        )
        print(f"  Sample total (so far): {time.perf_counter() - t0:.2f}s")

    results = build_instance_eval_report(
        model_type=MODEL_TYPE,
        variant=args.variant,
        unit=args.unit,
        samples=sample_rows,
        extras={
            "ground_truth_instance_source": str(gt_gpkg_path),
            "gt_origin": gt_origin,
            "instance_method": args.instance_method,
            "model_path": str(args.model_path.resolve()),
        },
    )
    _print_summary(results.get("mean"), len(sample_rows))

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    t_json = time.perf_counter()
    with args.output_json.open("w", encoding="utf-8") as out_file:
        out_file.write(
            json.dumps(
                json_safe_for_dump(results),
                indent=4,
                allow_nan=False,
                ensure_ascii=False,
            )
        )
    print(f"Saved metrics to {args.output_json} ({time.perf_counter() - t_json:.2f}s)")


if __name__ == "__main__":
    main()
