import argparse
import json
import os
import sys
import time

import numpy as np
import tensorflow as tf
from PIL import Image

# Add src to sys.path to import from training
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.data import list_samples, _load_rgb_image, _load_raster_mask
from training.model import weighted_crossentropy
from evaluation.inference import predict_full_image
from evaluation.instance_masks import (
    semantic_to_instance_label_map,
    semantic_to_instance_label_map_watershed,
)
from evaluation.metrics import compute_aji, compute_instance_metrics_dict
from evaluation.reporting import (
    build_instance_eval_report,
    build_sample_row,
    count_instances,
    json_safe_for_dump,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained U-Net model on test images."
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to the trained .keras model"
    )
    parser.add_argument(
        "--image-dir", required=True, help="Directory containing test images"
    )
    parser.add_argument(
        "--mask-dir", required=True, help="Directory containing ground truth masks"
    )
    parser.add_argument(
        "--output-json", required=True, help="Path to save evaluation metrics JSON"
    )
    parser.add_argument(
        "--save-predictions-dir",
        help="Optional directory to save predicted mask images",
    )
    parser.add_argument(
        "--num-inputs", type=int, default=7, help="Number of inputs (1, 2, or 7)"
    )
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
        help="How to derive GT and predicted instances for instance metrics (AJI, PR/F1).",
    )
    parser.add_argument(
        "--watershed-min-distance",
        type=int,
        default=1,
        help="peak_local_max min_distance when --instance-method watershed (GT and pred).",
    )
    parser.add_argument(
        "--watershed-boundary-dilate-iter",
        type=int,
        default=0,
        help="Binary dilation iterations on boundary mask for watershed ridge.",
    )
    parser.add_argument(
        "--watershed-connectivity",
        type=int,
        choices=(1, 2),
        default=1,
        help="skimage watershed connectivity (1 or 2) when --instance-method watershed.",
    )
    parser.add_argument(
        "--watershed-min-area-px",
        type=int,
        default=0,
        help="Drop instances smaller than this many pixels (0 disables).",
    )
    parser.add_argument(
        "--watershed-exclude-border",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="peak_local_max exclude_border when --instance-method watershed.",
    )
    parser.add_argument(
        "--watershed-ridge-level",
        type=float,
        default=None,
        help=(
            "Ridge elevation for boundary; omit for automatic (matches tuning JSON ridge_level null)."
        ),
    )
    parser.add_argument(
        "--model-type",
        default="unet",
        help="Tag for metrics.json (shared schema with YOLO patch eval).",
    )
    parser.add_argument(
        "--variant",
        default=None,
        help="Optional variant label recorded in metrics.json.",
    )
    parser.add_argument(
        "--unit",
        default="patch",
        help="Evaluation unit label in metrics.json (e.g. patch).",
    )
    args = parser.parse_args()
    _validate_args(args, parser)
    return args


def _raise_argument_error(message: str, parser: argparse.ArgumentParser | None = None):
    if parser is None:
        raise ValueError(message)
    parser.error(message)


def _validate_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser | None = None
) -> None:
    if args.num_inputs not in {1, 2, 7}:
        _raise_argument_error("num_inputs must be one of: 1, 2, 7", parser)
    if len(args.image_suffixes) < args.num_inputs:
        _raise_argument_error(
            "image_suffixes must provide at least num_inputs suffixes", parser
        )
    if args.patch_size <= 0 or args.stride <= 0:
        _raise_argument_error("patch_size and stride must be > 0", parser)
    if args.stride > args.patch_size:
        _raise_argument_error("stride must be <= patch_size", parser)
    if args.batch_size <= 0:
        _raise_argument_error("batch_size must be > 0", parser)
    if args.watershed_min_distance < 1:
        _raise_argument_error("watershed_min_distance must be >= 1", parser)
    if args.watershed_boundary_dilate_iter < 0:
        _raise_argument_error("watershed_boundary_dilate_iter must be >= 0", parser)
    if args.watershed_min_area_px < 0:
        _raise_argument_error("watershed_min_area_px must be >= 0", parser)
    if args.watershed_ridge_level is not None and not np.isfinite(
        args.watershed_ridge_level
    ):
        _raise_argument_error("watershed_ridge_level must be finite when set", parser)


def _prediction_png_path(save_dir: str, sample_id: str) -> str:
    return os.path.join(save_dir, f"{sample_id}_pred.png")


def _load_cached_prediction_png(path: str, expected_hw: tuple[int, int]) -> np.ndarray:
    """Load a saved class mask PNG; shape must match ``expected_hw`` (H, W)."""
    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            raise ValueError(
                f"Cached prediction must be single-channel (L mode): {path}"
            )
    if arr.ndim != 2:
        raise ValueError(f"Cached prediction must be 2D: {path}")
    if arr.shape != expected_hw:
        raise ValueError(
            f"Cached prediction shape {arr.shape} does not match image shape "
            f"{expected_hw}: {path}"
        )
    return arr.astype(np.int32)


def _validate_sample_data(
    images: list[np.ndarray], mask: np.ndarray, mask_path: str
) -> np.ndarray:
    if not images:
        raise ValueError("Sample must contain at least one input image.")

    expected_shape = images[0].shape
    if len(expected_shape) != 3:
        raise ValueError("All input images must have shape (H, W, C).")
    for img in images[1:]:
        if img.shape != expected_shape:
            raise ValueError("All input images must share the same shape.")

    if mask.ndim != 2:
        raise ValueError(f"Raster mask must be 2D: {mask_path}")

    image_shape = expected_shape[:2]
    if mask.shape != image_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {image_shape} "
            f"for {mask_path}"
        )

    mask_int = mask.astype(np.int32)
    if not np.all(mask == mask_int) or np.any((mask_int < 0) | (mask_int > 2)):
        raise ValueError(f"Mask values must be in [0, 2] for {mask_path}")

    return mask_int


def _instances_for_metrics(
    semantic: np.ndarray, args: argparse.Namespace
) -> np.ndarray:
    """Instance IDs from GT or predicted semantic mask (class 0/1/2) for AJI / PR-F1."""
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
            "Descriptive only: one evaluation sample found; skipping aggregate mean metrics."
        )
        return

    if mean_metrics is None:
        return
    print("\n--- Mean Metrics ---")
    for key, value in mean_metrics.items():
        print(f"{key}: {value:.4f}")


def main():
    args = parse_args()

    samples = list_samples(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        image_suffixes=args.image_suffixes,
        mask_ext=args.mask_ext,
        mask_stem_suffix=args.mask_stem_suffix,
        num_inputs=args.num_inputs,
    )

    if not samples:
        print(
            "ERROR: No samples matched the given image/mask directories; nothing to evaluate.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(samples)} samples to evaluate.")

    if args.save_predictions_dir:
        os.makedirs(args.save_predictions_dir, exist_ok=True)

    model: tf.keras.Model | None = None

    def _ensure_model() -> tf.keras.Model:
        nonlocal model
        if model is None:
            print(f"Loading model from {args.model_path}...")
            model = tf.keras.models.load_model(
                args.model_path,
                custom_objects={"weighted_crossentropy": weighted_crossentropy},
            )
        return model

    sample_rows: list[dict] = []

    for sample in samples:
        sample_id = sample["id"]
        print(f"Evaluating sample: {sample_id}")
        t0 = time.perf_counter()

        # Load images
        images = [_load_rgb_image(p) for p in sample["images"]]
        if len(images) != args.num_inputs:
            raise ValueError("Mismatch between num_inputs and loaded images.")
        true_mask = _validate_sample_data(
            images, _load_raster_mask(sample["mask"]), sample["mask"]
        )
        t_load = time.perf_counter()
        print(f"  Loaded inputs + GT mask: {t_load - t0:.2f}s")

        expected_hw = (int(images[0].shape[0]), int(images[0].shape[1]))
        pred_classes: np.ndarray | None = None

        if args.save_predictions_dir:
            cache_path = _prediction_png_path(args.save_predictions_dir, sample_id)
            if os.path.isfile(cache_path):
                print(f"Reusing cached prediction: {cache_path}")
                t_cache = time.perf_counter()
                pred_classes = _load_cached_prediction_png(cache_path, expected_hw)
                print(
                    f"  Loaded cached prediction: {time.perf_counter() - t_cache:.2f}s"
                )

        if pred_classes is None:
            t_inf = time.perf_counter()
            pred_classes, _ = predict_full_image(
                model=_ensure_model(),
                inputs=tuple(images),
                patch_size=args.patch_size,
                stride=args.stride,
                batch_size=args.batch_size,
            )
            print(f"  Inference: {time.perf_counter() - t_inf:.2f}s")

            if args.save_predictions_dir:
                out_img_path = _prediction_png_path(
                    args.save_predictions_dir, sample_id
                )
                Image.fromarray(pred_classes.astype(np.uint8)).save(out_img_path)

        t_inst = time.perf_counter()
        true_instances = _instances_for_metrics(true_mask, args)
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

        line = (
            f"Metrics for {sample_id}: AJI: {metrics['aji']:.4f}, "
            f"F1@0.5: {metrics['f1_iou50']:.4f}, F1@0.75: {metrics['f1_iou75']:.4f}, "
            f"mF1@0.5:0.95: {metrics['mF1_iou50_95']:.4f}"
        )
        print(line)
        print(f"  Sample total (so far): {time.perf_counter() - t0:.2f}s")

    if not sample_rows:
        print(
            "ERROR: Evaluation produced no metric rows (internal inconsistency).",
            file=sys.stderr,
        )
        sys.exit(1)

    results = build_instance_eval_report(
        model_type=args.model_type,
        variant=args.variant,
        unit=args.unit,
        samples=sample_rows,
    )
    mean_for_print = results.get("mean")
    _print_summary(mean_for_print, len(sample_rows))

    t_json = time.perf_counter()
    with open(args.output_json, "w", encoding="utf-8") as f:
        f.write(
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
