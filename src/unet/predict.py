"""Run U-Net sliding-window inference and write semantic prediction TIFFs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from common.arg_errors import raise_cli_argument_error
from common.manifest_io import collect_manifest_unet_samples, load_dataset_manifest
from common.samples import load_rgb_image, load_raster_mask
from unet.prediction_cache import (
    load_or_use_cached_prediction,
    prediction_tiff_path,
)
from unet.sample_checks import semantic_mask_after_sample_validation


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run U-Net inference and write semantic prediction TIFFs.",
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Dataset manifest listing per-sample image paths.",
    )
    parser.add_argument("--mask-dir", default=None, type=Path)
    parser.add_argument("--num-inputs", type=int, default=None)
    parser.add_argument("--mask-ext", default=None)
    parser.add_argument("--mask-stem-suffix", default="")
    parser.add_argument("--patch-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--unit",
        choices=("patch", "whole"),
        default="patch",
        help="Recorded in prediction metadata only.",
    )
    parser.add_argument("--variant", default=None)
    return parser


def _resolve_predict_samples(args: argparse.Namespace) -> list[dict[str, Any]]:
    doc = load_dataset_manifest(args.manifest)
    if args.variant is not None and args.variant != doc.variant:
        raise_cli_argument_error(
            f"--variant {args.variant!r} does not match manifest variant "
            f"{doc.variant!r}"
        )
    first_images = doc.samples[0].images
    if first_images is None:
        raise_cli_argument_error("U-Net predict requires manifest rows with images")
    num_inputs = len(first_images)
    if args.num_inputs is not None and args.num_inputs != num_inputs:
        raise_cli_argument_error(
            f"--num-inputs {args.num_inputs} does not match manifest ({num_inputs})"
        )
    args.num_inputs = num_inputs
    args.image_suffixes = []  # not used for manifest-driven cache keys
    return collect_manifest_unet_samples(
        doc,
        mask_dir=args.mask_dir,
        mask_ext=args.mask_ext,
        mask_stem_suffix=args.mask_stem_suffix,
    )


def _validate_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser | None = None
) -> None:
    if args.num_inputs is not None and args.num_inputs not in {1, 2, 7}:
        raise_cli_argument_error("num_inputs must be one of: 1, 2, 7", parser=parser)
    if args.patch_size <= 0 or args.stride <= 0:
        raise_cli_argument_error("patch_size and stride must be > 0", parser=parser)
    if args.stride > args.patch_size:
        raise_cli_argument_error("stride must be <= patch_size", parser=parser)
    if args.batch_size <= 0:
        raise_cli_argument_error("batch_size must be > 0", parser=parser)
    if args.mask_dir is not None and not args.mask_dir.is_dir():
        raise_cli_argument_error(
            f"mask-dir is not a directory: {args.mask_dir}", parser=parser
        )


def run_predict(args: argparse.Namespace) -> None:
    samples = _resolve_predict_samples(args)
    _validate_args(args)
    if args.num_inputs is None:
        raise_cli_argument_error("Could not determine num_inputs")

    semantic_dir = args.output_dir / "semantic"
    semantic_dir.mkdir(parents=True, exist_ok=True)

    if not samples:
        print(
            "ERROR: No samples matched the given manifest or image directories.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Found {len(samples)} samples for inference (unit={args.unit}).")
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

    for sample in samples:
        sample_id = sample["id"]
        print(f"Predicting sample: {sample_id}")
        t0 = time.perf_counter()
        images = [load_rgb_image(p) for p in sample["images"]]
        if len(images) != args.num_inputs:
            raise ValueError("Mismatch between num_inputs and loaded images.")
        if "mask" in sample:
            semantic_mask_after_sample_validation(
                images, load_raster_mask(sample["mask"]), sample["mask"]
            )
        height, width = int(images[0].shape[0]), int(images[0].shape[1])
        expected_hw = (height, width)

        def predict_fn() -> np.ndarray:
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
            return pred_classes

        load_or_use_cached_prediction(
            args=args,
            sample_id=sample_id,
            semantic_dir=semantic_dir,
            expected_hw=expected_hw,
            predict_fn=predict_fn,
        )
        out_path = prediction_tiff_path(semantic_dir, sample_id)
        print(f"  Wrote {out_path} ({time.perf_counter() - t0:.2f}s)")


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    run_predict(args)


if __name__ == "__main__":
    main()
