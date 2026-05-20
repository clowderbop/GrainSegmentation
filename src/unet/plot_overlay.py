"""Qualitative mask overlays on microscopy RGB images (UNet evaluation outputs)."""

from __future__ import annotations

import argparse
import os

import numpy as np
import tifffile
from PIL import Image

from common.image_io import load_tiff_rgb_hwc_float, load_tiff_single_channel_mask

Image.MAX_IMAGE_PIXELS = None
OVERLAY_MAX_DIM = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write ground-truth and prediction mask overlays on a microscopy image.",
    )
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--gt-path", required=True)
    parser.add_argument("--pred-paths", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output-overlay", required=True)
    args = parser.parse_args()
    if len(args.pred_paths) != len(args.labels):
        parser.error("Number of pred paths must match number of labels.")
    return args


def blend_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color_mask = np.zeros_like(image)
    color_mask[mask > 0] = [1.0, 0.0, 0.0]

    alpha = 0.4
    overlay = np.copy(image)
    active = mask > 0
    overlay[active] = image[active] * (1 - alpha) + color_mask[active] * alpha
    return overlay


def _resize_overlay_arrays(
    rgb_img: np.ndarray,
    gt_mask: np.ndarray,
    preds: list[np.ndarray],
    max_dim: int = OVERLAY_MAX_DIM,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    height, width = rgb_img.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return rgb_img, gt_mask, preds

    scale = max_dim / float(longest)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized_size = (resized_width, resized_height)

    resized_image = (
        np.asarray(
            Image.fromarray((rgb_img * 255.0).astype(np.uint8), mode="RGB").resize(
                resized_size, resample=Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    resized_gt = np.asarray(
        Image.fromarray(gt_mask).resize(resized_size, resample=Image.Resampling.NEAREST)
    )
    resized_preds = [
        np.asarray(
            Image.fromarray(pred).resize(
                resized_size, resample=Image.Resampling.NEAREST
            )
        )
        for pred in preds
    ]
    return resized_image, resized_gt, resized_preds


def _sanitize_overlay_label(label: str) -> str:
    safe_chars = []
    for char in label:
        if char.isalnum() or char in {"+", "-", "_"}:
            safe_chars.append(char)
        else:
            safe_chars.append("_")

    sanitized = "".join(safe_chars).strip("_")
    return sanitized or "model"


def _build_overlay_output_paths(
    output_path: str, labels: list[str]
) -> tuple[str, list[str]]:
    output_dir = os.path.dirname(output_path) or "."
    base_name = os.path.splitext(os.path.basename(output_path))[0] or "overlay"

    gt_output = os.path.join(output_dir, f"{base_name}_ground_truth.tif")
    pred_outputs = [
        os.path.join(output_dir, f"{base_name}_{_sanitize_overlay_label(label)}.tif")
        for label in labels
    ]
    return gt_output, pred_outputs


def generate_qualitative_overlay(
    image_path: str,
    gt_path: str,
    pred_paths: list[str],
    labels: list[str],
    output_path: str,
) -> None:
    rgb_img = load_tiff_rgb_hwc_float(image_path)
    gt_mask = load_tiff_single_channel_mask(gt_path)

    preds = [load_tiff_single_channel_mask(pp) for pp in pred_paths]

    rgb_img, gt_mask, preds = _resize_overlay_arrays(rgb_img, gt_mask, preds)
    gt_output_path, pred_output_paths = _build_overlay_output_paths(output_path, labels)

    if os.path.exists(output_path):
        os.remove(output_path)

    tifffile.imwrite(
        gt_output_path,
        (blend_overlay(rgb_img, gt_mask) * 255.0).astype(np.uint8),
        photometric="rgb",
        compression="deflate",
    )
    print(f"Saved qualitative overlay to {gt_output_path}")

    for label, pred, pred_output_path in zip(labels, preds, pred_output_paths):
        tifffile.imwrite(
            pred_output_path,
            (blend_overlay(rgb_img, pred) * 255.0).astype(np.uint8),
            photometric="rgb",
            compression="deflate",
        )
        print(f"Saved qualitative overlay to {pred_output_path} ({label})")


def main() -> None:
    args = parse_args()
    generate_qualitative_overlay(
        args.image_path,
        args.gt_path,
        args.pred_paths,
        args.labels,
        args.output_overlay,
    )


if __name__ == "__main__":
    main()
