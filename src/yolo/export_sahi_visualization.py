"""Write SAHI whole-image prediction overlay TIFFs from prediction labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.manifest_io import collect_manifest_image_paths
from common.yolo_seg_labels import yolo_seg_labels_to_instance_map
from yolo.config import variant_choices
from yolo.predict import load_image_for_yolo


def _visualization_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 1:
        return image[:, :, 0]
    return image[:, :, :3]


def _as_rgb_uint8(image: np.ndarray) -> np.ndarray:
    display = _visualization_image(image)
    if display.ndim == 2:
        display = np.repeat(display[:, :, None], 3, axis=2)
    elif display.shape[2] == 1:
        display = np.repeat(display, 3, axis=2)
    return np.clip(display, 0, 255).astype(np.uint8, copy=False)


def write_mask_overlay_visual(
    image: np.ndarray, pred_map: np.ndarray, out_path: Path
) -> None:
    import tifffile

    visual = _as_rgb_uint8(image).astype(np.float32)
    labels = np.unique(pred_map)
    labels = labels[labels > 0]
    for label in labels:
        mask = pred_map == label
        color = np.array(
            [
                (37 * int(label)) % 255,
                (97 * int(label)) % 255,
                (173 * int(label)) % 255,
            ],
            dtype=np.float32,
        )
        visual[mask] = (0.45 * visual[mask]) + (0.55 * color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        out_path,
        np.clip(visual, 0, 255).astype(np.uint8),
        photometric="rgb",
        compression="deflate",
    )


def _resolve_manifest_image_samples(
    args: argparse.Namespace,
) -> list[tuple[Path, str]]:
    if args.manifest is not None:
        return collect_manifest_image_paths(args.manifest)
    if args.test_tiff is None:
        raise ValueError("Provide --manifest or --test-tiff")
    tiff = args.test_tiff.resolve()
    return [(tiff, tiff.stem)]


def export_sample_visualization(
    *,
    image_path: Path,
    pred_label_path: Path,
    sample_out_dir: Path,
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not pred_label_path.is_file():
        raise FileNotFoundError(f"Prediction labels not found: {pred_label_path}")

    image = load_image_for_yolo(image_path)
    height, width = image.shape[:2]
    pred_map = yolo_seg_labels_to_instance_map(
        pred_label_path,
        image_width=width,
        image_height=height,
        has_confidence=True,
    )

    sample_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = sample_out_dir / "prediction_visual.tif"
    write_mask_overlay_visual(image, pred_map, out_path)
    print(f"Wrote {out_path}")


def run_export_sahi_visualization(args: argparse.Namespace) -> None:
    pred_labels_dir = args.pred_labels_dir.resolve()
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    for image_path, sample_id in _resolve_manifest_image_samples(args):
        pred_label_path = pred_labels_dir / f"{sample_id}.txt"
        if not pred_label_path.is_file():
            pred_label_path = pred_labels_dir / f"{image_path.stem}.txt"
        export_sample_visualization(
            image_path=image_path,
            pred_label_path=pred_label_path,
            sample_out_dir=out_root / image_path.stem,
        )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export SAHI prediction overlay TIFF visualizations.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pred-labels-dir", required=True, type=Path)
    parser.add_argument("--manifest", default=None, type=Path)
    parser.add_argument("--test-tiff", default=None, type=Path)
    parser.add_argument("--variant", choices=variant_choices(), default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.manifest is None and args.test_tiff is None:
        parser.error("Provide --manifest or --test-tiff")
    run_export_sahi_visualization(args)


if __name__ == "__main__":
    main()
