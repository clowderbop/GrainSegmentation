"""Write SAHI whole-image prediction overlay TIFFs from instance label maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.instance_predictions import instance_map_filename, read_instance_map_tiff
from common.manifest_io import collect_manifest_image_paths
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
    if args.image is None:
        raise ValueError("Provide --manifest or --image")
    image_path = args.image.resolve()
    return [(image_path, image_path.stem)]


def export_sample_visualization(
    *,
    image_path: Path,
    pred_instances_path: Path,
    sample_out_dir: Path,
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not pred_instances_path.is_file():
        raise FileNotFoundError(f"Prediction instance map not found: {pred_instances_path}")

    image = load_image_for_yolo(image_path)
    pred_map = read_instance_map_tiff(pred_instances_path)
    if pred_map.shape[:2] != image.shape[:2]:
        raise ValueError(
            f"Prediction map shape {pred_map.shape} does not match image {image.shape[:2]}"
        )

    sample_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = sample_out_dir / "prediction_visual.tif"
    write_mask_overlay_visual(image, pred_map, out_path)
    print(f"Wrote {out_path}")


def run_export_sahi_visualization(args: argparse.Namespace) -> None:
    pred_dir = args.pred_dir.resolve()
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    samples = _resolve_manifest_image_samples(args)
    n_samples = len(samples)
    print(
        f"SAHI visualization export: {n_samples} sample(s) -> {out_root} "
        f"(pred_dir={pred_dir})"
    )

    for idx, (image_path, sample_id) in enumerate(samples):
        print(f"Rendering visualization {sample_id} ({idx + 1}/{n_samples})...")
        pred_path = pred_dir / "instances" / instance_map_filename(sample_id)
        if not pred_path.is_file():
            pred_path = pred_dir / "instances" / instance_map_filename(image_path.stem)
        export_sample_visualization(
            image_path=image_path,
            pred_instances_path=pred_path,
            sample_out_dir=out_root / image_path.stem,
        )
    print(f"Wrote {n_samples} visualization(s) under {out_root}")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export SAHI prediction overlay TIFF visualizations.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--pred-dir",
        required=True,
        type=Path,
        help="YOLO predict output root containing instances/{sample_id}_instances.tif",
    )
    parser.add_argument("--manifest", default=None, type=Path)
    parser.add_argument("--image", default=None, type=Path)
    parser.add_argument("--variant", choices=variant_choices(), default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.manifest is None and args.image is None:
        parser.error("Provide --manifest or --image")
    run_export_sahi_visualization(args)


if __name__ == "__main__":
    main()
