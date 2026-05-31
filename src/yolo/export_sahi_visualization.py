"""Write SAHI whole-image prediction overlay TIFFs from instance prediction sets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common.manifest_io import collect_manifest_image_paths, load_dataset_manifest, resolve_row_path
from common.prediction_set import (
    PredictionSet,
    load_prediction_set,
    prediction_set_path,
    segmentation_to_binary_mask,
)
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


def _overlay_color_for_index(index: int) -> np.ndarray:
    return np.array(
        [
            (37 * index) % 255,
            (97 * index) % 255,
            (173 * index) % 255,
        ],
        dtype=np.float32,
    )


def write_mask_overlay_visual(
    image: np.ndarray, pred_map: np.ndarray, out_path: Path
) -> None:
    import tifffile

    visual = _as_rgb_uint8(image).astype(np.float32)
    labels = np.unique(pred_map)
    labels = labels[labels > 0]
    for label in labels:
        mask = pred_map == label
        color = _overlay_color_for_index(int(label))
        visual[mask] = (0.45 * visual[mask]) + (0.55 * color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        out_path,
        np.clip(visual, 0, 255).astype(np.uint8),
        photometric="rgb",
        compression="deflate",
    )


def write_prediction_set_overlay_visual(
    image: np.ndarray, prediction_set: PredictionSet, out_path: Path
) -> None:
    import tifffile

    visual = _as_rgb_uint8(image).astype(np.float32)
    detections = list(prediction_set.detections)
    if prediction_set.producer == "yolo":
        detections.sort(key=lambda det: float(det["score"]), reverse=False)
    for index, det in enumerate(detections):
        mask = segmentation_to_binary_mask(det["segmentation"])
        if not mask.any():
            continue
        color = _overlay_color_for_index(index + 1)
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


def _resolve_prediction_set_path(
    *,
    pred_dir: Path,
    manifest_path: Path | None,
    sample_id: str,
) -> Path:
    if manifest_path is not None:
        doc = load_dataset_manifest(manifest_path)
        for row in doc.samples:
            if row.sample_id != sample_id:
                continue
            if row.instance_prediction_set:
                resolved = resolve_row_path(doc, row.instance_prediction_set)
                assert resolved is not None
                return resolved
    return prediction_set_path(pred_dir, sample_id)


def export_sample_visualization_from_prediction_set(
    *,
    image_path: Path,
    prediction_set_path: Path,
    sample_out_dir: Path,
) -> None:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not prediction_set_path.is_file():
        raise FileNotFoundError(f"Prediction set not found: {prediction_set_path}")

    image = load_image_for_yolo(image_path)
    prediction_set = load_prediction_set(prediction_set_path)
    if (prediction_set.height, prediction_set.width) != image.shape[:2]:
        raise ValueError(
            f"Prediction set size {(prediction_set.height, prediction_set.width)} "
            f"does not match image {image.shape[:2]}"
        )

    sample_out_dir.mkdir(parents=True, exist_ok=True)
    out_path = sample_out_dir / "prediction_visual.tif"
    write_prediction_set_overlay_visual(image, prediction_set, out_path)
    print(f"Wrote {out_path}")


def run_export_sahi_visualization(args: argparse.Namespace) -> None:
    pred_dir = args.pred_dir.resolve()
    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest.resolve() if args.manifest is not None else None

    samples = _resolve_manifest_image_samples(args)
    n_samples = len(samples)
    print(
        f"SAHI visualization export: {n_samples} sample(s) -> {out_root} "
        f"(pred_dir={pred_dir})"
    )

    for idx, (image_path, sample_id) in enumerate(samples):
        print(f"Rendering visualization {sample_id} ({idx + 1}/{n_samples})...")
        ps_path = _resolve_prediction_set_path(
            pred_dir=pred_dir,
            manifest_path=manifest_path,
            sample_id=sample_id,
        )
        export_sample_visualization_from_prediction_set(
            image_path=image_path,
            prediction_set_path=ps_path,
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
        help="YOLO predict output root containing prediction_sets/{sample_id}.json",
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
