"""CLI: write tiled detector proposal cache for one (variant, conf, mask_threshold)."""

from __future__ import annotations

import argparse
from pathlib import Path

from sahi import AutoDetectionModel

from common.manifest_io import collect_manifest_image_paths
from common.test_inference import load_test_inference_recipe
from common.variants import repo_root
from yolo.predict import device_for_sahi, load_image_for_yolo
from yolo.profile_tune_work import (
    default_grainseg_and_run_roots,
    ensure_staged_train_manifest,
    sahi_window_kwargs,
    weights_path,
)
from yolo.sliced_detection import run_whole_sliced_detection
from yolo.tiled_proposal_cache import (
    detector_cache_expected_record,
    load_or_write_tiled_proposals,
    proposal_cache_dir,
)
from yolo.train import _parse_device


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--conf", type=float, required=True)
    parser.add_argument("--mask-threshold", type=float, required=True)
    parser.add_argument("--grainseg-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--work-root",
        type=Path,
        default=None,
        help="Profile tune work directory (default: output-dir/_work).",
    )
    return parser.parse_args(argv)


def write_detector_proposal_cache(
    *,
    variant: str,
    conf: float,
    mask_threshold: float,
    output_dir: Path,
    grainseg_root: Path,
    run_root: Path,
    work_root: Path,
    device: str,
    repo: Path,
) -> Path:
    weights = weights_path(grainseg_root, variant, run_root)
    if not weights.is_file():
        raise FileNotFoundError(f"Missing YOLO weights for {variant}: {weights}")

    staged_manifest = ensure_staged_train_manifest(
        grainseg_root=grainseg_root,
        variant=variant,
        work_root=work_root,
        repo=repo,
    )
    pairs = collect_manifest_image_paths(staged_manifest)
    if len(pairs) != 1:
        raise ValueError(
            f"Profile tune detector expects one train whole sample, got {len(pairs)}"
        )
    image_path, sample_id = pairs[0]
    cache_dir = proposal_cache_dir(work_root / variant, conf=conf, mask_threshold=mask_threshold)
    record = detector_cache_expected_record(
        variant=variant,
        weights_path=weights,
        conf=conf,
        mask_threshold=mask_threshold,
        sample_id=sample_id,
    )

    def compute_proposals() -> list:
        image = load_image_for_yolo(image_path)
        recipe = load_test_inference_recipe()
        sahi_device = device_for_sahi(_parse_device(device))
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(weights.resolve()),
            confidence_threshold=conf,
            mask_threshold=mask_threshold,
            device=sahi_device,
            image_size=recipe.whole.window,
        )
        window_kwargs = sahi_window_kwargs()
        return run_whole_sliced_detection(
            image,
            detection_model,
            slice_height=int(window_kwargs["slice_height"]),
            slice_width=int(window_kwargs["slice_width"]),
            overlap_height_ratio=float(window_kwargs["overlap_height_ratio"]),
            overlap_width_ratio=float(window_kwargs["overlap_width_ratio"]),
        )

    load_or_write_tiled_proposals(cache_dir, expected=record, compute_fn=compute_proposals)
    return cache_dir


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    repo = repo_root()
    grainseg_root, run_root = default_grainseg_and_run_roots(
        args.grainseg_root, args.run_root
    )
    work_root = args.work_root or (args.output_dir / "_work")
    cache_dir = write_detector_proposal_cache(
        variant=args.variant,
        conf=args.conf,
        mask_threshold=args.mask_threshold,
        output_dir=args.output_dir,
        grainseg_root=grainseg_root,
        run_root=run_root,
        work_root=work_root,
        device=args.device,
        repo=repo,
    )
    print(f"Tiled detector proposals ready → {cache_dir}")


if __name__ == "__main__":
    main()
