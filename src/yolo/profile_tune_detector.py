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
from yolo.inference_profile_tune import (
    TuneGridSpec,
    detector_job_at_index,
    load_tune_grid,
    tune_grid_path,
)
from yolo.profile_tune_cli import parse_profile_tune_variants
from yolo.tiled_proposal_cache import (
    collect_tiled_detector_proposals,
    detector_cache_expected_record,
    load_or_write_tiled_proposals,
    load_tiled_proposals,
    proposal_cache_dir,
    proposal_cache_record,
    recipe_whole_window_fingerprint,
    weights_sha256,
)
from yolo.train import _parse_device


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--mask-threshold", type=float, default=None)
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=None,
        help=f"Search grid YAML (default: {tune_grid_path()}).",
    )
    parser.add_argument(
        "--array-index",
        type=int,
        default=None,
        help="SLURM array task id (1-based) selecting variant/conf/mask_threshold.",
    )
    parser.add_argument(
        "--variants",
        default=None,
        help="Comma-separated registry variants (default: all).",
    )
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


def resolve_detector_job(
    *,
    spec: TuneGridSpec,
    variants: tuple[str, ...],
    variant: str | None,
    conf: float | None,
    mask_threshold: float | None,
    array_index: int | None,
) -> tuple[str, float, float]:
    explicit = (variant, conf, mask_threshold)
    if array_index is not None and any(v is not None for v in explicit):
        raise ValueError(
            "Specify either --array-index or --variant/--conf/--mask-threshold, not both"
        )
    if array_index is not None:
        return detector_job_at_index(spec, variants, array_index)
    if None in explicit:
        raise ValueError(
            "One of --array-index or all of --variant, --conf, --mask-threshold is required"
        )
    return variant, conf, mask_threshold


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
    recipe = load_test_inference_recipe()
    expected = detector_cache_expected_record(
        variant=variant,
        weights_path=weights,
        conf=conf,
        mask_threshold=mask_threshold,
        sample_id=sample_id,
        recipe=recipe,
    )
    try:
        load_tiled_proposals(cache_dir, expected=expected)
        return cache_dir
    except (FileNotFoundError, ValueError):
        pass

    image = load_image_for_yolo(image_path)
    height, width = int(image.shape[0]), int(image.shape[1])
    record = proposal_cache_record(
        variant=variant,
        weights_sha256=weights_sha256(weights),
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=conf,
        mask_threshold=mask_threshold,
        sample_id=sample_id,
        height=height,
        width=width,
    )

    def compute_proposals():
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
        return collect_tiled_detector_proposals(
            image,
            detection_model,
            slice_height=int(window_kwargs["slice_height"]),
            slice_width=int(window_kwargs["slice_width"]),
            overlap_height_ratio=float(window_kwargs["overlap_height_ratio"]),
            overlap_width_ratio=float(window_kwargs["overlap_width_ratio"]),
            mask_threshold=mask_threshold,
        )

    load_or_write_tiled_proposals(
        cache_dir,
        expected=expected,
        meta=record,
        compute_fn=compute_proposals,
    )
    return cache_dir


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    repo = repo_root()
    variants = parse_profile_tune_variants(args.variants)
    spec = load_tune_grid(args.grid_config)
    variant, conf, mask_threshold = resolve_detector_job(
        spec=spec,
        variants=variants,
        variant=args.variant,
        conf=args.conf,
        mask_threshold=args.mask_threshold,
        array_index=args.array_index,
    )
    grainseg_root, run_root = default_grainseg_and_run_roots(
        args.grainseg_root, args.run_root
    )
    work_root = args.work_root or (args.output_dir / "_work")
    if args.array_index is not None:
        print(
            f"Detector array task {args.array_index}: "
            f"variant={variant} conf={conf:g} mask_threshold={mask_threshold:g}"
        )
    cache_dir = write_detector_proposal_cache(
        variant=variant,
        conf=conf,
        mask_threshold=mask_threshold,
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
