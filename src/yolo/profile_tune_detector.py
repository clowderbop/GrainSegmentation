"""CLI: write tiled detector proposal caches for profile selection (ADR 0005)."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

from common.profile_tune_paths import profile_tune_cache_root
from common.test_inference import load_test_inference_recipe, profile_tune_fixed_mask_threshold
from common.variants import repo_root
from yolo.inference_profile_tune import (
    TuneGridSpec,
    detector_keys_per_variant,
    iter_detector_keys,
    load_tune_grid,
    tune_grid_path,
    variant_at_detector_array_index,
)
from yolo.profile_tune_cli import parse_profile_tune_variants
from yolo.profile_tune_detector_cache import (
    format_scratch_cache_label,
    prepare_detector_variant,
    write_detector_key_proposals_if_needed,
)
from yolo.profile_tune_work import default_grainseg_and_run_roots


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--conf", type=float, default=None)
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
        help="SLURM array task id (1-based) selecting an input configuration (variant).",
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
        help="Profile tune durable cache root (default: output-dir/.cache).",
    )
    parser.add_argument(
        "--local-train-image",
        type=Path,
        default=None,
        help="Pre-staged train whole stacked TIFF (e.g. copied to $TMPDIR by SLURM).",
    )
    parser.add_argument(
        "--train-image-staging-dir",
        type=Path,
        default=None,
        help="Copy the train whole stacked TIFF into this directory before inference.",
    )
    return parser.parse_args(argv)


def resolve_detector_variant(
    *,
    variants: tuple[str, ...],
    variant: str | None,
    conf: float | None,
    array_index: int | None,
) -> str:
    explicit_variant = variant
    if array_index is not None and (explicit_variant is not None or conf is not None):
        raise ValueError(
            "Specify either --array-index or --variant/--conf, not both"
        )
    if array_index is not None:
        return variant_at_detector_array_index(variants, array_index)
    if explicit_variant is None:
        raise ValueError(
            "One of --array-index or --variant is required for profile tune detector"
        )
    return explicit_variant


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
    local_train_image: Path | None = None,
    train_image_staging_dir: Path | None = None,
) -> Path:
    del output_dir, repo  # retained for CLI compatibility
    prepared, staging_note = prepare_detector_variant(
        variant=variant,
        grainseg_root=grainseg_root,
        run_root=run_root,
        device=device,
        local_train_image=local_train_image,
        train_image_staging_dir=train_image_staging_dir,
    )
    if staging_note:
        print(staging_note, flush=True)
    cache_dir, _model, _wrote = write_detector_key_proposals_if_needed(
        prepared,
        variant=variant,
        conf=conf,
        mask_threshold=mask_threshold,
        work_root=work_root,
        log_skip=False,
    )
    return cache_dir


def run_detector_variant_bundle(
    *,
    variant: str,
    spec: TuneGridSpec,
    output_dir: Path,
    grainseg_root: Path,
    run_root: Path,
    work_root: Path,
    device: str,
    repo: Path,
    local_train_image: Path | None = None,
    train_image_staging_dir: Path | None = None,
    detector_keys: Iterable[float] | None = None,
) -> None:
    """Stage train image once, load YOLO once, write all conf-key caches for a variant."""
    del repo
    scratch_cache = profile_tune_cache_root(output_dir)
    prepared, staging_note = prepare_detector_variant(
        variant=variant,
        grainseg_root=grainseg_root,
        run_root=run_root,
        device=device,
        local_train_image=local_train_image,
        train_image_staging_dir=train_image_staging_dir,
    )
    if staging_note:
        print(staging_note, flush=True)

    fixed_mask = profile_tune_fixed_mask_threshold(prepared.recipe)
    detection_model = None
    keys = (
        list(detector_keys)
        if detector_keys is not None
        else list(iter_detector_keys(spec))
    )

    for conf in keys:
        try:
            cache_dir, detection_model, wrote = write_detector_key_proposals_if_needed(
                prepared,
                variant=variant,
                conf=conf,
                mask_threshold=fixed_mask,
                work_root=work_root,
                detection_model=detection_model,
                log_skip=True,
            )
        except Exception:
            print(
                f"Detector key failed: variant={variant} conf={conf:g} "
                f"mask_threshold={fixed_mask:g}",
                flush=True,
            )
            raise

        if wrote:
            label = format_scratch_cache_label(cache_dir, scratch_cache)
            print(
                f"Tiled detector proposals written to {label} → {cache_dir}",
                flush=True,
            )


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    variants = parse_profile_tune_variants(args.variants)
    spec = load_tune_grid(args.grid_config)
    variant = resolve_detector_variant(
        variants=variants,
        variant=args.variant,
        conf=args.conf,
        array_index=args.array_index,
    )
    grainseg_root, run_root = default_grainseg_and_run_roots(
        args.grainseg_root, args.run_root
    )
    work_root = args.work_root or profile_tune_cache_root(args.output_dir)
    scratch_cache = profile_tune_cache_root(args.output_dir)

    if args.array_index is not None:
        print(
            f"Detector array task {args.array_index}: variant={variant} "
            f"({detector_keys_per_variant(spec)} detector keys)",
            flush=True,
        )

    if args.conf is not None:
        recipe = load_test_inference_recipe()
        fixed_mask = profile_tune_fixed_mask_threshold(recipe)
        cache_dir = write_detector_proposal_cache(
            variant=variant,
            conf=args.conf,
            mask_threshold=fixed_mask,
            output_dir=args.output_dir,
            grainseg_root=grainseg_root,
            run_root=run_root,
            work_root=work_root,
            device=args.device,
            repo=repo_root(),
            local_train_image=args.local_train_image,
            train_image_staging_dir=args.train_image_staging_dir,
        )
        label = format_scratch_cache_label(cache_dir, scratch_cache)
        print(f"Tiled detector proposals written to {label} → {cache_dir}")
        return

    run_detector_variant_bundle(
        variant=variant,
        spec=spec,
        output_dir=args.output_dir,
        grainseg_root=grainseg_root,
        run_root=run_root,
        work_root=work_root,
        device=args.device,
        repo=repo_root(),
        local_train_image=args.local_train_image,
        train_image_staging_dir=args.train_image_staging_dir,
    )


if __name__ == "__main__":
    main()
