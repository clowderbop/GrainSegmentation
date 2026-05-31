"""CLI: grid coordinator — merge cached proposals, eval, audit (ADR 0005)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.manifest_io import collect_manifest_image_paths
from common.prediction_set import (
    assert_yolo_grains_non_overlapping,
    build_yolo_prediction_set_from_sahi_predictions,
    merge_yolo_proposals_by_score,
    prediction_set_path,
    save_prediction_set,
)
from common.test_inference import YoloInferenceProfileCandidate
from common.variants import all_variant_names, repo_root
from yolo.inference_profile_tune import (
    GridResumeContext,
    TuneGridSpec,
    iter_detector_jobs,
    iter_grid_candidates,
    load_grid_results_csv,
    load_tune_grid,
    run_grid_search,
)
from yolo.profile_tune_dry_run import dry_run_scorer
from yolo.profile_tune_work import (
    default_grainseg_and_run_roots,
    ensure_staged_train_manifest,
    evaluate_variant_predictions,
    weights_path,
)
from yolo.sliced_detection import merge_sliced_object_predictions
from yolo.tiled_proposal_cache import (
    detector_cache_expected_record,
    load_tiled_proposals,
    proposal_cache_dir,
)


def validate_detector_caches(
    *,
    work_root: Path,
    spec: TuneGridSpec,
    variants: tuple[str, ...],
    grainseg_root: Path,
    run_root: Path,
) -> None:
    missing: list[tuple[str, float, float]] = []
    for variant, conf, mask_threshold in iter_detector_jobs(spec, variants):
        weights = weights_path(grainseg_root, variant, run_root)
        cache_dir = proposal_cache_dir(work_root / variant, conf=conf, mask_threshold=mask_threshold)
        expected = detector_cache_expected_record(
            variant=variant,
            weights_path=weights,
            conf=conf,
            mask_threshold=mask_threshold,
            sample_id="train",
        )
        try:
            load_tiled_proposals(cache_dir, expected=expected)
        except (FileNotFoundError, ValueError):
            missing.append((variant, conf, mask_threshold))
    if missing:
        formatted = ", ".join(
            f"{variant}(conf={conf:g}, mask={mask:g})"
            for variant, conf, mask in missing
        )
        raise FileNotFoundError(f"Missing or invalid detector caches: {formatted}")


def score_variant_from_cache(
    *,
    variant: str,
    candidate: YoloInferenceProfileCandidate,
    variant_output_dir: Path,
    grainseg_root: Path,
    run_root: Path,
    work_root: Path,
    repo: Path,
) -> Path:
    weights = weights_path(grainseg_root, variant, run_root)
    cache_dir = proposal_cache_dir(
        work_root / variant, conf=candidate.conf, mask_threshold=candidate.mask_threshold
    )
    expected = detector_cache_expected_record(
        variant=variant,
        weights_path=weights,
        conf=candidate.conf,
        mask_threshold=candidate.mask_threshold,
        sample_id="train",
    )
    proposals, _meta = load_tiled_proposals(cache_dir, expected=expected)
    merged_predictions = merge_sliced_object_predictions(
        proposals,
        postprocess_type=candidate.postprocess_type,
        match_metric=candidate.match_metric,
        match_threshold=candidate.match_threshold,
    )

    staged_manifest = ensure_staged_train_manifest(
        grainseg_root=grainseg_root,
        variant=variant,
        work_root=work_root,
        repo=repo,
    )
    from yolo.predict import load_image_for_yolo

    pairs = collect_manifest_image_paths(staged_manifest)
    image_path, sample_id = pairs[0]
    image = load_image_for_yolo(image_path)
    height, width = int(image.shape[0]), int(image.shape[1])

    variant_output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = prediction_set_path(variant_output_dir, sample_id)
    pred_set = build_yolo_prediction_set_from_sahi_predictions(
        merged_predictions,
        height=height,
        width=width,
        mask_threshold=candidate.mask_threshold,
    )
    merged_set = merge_yolo_proposals_by_score(pred_set)
    assert_yolo_grains_non_overlapping(merged_set)
    save_prediction_set(pred_path, merged_set)

    return evaluate_variant_predictions(
        variant=variant,
        variant_output_dir=variant_output_dir,
        staged_manifest=staged_manifest,
        repo=repo,
    )


def _parse_variants(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return all_variant_names()
    names = tuple(v.strip() for v in raw.split(",") if v.strip())
    if not names:
        raise ValueError("--variants must list at least one registry variant")
    return names


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grainseg-root", type=Path, default=None)
    parser.add_argument("--run-root", type=Path, default=None)
    parser.add_argument("--grid-config", type=Path, default=None)
    parser.add_argument("--variants", default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Skip candidate×variant merge+eval when instance_metrics.json and "
            "resume metadata match; append new candidates to grid/results.csv. "
            "--no-resume re-runs all merge+eval paths (tiled-proposal cache still reused) "
            "and rebuilds results.csv incrementally without deleting it upfront."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Synthetic metrics (tests / plumbing checks).",
    )
    parser.add_argument(
        "--recompute-winner-from-csv",
        action="store_true",
        help="Write grid/winner.json from existing grid/results.csv and exit.",
    )
    parser.add_argument("--work-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    variants = _parse_variants(args.variants)

    if args.recompute_winner_from_csv:
        winner = recompute_winner_from_csv(args.output_dir, variant_names=variants)
        print(json.dumps(winner.to_dict(), indent=2))
        return

    repo = repo_root()
    grainseg_root, run_root = default_grainseg_and_run_roots(
        args.grainseg_root, args.run_root
    )
    work_root = args.work_root or (args.output_dir / "_work")
    spec = load_tune_grid(args.grid_config)
    resume_context = GridResumeContext(
        grainseg_root=grainseg_root,
        run_root=run_root,
        grid_config=args.grid_config,
    )

    if args.dry_run:
        run_grid_search(
            candidates=list(iter_grid_candidates(spec)),
            variants=variants,
            output_dir=args.output_dir,
            score_variant=dry_run_scorer({}),
            resume=args.resume,
            resume_context=resume_context if args.resume else None,
        )
        return

    validate_detector_caches(
        work_root=work_root,
        spec=spec,
        variants=variants,
        grainseg_root=grainseg_root,
        run_root=run_root,
    )

    def score_variant(
        variant: str, candidate: YoloInferenceProfileCandidate, out_dir: Path
    ) -> Path:
        return score_variant_from_cache(
            variant=variant,
            candidate=candidate,
            variant_output_dir=out_dir,
            grainseg_root=grainseg_root,
            run_root=run_root,
            work_root=work_root,
            repo=repo,
        )

    winner, _ = run_grid_search(
        candidates=list(iter_grid_candidates(spec)),
        variants=variants,
        output_dir=args.output_dir,
        score_variant=score_variant,
        resume=args.resume,
        resume_context=resume_context,
    )
    print("\nFinal YOLO inference profile (promote with yolo.promote_inference_profile):")
    print(json.dumps(winner.to_dict(), indent=2))


def recompute_winner_from_csv(
    output_dir: Path, *, variant_names: tuple[str, ...] | None = None
) -> YoloInferenceProfileCandidate:
    """Recompute grid/winner.json from an existing grid/results.csv."""
    from yolo.inference_profile_tune import finalize_grid_winner

    variants = variant_names or all_variant_names()
    results_csv = output_dir / "grid" / "results.csv"
    rows = load_grid_results_csv(results_csv)
    if not rows:
        raise ValueError(f"No rows in {results_csv}")
    return finalize_grid_winner(output_dir / "grid", rows, variant_names=variants)


if __name__ == "__main__":
    main()
