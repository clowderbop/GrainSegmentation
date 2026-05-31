"""Shared CLI helpers for YOLO profile tune entrypoints."""

from __future__ import annotations

from pathlib import Path

from common.variants import all_variant_names
from yolo.inference_profile_tune import TuneGridSpec, iter_detector_jobs
from yolo.profile_tune_work import weights_path
from yolo.tiled_proposal_cache import (
    detector_cache_expected_record,
    load_tiled_proposals,
    proposal_cache_dir,
)


def parse_profile_tune_variants(raw: str | None) -> tuple[str, ...]:
    if raw is None:
        return all_variant_names()
    names = tuple(v.strip() for v in raw.split(",") if v.strip())
    if not names:
        raise ValueError("--variants must list at least one registry variant")
    return names


def validate_detector_caches(
    *,
    work_root: Path,
    spec: TuneGridSpec,
    variants: tuple[str, ...],
    grainseg_root: Path,
    run_root: Path,
) -> None:
    """Ensure all tiled-proposal caches exist for the full detector job grid."""
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
