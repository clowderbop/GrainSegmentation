"""Stage profile selection caches from scratch to local work root (ADR 0005)."""

from __future__ import annotations

import argparse
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from common.profile_tune_gt_cache import gt_cache_dir
from common.test_inference import YoloInferenceProfileCandidate
from yolo.tiled_proposal_cache import proposal_cache_dir


@dataclass(frozen=True)
class CandidateStageTimings:
    copy_gt_s: float
    copy_proposals_s: float
    per_variant_proposals_s: dict[str, float]


@dataclass(frozen=True)
class StagedDetectorTrainImage:
    image_path: Path
    sample_id: str
    copy_s: float


def resolve_train_whole_image_path(
    *, grainseg_root: Path, variant: str
) -> tuple[Path, str]:
    """Scratch path and sample id for the variant train whole stacked TIFF."""
    from common.manifest_io import build_yolo_whole_manifest, resolve_row_path

    manifest = build_yolo_whole_manifest(
        split="train", variant=variant, grainseg_root=grainseg_root
    )
    row = manifest.samples[0]
    if row.image is None:
        raise ValueError(f"YOLO whole train manifest row has no image for {variant}")
    return resolve_row_path(manifest, row.image), row.sample_id


def stage_detector_train_image(
    *,
    grainseg_root: Path,
    variant: str,
    tmp_dir: Path,
) -> StagedDetectorTrainImage:
    """Copy only the train whole stacked TIFF into a job-local directory."""
    src, sample_id = resolve_train_whole_image_path(
        grainseg_root=grainseg_root, variant=variant
    )
    tmp_dir.mkdir(parents=True, exist_ok=True)
    dst = tmp_dir / src.name
    t0 = time.perf_counter()
    shutil.copy2(src, dst)
    copy_s = time.perf_counter() - t0
    return StagedDetectorTrainImage(image_path=dst, sample_id=sample_id, copy_s=copy_s)


def copy_tree_timed(src: Path, dst: Path) -> float:
    """Copy a directory tree and return elapsed seconds."""
    t0 = time.perf_counter()
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return time.perf_counter() - t0


def _require_dir(path: Path, *, scratch_cache_root: Path, label: str) -> None:
    if path.is_dir():
        return
    raise FileNotFoundError(
        f"Missing profile selection cache under scratch {scratch_cache_root}: {label} "
        f"(expected {path})"
    )


def format_candidate_stage_timings(timings: CandidateStageTimings) -> str:
    """Single-line summary for SLURM logs (staging vs scoring separation)."""
    parts = [
        f"copy_gt_s={timings.copy_gt_s:.1f}",
        f"copy_proposals_s={timings.copy_proposals_s:.1f}",
    ]
    for variant, seconds in sorted(timings.per_variant_proposals_s.items()):
        parts.append(f"copy_proposals_{variant}_s={seconds:.1f}")
    return " ".join(parts)


def stage_candidate_work(
    *,
    scratch_cache_root: Path,
    tmp_work_root: Path,
    candidate: YoloInferenceProfileCandidate,
    variants: tuple[str, ...],
) -> tuple[Path, CandidateStageTimings]:
    """Copy GT cache and per-candidate proposal trees into a local work root."""
    tmp_work_root.mkdir(parents=True, exist_ok=True)

    gt_src = gt_cache_dir(scratch_cache_root)
    _require_dir(
        gt_src,
        scratch_cache_root=scratch_cache_root,
        label="gt_cache/train",
    )
    t0 = time.perf_counter()
    copy_tree_timed(gt_src, gt_cache_dir(tmp_work_root))
    copy_gt_s = time.perf_counter() - t0

    per_variant: dict[str, float] = {}
    t_proposals = time.perf_counter()
    for variant in variants:
        rel = proposal_cache_dir(Path(variant), conf=candidate.conf)
        src = scratch_cache_root / rel
        _require_dir(
            src,
            scratch_cache_root=scratch_cache_root,
            label=f"{variant}/tiled_proposals/c{candidate.conf:g}",
        )
        t_var = time.perf_counter()
        copy_tree_timed(src, tmp_work_root / rel)
        per_variant[variant] = time.perf_counter() - t_var
    copy_proposals_s = time.perf_counter() - t_proposals

    return tmp_work_root, CandidateStageTimings(
        copy_gt_s=copy_gt_s,
        copy_proposals_s=copy_proposals_s,
        per_variant_proposals_s=per_variant,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    cand = sub.add_parser(
        "candidate", help="Stage GT + proposal caches for one grid candidate"
    )
    cand.add_argument("--output-dir", type=Path, required=True)
    cand.add_argument("--tmp-work-root", type=Path, required=True)
    cand.add_argument("--grid-config", type=Path, default=None)
    cand.add_argument("--candidate-id", default=None)
    cand.add_argument("--array-index", type=int, default=None)
    cand.add_argument("--variants", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    from common.profile_tune_paths import profile_tune_cache_root

    from yolo.inference_profile_tune import load_tune_grid
    from yolo.profile_tune_candidate import resolve_candidate
    from yolo.profile_tune_cli import parse_profile_tune_variants

    args = _parse_args(argv)
    if args.command != "candidate":
        raise ValueError(f"Unknown command: {args.command}")

    spec = load_tune_grid(args.grid_config)
    candidate = resolve_candidate(
        spec=spec,
        candidate_id=args.candidate_id,
        array_index=args.array_index,
    )
    variants = parse_profile_tune_variants(args.variants)
    scratch_cache = profile_tune_cache_root(args.output_dir)
    work_root, timings = stage_candidate_work(
        scratch_cache_root=scratch_cache,
        tmp_work_root=args.tmp_work_root,
        candidate=candidate,
        variants=variants,
    )
    print(f"Staged candidate caches → {work_root}", flush=True)
    print(format_candidate_stage_timings(timings), flush=True)


if __name__ == "__main__":
    main()
