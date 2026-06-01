"""SLURM job helpers for YOLO profile selection (testable resource contract)."""

from __future__ import annotations

from pathlib import Path

from common.variants import repo_root

# ADR 0007: detector jobs run as a throttled SLURM array (default max 6 concurrent).
PROFILE_TUNE_DETECTOR_MAX_PARALLEL_DEFAULT = "6"
PROFILE_TUNE_DETECTOR_RESOURCES: dict[str, str] = {
    "job-name": "yolo_prof_det",
    "output": "logs/yolo_prof_det-%a-%j.log",
    "mem": "32G",
    "cpus-per-task": "8",
    "time": "00:10:00",
}

# ADR 0007: candidate scoring is single-threaded; 50G interim after crop-local adapter fix.
PROFILE_TUNE_CANDIDATE_RESOURCES: dict[str, str] = {
    "job-name": "yolo_prof_cand",
    "output": "logs/yolo_prof_cand-%a-%j.log",
    "mem": "50G",
    "cpus-per-task": "1",
    "time": "04:00:00",
}

PROFILE_TUNE_VENV_PREP_RESOURCES: dict[str, str] = {
    "job-name": "yolo_prof_venv",
    "output": "logs/yolo_prof_venv-%j.log",
    "mem": "32G",
    "cpus-per-task": "8",
    "time": "00:30:00",
}

# ADR 0006: OpenCV GT rasterization; common-only sync (8 CPUs for rasterize — not candidate).
PROFILE_TUNE_GT_CACHE_RESOURCES: dict[str, str] = {
    "job-name": "yolo_prof_gt",
    "output": "logs/yolo_prof_gt-%j.log",
    "mem": "32G",
    "cpus-per-task": "8",
    "time": "04:00:00",
}

PROFILE_TUNE_GT_CACHE_COMMON_CD = 'cd "$REPO_ROOT/src/common"'
PROFILE_TUNE_GT_CACHE_MODULE = "common.profile_tune_gt_cache"
PROFILE_TUNE_GT_CACHE_TRAIN_LABELS_GPKG = "dataset/train/train_labels.gpkg"
PROFILE_TUNE_GT_CACHE_OUTPUT_REL = "_work/gt_cache/train/"

# Canonical salvage text lives in submit --help; pipeline.md points there (ADR 0006/0007).
SUBMIT_PROFILE_TUNE_USAGE_MARKERS: tuple[str, ...] = (
    "docs/adr/0006-gpkg-ground-truth-rasterization.md",
    "docs/adr/0007-profile-selection-proposal-cache-and-scoring.md",
    "Delete the entire runs/yolo_inference_profile_tune",
    "Submit a new RUN_ID",
    "Do not pass --skip-detectors to reuse _work/ from a pre-fix",
    "schema_version 2",
    "--skip-detectors is only for re-submitting",
    "DETECTOR_MAX_PARALLEL",
    "50G",
    "run_profile_tune_venv_prep.sh",
    "SHARED_VENV_ROOT",
)

PIPELINE_PROFILE_TUNE_DOC_MARKERS: tuple[str, ...] = (
    "submit_inference_profile_tune.sh --help",
    "docs/adr/0006-gpkg-ground-truth-rasterization.md",
    "docs/adr/0007-profile-selection-proposal-cache-and-scoring.md",
    "`schema_version` 2",
    "_work/gt_cache/train/",
    "1 CPU",
    "50G",
    "DETECTOR_MAX_PARALLEL",
    "run_profile_tune_venv_prep.sh",
)


def run_profile_tune_candidate_script_path() -> Path:
    return repo_root() / "SLURM" / "yolo" / "run_profile_tune_candidate.sh"


def run_profile_tune_detector_script_path() -> Path:
    return repo_root() / "SLURM" / "yolo" / "run_profile_tune_detector.sh"


def run_profile_tune_gt_cache_script_path() -> Path:
    return repo_root() / "SLURM" / "yolo" / "run_profile_tune_gt_cache.sh"


def run_profile_tune_venv_prep_script_path() -> Path:
    return repo_root() / "SLURM" / "yolo" / "run_profile_tune_venv_prep.sh"


def run_profile_tune_finalize_script_path() -> Path:
    return repo_root() / "SLURM" / "yolo" / "run_profile_tune_finalize.sh"


def submit_inference_profile_tune_script_path() -> Path:
    return repo_root() / "SLURM" / "yolo" / "submit_inference_profile_tune.sh"


def pipeline_md_path() -> Path:
    return repo_root() / "SLURM" / "yolo" / "pipeline.md"
