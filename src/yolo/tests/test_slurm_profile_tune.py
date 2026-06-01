"""SLURM wrapper contracts for YOLO profile selection (ADR 0006/0007)."""

from __future__ import annotations

import subprocess

from common.variants import repo_root
from yolo.slurm_profile_tune import (
    PIPELINE_PROFILE_TUNE_DOC_MARKERS,
    PROFILE_TUNE_CANDIDATE_RESOURCES,
    PROFILE_TUNE_DETECTOR_MAX_PARALLEL_DEFAULT,
    PROFILE_TUNE_DETECTOR_RESOURCES,
    PROFILE_TUNE_GT_CACHE_COMMON_CD,
    PROFILE_TUNE_GT_CACHE_MODULE,
    PROFILE_TUNE_GT_CACHE_OUTPUT_REL,
    PROFILE_TUNE_GT_CACHE_RESOURCES,
    PROFILE_TUNE_GT_CACHE_TRAIN_LABELS_GPKG,
    SUBMIT_PROFILE_TUNE_USAGE_MARKERS,
    pipeline_md_path,
    run_profile_tune_candidate_script_path,
    run_profile_tune_detector_script_path,
    run_profile_tune_gt_cache_script_path,
    submit_inference_profile_tune_script_path,
)


def test_profile_tune_candidate_slurm_requests_one_cpu() -> None:
    script = run_profile_tune_candidate_script_path()
    assert script.is_file(), f"Missing SLURM job script: {script}"
    text = script.read_text(encoding="utf-8")
    assert "yolo.profile_tune_candidate" in text
    for key in ("mem", "cpus-per-task", "time"):
        assert key in PROFILE_TUNE_CANDIDATE_RESOURCES
        assert f"#SBATCH --{key}={PROFILE_TUNE_CANDIDATE_RESOURCES[key]}" in text
    assert PROFILE_TUNE_CANDIDATE_RESOURCES["cpus-per-task"] == "1"


def test_profile_tune_detector_slurm_uses_array_task_index() -> None:
    script = run_profile_tune_detector_script_path()
    text = script.read_text(encoding="utf-8")
    assert "yolo.profile_tune_detector" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert "--array-index" in text
    assert f"#SBATCH --output={PROFILE_TUNE_DETECTOR_RESOURCES['output']}" in text


def test_submit_profile_tune_submits_throttled_detector_array() -> None:
    text = submit_inference_profile_tune_script_path().read_text(encoding="utf-8")
    assert "run_profile_tune_detector.sh" in text
    assert '--array=1-${detector_count}%${detector_max_parallel}' in text
    assert "DETECTOR_MAX_PARALLEL" in text
    assert f'="${{DETECTOR_MAX_PARALLEL:-{PROFILE_TUNE_DETECTOR_MAX_PARALLEL_DEFAULT}}}"' in text
    assert "detector_job_ids" not in text


def test_profile_tune_gt_cache_slurm_matches_adr_0006_contract() -> None:
    """ADR 0006: common-only job, train GPKG, shared gt_cache/train/ layout."""
    script = run_profile_tune_gt_cache_script_path()
    text = script.read_text(encoding="utf-8")
    for key in ("mem", "cpus-per-task", "time"):
        assert f"#SBATCH --{key}={PROFILE_TUNE_GT_CACHE_RESOURCES[key]}" in text
    assert PROFILE_TUNE_GT_CACHE_COMMON_CD in text
    assert "uv sync (common only)" in text
    assert PROFILE_TUNE_GT_CACHE_MODULE in text
    assert PROFILE_TUNE_GT_CACHE_TRAIN_LABELS_GPKG in text
    assert PROFILE_TUNE_GT_CACHE_OUTPUT_REL in text
    assert 'cd "$REPO_ROOT/src/yolo"' not in text
    assert "yolo.profile_tune_gt_cache" not in text


def test_submit_profile_tune_usage_documents_adr_salvage() -> None:
    result = subprocess.run(
        ["bash", str(submit_inference_profile_tune_script_path()), "--help"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    usage = result.stderr
    for marker in SUBMIT_PROFILE_TUNE_USAGE_MARKERS:
        assert marker in usage, f"missing submit usage marker: {marker!r}"


def test_pipeline_md_points_at_submit_help_for_salvage() -> None:
    text = pipeline_md_path().read_text(encoding="utf-8")
    for marker in PIPELINE_PROFILE_TUNE_DOC_MARKERS:
        assert marker in text, f"missing pipeline.md marker: {marker!r}"
