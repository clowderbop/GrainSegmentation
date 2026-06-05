"""SLURM wrapper contracts for YOLO profile selection (ADR 0005)."""

from __future__ import annotations

import subprocess

from common.variants import repo_root
from yolo.inference_profile_tune import load_tune_grid
from yolo.slurm_profile_tune import (
    PROFILE_TUNE_CANDIDATE_RESOURCES,
    PROFILE_TUNE_DETECTOR_MAX_PARALLEL_DEFAULT,
    PROFILE_TUNE_DETECTOR_RESOURCES,
    PROFILE_TUNE_DETECTOR_WALLTIME_LONG,
    PROFILE_TUNE_DETECTOR_WALLTIME_SHORT,
    profile_tune_detector_walltime,
    PROFILE_TUNE_GT_CACHE_COMMON_CD,
    PROFILE_TUNE_GT_CACHE_MODULE,
    PROFILE_TUNE_GT_CACHE_OUTPUT_REL,
    PROFILE_TUNE_GT_CACHE_RESOURCES,
    PROFILE_TUNE_GT_CACHE_TRAIN_LABELS_GPKG,
    PROFILE_TUNE_RUNBOOK_REL,
    PROFILE_TUNE_VENV_PREP_RESOURCES,
    profile_tune_runbook_path,
    run_profile_tune_candidate_script_path,
    run_profile_tune_detector_script_path,
    run_profile_tune_finalize_script_path,
    run_profile_tune_gt_cache_script_path,
    run_profile_tune_venv_prep_script_path,
    submit_inference_profile_tune_script_path,
)


def test_profile_tune_candidate_slurm_requests_one_cpu() -> None:
    script = run_profile_tune_candidate_script_path()
    assert script.is_file(), f"Missing SLURM job script: {script}"
    text = script.read_text(encoding="utf-8")
    assert "yolo.profile_tune_candidate" in text
    assert PROFILE_TUNE_CANDIDATE_RESOURCES["mem"] == "50G"
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


def test_profile_tune_detector_stages_train_image_to_tmpdir() -> None:
    text = run_profile_tune_detector_script_path().read_text(encoding="utf-8")
    assert 'TMP_IMAGE_DIR="$TMPDIR/profile_tune_det_${SLURM_ARRAY_TASK_ID:-0}_${SLURM_JOB_ID:-local}"' in text
    assert '--work-root "$WORK_ROOT"' in text
    assert 'WORK_ROOT="${OUTPUT_DIR}/.cache"' in text
    assert '--train-image-staging-dir "$TMP_IMAGE_DIR"' in text


def test_profile_tune_detector_walltime_tiers_from_detector_key_count(
    tmp_path: Path,
) -> None:
    import yaml

    narrow = tmp_path / "narrow.yaml"
    narrow.write_text(
        yaml.safe_dump({"grid": {"conf": [0.2, 0.3]}}),
        encoding="utf-8",
    )
    wide = tmp_path / "wide.yaml"
    wide.write_text(
        yaml.safe_dump({"grid": {"conf": [0.2, 0.3, 0.4, 0.5]}}),
        encoding="utf-8",
    )
    assert profile_tune_detector_walltime(load_tune_grid(narrow)) == (
        PROFILE_TUNE_DETECTOR_WALLTIME_SHORT
    )
    assert profile_tune_detector_walltime(load_tune_grid(wide)) == (
        PROFILE_TUNE_DETECTOR_WALLTIME_LONG
    )


def test_submit_profile_tune_passes_tiered_detector_walltime() -> None:
    text = submit_inference_profile_tune_script_path().read_text(encoding="utf-8")
    assert "profile_tune_detector_walltime" in text
    assert '"--time=${detector_walltime}"' in text


def test_profile_tune_candidate_stages_shared_venv_without_sync() -> None:
    text = run_profile_tune_candidate_script_path().read_text(encoding="utf-8")
    assert "uv sync" not in text
    assert "yolo_venv_stage_local" in text
    assert "uv run --no-sync" in text


def test_profile_tune_candidate_stages_caches_to_tmpdir_before_scoring() -> None:
    text = run_profile_tune_candidate_script_path().read_text(encoding="utf-8")
    assert 'WORK_ROOT="$TMPDIR/profile_tune_cand_${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${SLURM_ARRAY_TASK_ID}"' in text
    assert "yolo.profile_tune_cache_stage" in text
    assert '--work-root "$WORK_ROOT"' in text
    assert "${OUTPUT_DIR}/.cache" in text


def test_profile_tune_finalize_stages_shared_venv_without_sync() -> None:
    text = run_profile_tune_finalize_script_path().read_text(encoding="utf-8")
    assert "uv sync" not in text
    assert "yolo_venv_stage_local" in text
    assert "uv run --no-sync" in text


def test_profile_tune_venv_prep_syncs_shared_yolo_env() -> None:
    script = run_profile_tune_venv_prep_script_path()
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    for key in ("mem", "cpus-per-task", "time"):
        assert f"#SBATCH --{key}={PROFILE_TUNE_VENV_PREP_RESOURCES[key]}" in text
    assert "yolo_venv_prepare_shared" in text


def test_submit_profile_tune_includes_venv_prep_before_candidates() -> None:
    text = submit_inference_profile_tune_script_path().read_text(encoding="utf-8")
    assert "run_profile_tune_venv_prep.sh" in text
    assert "SHARED_VENV_ROOT" in text
    assert "venv_prep_job_id" in text
    assert 'fin_export="ALL,OUTPUT_DIR=${OUTPUT_DIR},GRID_CONFIG=${GRID_CONFIG},SHARED_VENV_ROOT=${SHARED_VENV_ROOT}"' in text
    assert 'cand_export="ALL,OUTPUT_DIR=${OUTPUT_DIR},GRID_CONFIG=${GRID_CONFIG},SHARED_VENV_ROOT=${SHARED_VENV_ROOT}"' in text


def test_submit_profile_tune_submits_throttled_detector_array() -> None:
    text = submit_inference_profile_tune_script_path().read_text(encoding="utf-8")
    assert "run_profile_tune_detector.sh" in text
    assert '--array=1-${detector_count}%${detector_max_parallel}' in text
    assert "DETECTOR_MAX_PARALLEL" in text
    assert f'="${{DETECTOR_MAX_PARALLEL:-{PROFILE_TUNE_DETECTOR_MAX_PARALLEL_DEFAULT}}}"' in text
    assert "detector_job_ids" not in text


def test_profile_tune_gt_cache_slurm_matches_adr_0006_contract() -> None:
    """ADR 0005: common-only job, train GPKG, shared gt_cache/train/ layout."""
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


def test_submit_profile_tune_usage_points_at_runbook() -> None:
    result = subprocess.run(
        ["bash", str(submit_inference_profile_tune_script_path()), "--help"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    usage = result.stderr
    assert str(PROFILE_TUNE_RUNBOOK_REL) in usage
    assert "profile-selection" in usage
    assert "Salvage" not in usage
    assert "pre-fix" not in usage


def test_submit_profile_tune_dry_run_echoes_staging_summary() -> None:
    text = submit_inference_profile_tune_script_path().read_text(encoding="utf-8")
    assert "DRY-RUN detector array" in text
    assert "input-configuration tasks" in text
    assert "DRY-RUN candidate array" in text
    assert "$TMPDIR" in text
    assert "DRY-RUN finalize" in text
    assert "mean_pq" in text


def test_submit_profile_tune_usage_documents_layout_and_staging() -> None:
    result = subprocess.run(
        ["bash", str(submit_inference_profile_tune_script_path()), "--help"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    usage = result.stderr
    assert ".cache/" in usage
    assert "$TMPDIR" in usage or "TMPDIR" in usage
    assert "input configuration" in usage.lower() or "per variant" in usage.lower()
    assert "grid/" in usage


def test_submit_profile_tune_usage_documents_skip_detectors() -> None:
    result = subprocess.run(
        ["bash", str(submit_inference_profile_tune_script_path()), "--help"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    usage = result.stderr
    assert "--skip-detectors" in usage
    assert "SKIP_DETECTORS" in usage
    assert "valid v2 .cache/" in usage
    assert "old _work/" in usage
    assert "RUN_ID" in usage


def test_profile_tune_runbook_exists() -> None:
    path = profile_tune_runbook_path()
    assert path.is_file(), f"missing runbook: {path}"
    text = path.read_text(encoding="utf-8")
    assert "## Profile selection" in text
    assert "submit_inference_profile_tune.sh" in text


def test_staging_reference_documents_profile_tune_exception() -> None:
    path = repo_root() / "docs/reference/staging.md"
    text = path.read_text(encoding="utf-8")
    assert "## Profile selection" in text
    assert ".cache/" in text
    assert "grid/" in text
    assert "$TMPDIR" in text


def test_profile_tune_adr_documents_cache_staging_and_detector_bundling() -> None:
    adr = repo_root() / "docs/adr/0005-yolo-inference-profile-train-selection.md"
    text = adr.read_text(encoding="utf-8")
    assert ".cache/" in text
    assert "$TMPDIR" in text
    assert "input configuration" in text.lower()
    assert "tiled_proposals" in text
    assert "gt_cache/train" in text
    assert "_work/" in text


def test_profile_tune_runbook_documents_cache_staging_and_detector_array() -> None:
    text = profile_tune_runbook_path().read_text(encoding="utf-8")
    profile_section = text.split("## Profile selection", 1)[1].split("## Test evaluations", 1)[0]
    assert ".cache/" in profile_section
    assert "grid/" in profile_section
    assert "$TMPDIR" in profile_section
    assert "input configuration" in profile_section.lower()
    assert "profile_tune_cache_stage" in profile_section
    assert "gt_cache/train" in profile_section
    assert "mean_pq" in profile_section
    assert "_work/" in profile_section
