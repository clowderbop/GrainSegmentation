"""SLURM wrapper contracts for U-Net watershed predict-then-tune workflow."""

from __future__ import annotations

import subprocess
from pathlib import Path

from common.variants import repo_root
from unet.slurm_watershed_tune import (
    WATERSHED_SEMANTIC_PREDS_DIR_NAME,
    WATERSHED_TUNE_PREDICT_RESOURCES,
    WATERSHED_TUNE_PREDS_ROOT_REL,
    WATERSHED_TUNE_RUNBOOK_REL,
    WATERSHED_TUNE_TUNE_RESOURCES,
    run_watershed_tune_predict_script_path,
    run_watershed_tuning_script_path,
    submit_watershed_tuning_script_path,
    watershed_tune_preds_semantic_dir,
    watershed_tune_runbook_path,
)
from unet.watershed_tune_grid import WATERSHED_TUNE_GRID_CONFIG_REL


def test_watershed_tune_preds_semantic_dir_is_per_variant_scratch_layout() -> None:
    root = Path("/scratch/example/GrainSeg")
    assert watershed_tune_preds_semantic_dir(root, "PPL_AllPPX") == (
        root / "runs/watershed_tune_preds/PPL_AllPPX/semantic"
    )


def test_run_watershed_tune_predict_script_writes_semantic_predictions() -> None:
    script = run_watershed_tune_predict_script_path()
    assert script.is_file(), f"Missing SLURM job script: {script}"
    text = script.read_text(encoding="utf-8")
    assert "unet.predict" in text
    assert "--unit whole" in text
    assert WATERSHED_TUNE_PREDS_ROOT_REL in text
    assert 'VARIANT_PREDS_DIR="$PREDS_ROOT/$WATERSHED_SUBDIR"' in text
    assert f'$VARIANT_PREDS_DIR/{WATERSHED_SEMANTIC_PREDS_DIR_NAME}/' in text
    for key in ("mem", "cpus-per-task", "gpus-per-node", "time"):
        assert key in WATERSHED_TUNE_PREDICT_RESOURCES
        assert f"#SBATCH --{key}={WATERSHED_TUNE_PREDICT_RESOURCES[key]}" in text


def test_run_watershed_tune_predict_uses_test_inference_recipe_geometry() -> None:
    script = run_watershed_tune_predict_script_path()
    text = script.read_text(encoding="utf-8")
    assert "test_inference.sh" in text
    assert "load_test_inference_exports" in text
    assert 'PATCH_SIZE="$UNET_WHOLE_PATCH_SIZE"' in text
    assert 'STRIDE="$UNET_WHOLE_STRIDE"' in text
    assert "PATCH_SIZE=1024" not in text
    assert "STRIDE=512" not in text


def test_shell_scripts_share_preds_root_with_python_helper() -> None:
    for script in (
        run_watershed_tune_predict_script_path(),
        run_watershed_tuning_script_path(),
        submit_watershed_tuning_script_path(),
    ):
        assert WATERSHED_TUNE_PREDS_ROOT_REL in script.read_text(encoding="utf-8")

    tune_text = run_watershed_tuning_script_path().read_text(encoding="utf-8")
    assert (
        f'runs/watershed_tune_preds/$WATERSHED_SUBDIR/{WATERSHED_SEMANTIC_PREDS_DIR_NAME}'
        in tune_text
    )
    example = watershed_tune_preds_semantic_dir(
        Path("/scratch/example/GrainSeg"), "PPL_AllPPX"
    )
    assert str(example).endswith(
        f"{WATERSHED_TUNE_PREDS_ROOT_REL}/PPL_AllPPX/{WATERSHED_SEMANTIC_PREDS_DIR_NAME}"
    )


def test_run_watershed_tuning_stages_manifest_metadata_without_channel_copies() -> (
    None
):
    tune_text = run_watershed_tuning_script_path().read_text(encoding="utf-8")
    assert "stage_manifest_metadata_in_unet_env" in tune_text
    assert "stage_manifest_run_in_unet_env" not in tune_text
    assert "no channel copies" in tune_text

    manifest_shell = (
        repo_root() / "SLURM" / "utils" / "manifest_shell.sh"
    ).read_text(encoding="utf-8")
    assert "stage_manifest_metadata_in_unet_env" in manifest_shell
    assert "--metadata-only" in manifest_shell

    predict_text = run_watershed_tune_predict_script_path().read_text(
        encoding="utf-8"
    )
    assert "stage_manifest_run_in_unet_env" in predict_text


def test_run_watershed_tuning_passes_log_extraction_cache_when_env_set() -> None:
    text = run_watershed_tuning_script_path().read_text(encoding="utf-8")
    assert "LOG_EXTRACTION_CACHE" in text
    assert "--log-extraction-cache" in text


def test_run_watershed_tuning_requires_preds_dir_not_model_path() -> None:
    script = run_watershed_tuning_script_path()
    text = script.read_text(encoding="utf-8")
    assert "--preds-dir" in text
    assert "--model-path" not in text
    assert "unet.tune_watershed" in text
    assert "unet.predict" not in text
    assert "gpus-per-node" not in text
    assert "tensorflow.sh" not in text
    assert "install_unet_tensorflow_wheel" not in text
    assert "--patch-size" not in text
    for key in ("mem", "cpus-per-task", "time"):
        assert key in WATERSHED_TUNE_TUNE_RESOURCES
        assert f"#SBATCH --{key}={WATERSHED_TUNE_TUNE_RESOURCES[key]}" in text


def test_submit_watershed_tuning_submits_predict_then_tune_with_dependency() -> None:
    text = submit_watershed_tuning_script_path().read_text(encoding="utf-8")
    assert "run_watershed_tune_predict.sh" in text
    assert "run_watershed_tuning.sh" in text
    assert "predict_job_id" in text
    assert "--dependency=afterok:${predict_job_id}" in text
    assert "watershed_tune_preds" in text
    assert "PREDS_DIR" in text


def test_submit_watershed_tuning_supports_tune_only_from_cached_preds() -> None:
    text = submit_watershed_tuning_script_path().read_text(encoding="utf-8")
    assert "--use-cached-preds" in text
    assert "USE_CACHED_PREDS" in text
    assert "require_cached_preds_dir" in text
    assert "*_pred.tif" in text
    result = subprocess.run(
        [
            "bash",
            str(submit_watershed_tuning_script_path()),
            "--dry-run",
            "--use-cached-preds",
        ],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "run_watershed_tune_predict.sh" not in result.stdout
    assert "run_watershed_tuning.sh" in result.stdout
    assert "--dependency=afterok" not in result.stdout
    assert "DRY-RUN tune from cached preds" in result.stderr


def test_submit_watershed_tuning_usage_points_at_runbook() -> None:
    result = subprocess.run(
        ["bash", str(submit_watershed_tuning_script_path()), "--help"],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    usage = result.stderr
    assert str(WATERSHED_TUNE_RUNBOOK_REL) in usage
    assert "watershed-tuning" in usage
    assert "predict" in usage.lower()


def test_watershed_tune_slurm_wall_time_covers_production_grid() -> None:
    assert WATERSHED_TUNE_TUNE_RESOURCES["time"] == "12:00:00"
    text = run_watershed_tuning_script_path().read_text(encoding="utf-8")
    assert "#SBATCH --time=12:00:00" in text


def test_unet_runbook_documents_predict_then_tune_workflow() -> None:
    path = watershed_tune_runbook_path()
    assert path.is_file(), f"missing runbook: {path}"
    text = path.read_text(encoding="utf-8")
    section = text.split("## Watershed tuning", 1)[1].split("## CC vs watershed", 1)[0]
    assert "run_watershed_tune_predict.sh" in section
    assert "run_watershed_tuning.sh" in section
    assert "watershed_tune_preds" in section
    assert "predict" in section.lower()
    assert "--preds-dir" in section


def test_unet_runbook_documents_production_grid_runtime_and_login_node_policy() -> (
    None
):
    section = watershed_tune_runbook_path().read_text(encoding="utf-8").split(
        "## Watershed tuning", 1
    )[1].split("## CC vs watershed", 1)[0]
    assert str(WATERSHED_TUNE_GRID_CONFIG_REL) in section
    assert "configured candidate count" in section
    assert "min_distance" in section
    assert "min_area_px" in section
    assert "whole-section PQ" in section
    assert "MergedViewPqResult" in section
    assert "login node" in section.lower()
    assert "SLURM" in section


def test_unet_runbook_documents_extraction_cache_ratio_and_log_verification() -> None:
    section = watershed_tune_runbook_path().read_text(encoding="utf-8").split(
        "## Watershed tuning", 1
    )[1].split("## CC vs watershed", 1)[0]
    assert "watershed_tune_extraction_cache" in section
    assert "24" in section
    assert "72" in section
    assert "extraction cache: hit" in section
    assert "extraction cache: miss" in section
    assert "--log-extraction-cache" in section
    assert "LOG_EXTRACTION_CACHE" in section


def test_unet_runbook_documents_grid_csv_row_order_for_diffing() -> None:
    section = watershed_tune_runbook_path().read_text(encoding="utf-8").split(
        "## Watershed tuning", 1
    )[1].split("## CC vs watershed", 1)[0]
    assert "Grid CSV row order" in section
    assert "itertools.product" in section
    assert "min_distance" in section
    assert "ridge_level" in section
