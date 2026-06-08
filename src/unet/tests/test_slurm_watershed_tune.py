"""SLURM wrapper contracts for U-Net watershed predict-then-tune workflow."""

from __future__ import annotations

import subprocess
from pathlib import Path

from common.variants import repo_root
from unet.slurm_watershed_tune import (
    WATERSHED_TUNE_TUNE_RESOURCES,
    run_watershed_tuning_script_path,
    submit_watershed_tuning_script_path,
    watershed_tune_preds_semantic_dir,
)


def test_watershed_tune_preds_semantic_dir_is_per_variant_scratch_layout() -> None:
    """INTENT: semantic prediction cache path matches the per-variant scratch layout."""
    root = Path("/scratch/example/GrainSeg")
    assert watershed_tune_preds_semantic_dir(root, "PPL_AllPPX") == (
        root / "runs/watershed_tune_preds/PPL_AllPPX/semantic"
    )


def test_submit_watershed_tuning_supports_tune_only_from_cached_preds() -> None:
    """INTENT: --use-cached-preds dry-run submits tune only without predict dependency."""
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


def test_watershed_tune_slurm_wall_time_matches_resource_dict() -> None:
    """INTENT: tune-phase SLURM wall time in code matches the committed resource dict."""
    assert WATERSHED_TUNE_TUNE_RESOURCES["time"] == "12:00:00"
