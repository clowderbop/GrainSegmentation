"""SLURM wrapper contracts for U-Net watershed predict-then-tune workflow."""

from __future__ import annotations

import subprocess

from common.variants import repo_root
from unet.slurm_watershed_tune import submit_watershed_tuning_script_path


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
