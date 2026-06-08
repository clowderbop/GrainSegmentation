"""SLURM wrapper contracts for YOLO profile selection (ADR 0005)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from common.variants import repo_root
from yolo.inference_profile_tune import load_tune_grid
from yolo.slurm_profile_tune import (
    PROFILE_TUNE_DETECTOR_WALLTIME_LONG,
    PROFILE_TUNE_DETECTOR_WALLTIME_SHORT,
    PROFILE_TUNE_RUNBOOK_REL,
    profile_tune_detector_walltime,
    submit_inference_profile_tune_script_path,
)


def test_profile_tune_detector_walltime_tiers_from_detector_key_count(
    tmp_path: Path,
) -> None:
    """INTENT: detector SLURM wall time tier follows conf-axis count in the grid YAML."""
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


def test_submit_profile_tune_usage_points_at_runbook() -> None:
    """INTENT: submit script --help references the profile-selection runbook."""
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
