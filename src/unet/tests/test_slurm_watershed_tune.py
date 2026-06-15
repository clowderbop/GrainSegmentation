"""SLURM wrapper contracts for U-Net watershed predict-then-tune workflow."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

from common.variants import repo_root
from unet.slurm_watershed_tune import (
    run_watershed_tune_merge_script_path,
    run_watershed_tune_shard_script_path,
    submit_watershed_tuning_script_path,
)
from unet.watershed_tune_grid_shard import watershed_tune_shard_count_for_grid_config


def _sbatch_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.startswith("sbatch")]


def _write_grid_config(path: Path, grid_axes: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump({"grid": grid_axes}), encoding="utf-8")


def _three_axis_grid_yaml(tmp_path: Path) -> Path:
    grid_path = tmp_path / "grid.yaml"
    _write_grid_config(
        grid_path,
        {
            "min_distance": [5, 9, 11],
            "h_maxima": [0],
            "boundary_dilate_iter": [0],
            "watershed_connectivity": [1],
            "min_area_px": [0],
            "exclude_border": [0],
            "ridge_level": [None],
        },
    )
    return grid_path


def test_submit_watershed_tuning_default_dry_run_submits_shard_array_and_merge() -> (
    None
):
    """INTENT: default dry-run chains predict → shard array → merge with afterok per variant."""
    result = subprocess.run(
        [
            "bash",
            str(submit_watershed_tuning_script_path()),
            "--dry-run",
        ],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "run_watershed_tune_predict.sh" in stdout
    assert str(run_watershed_tune_shard_script_path().name) in stdout
    assert str(run_watershed_tune_merge_script_path().name) in stdout
    assert "--array=1-6%6" in stdout

    predict_lines = [
        line
        for line in _sbatch_lines(stdout)
        if "run_watershed_tune_predict.sh" in line
    ]
    shard_lines = [
        line for line in _sbatch_lines(stdout) if "run_watershed_tune_shard.sh" in line
    ]
    merge_lines = [
        line for line in _sbatch_lines(stdout) if "run_watershed_tune_merge.sh" in line
    ]
    assert len(predict_lines) == len(shard_lines) == len(merge_lines)
    for predict, shard, merge in zip(
        predict_lines, shard_lines, merge_lines, strict=True
    ):
        assert "--dependency=afterok" not in predict
        assert "--dependency=afterok:DRYRUN" in shard
        assert "--dependency=afterok:DRYRUN" in merge


def test_submit_watershed_tuning_single_job_dry_run_skips_shard_and_merge() -> None:
    """INTENT: --single-job dry-run submits one monolithic tune job without array or merge."""
    result = subprocess.run(
        [
            "bash",
            str(submit_watershed_tuning_script_path()),
            "--dry-run",
            "--single-job",
        ],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "run_watershed_tuning.sh" in stdout
    assert str(run_watershed_tune_shard_script_path().name) not in stdout
    assert str(run_watershed_tune_merge_script_path().name) not in stdout
    assert "--array=" not in stdout


def test_submit_watershed_tuning_use_cached_preds_dry_run_submits_shard_array_and_merge() -> (
    None
):
    """INTENT: --use-cached-preds dry-run skips predict but chains shard array → merge."""
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
    stdout = result.stdout
    assert "run_watershed_tune_predict.sh" not in stdout
    assert str(run_watershed_tune_shard_script_path().name) in stdout
    assert str(run_watershed_tune_merge_script_path().name) in stdout
    assert "--array=1-6%6" in stdout
    assert "DRY-RUN shard array from cached preds" in result.stderr

    shard_lines = [
        line for line in _sbatch_lines(stdout) if "run_watershed_tune_shard.sh" in line
    ]
    merge_lines = [
        line for line in _sbatch_lines(stdout) if "run_watershed_tune_merge.sh" in line
    ]
    assert len(shard_lines) == len(merge_lines)
    for shard, merge in zip(shard_lines, merge_lines, strict=True):
        assert "--dependency=afterok" not in shard
        assert "--dependency=afterok:DRYRUN" in merge


def test_watershed_tune_shard_count_for_grid_config_derives_from_yaml(
    tmp_path: Path,
) -> None:
    """INTENT: SLURM shard array size is derived from grid YAML at submit time."""
    grid_path = tmp_path / "grid.yaml"
    _write_grid_config(
        grid_path,
        {
            "min_distance": [5, 9],
            "h_maxima": [0],
            "boundary_dilate_iter": [0, 1],
            "watershed_connectivity": [1],
            "min_area_px": [0],
            "exclude_border": [0],
            "ridge_level": [None],
        },
    )
    assert watershed_tune_shard_count_for_grid_config(grid_path) == 4


def test_submit_watershed_tuning_dry_run_array_size_matches_custom_grid_yaml(
    tmp_path: Path,
) -> None:
    """INTENT: submit script reads GRID_CONFIG env to size the shard job array."""
    grid_path = _three_axis_grid_yaml(tmp_path)
    env = {**os.environ, "GRID_CONFIG": str(grid_path)}
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
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "--array=1-3" in result.stdout


def test_submit_watershed_tuning_dry_run_honors_grid_config_flag(
    tmp_path: Path,
) -> None:
    """INTENT: submit --grid-config overrides default grid for array sizing and export."""
    grid_path = _three_axis_grid_yaml(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(submit_watershed_tuning_script_path()),
            "--dry-run",
            "--use-cached-preds",
            "--grid-config",
            str(grid_path),
        ],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--array=1-3" in result.stdout
    assert f"GRID_CONFIG={grid_path}" in result.stdout


def test_submit_watershed_tuning_dry_run_honors_shard_max_parallel_env() -> None:
    """INTENT: WATERSHED_TUNE_SHARD_MAX_PARALLEL throttles the shard array percent cap."""
    env = {**os.environ, "WATERSHED_TUNE_SHARD_MAX_PARALLEL": "2"}
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
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert "--array=1-6%2" in result.stdout
    assert "--array=1-6%6" not in result.stdout
