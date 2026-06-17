"""SLURM wrapper contracts for U-Net watershed predict-then-tune workflow."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from common.variants import repo_root
from unet.slurm_watershed_tune import (
    WATERSHED_TUNE_MERGE_WALLTIME,
    run_watershed_tune_merge_script_path,
    run_watershed_tune_shard_script_path,
    submit_watershed_tuning_script_path,
    watershed_tune_monolithic_walltime_for_grid_config,
    watershed_tune_shard_walltime_for_grid_config,
    watershed_tune_walltime_for_combo_count,
    watershed_tune_walltimes_for_grid_config,
)
from unet.tests.watershed_tune_grid_fixtures import (
    grid_path_from_axes,
    minimal_grid_axes,
    write_watershed_tune_grid_config,
)
from unet.watershed_tune_grid_shard import watershed_tune_shard_count_for_grid_config


def _sbatch_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.startswith("sbatch")]


def _write_grid_config(path: Path, grid_axes: dict[str, object]) -> None:
    write_watershed_tune_grid_config(path, grid_axes)


def _shard_array_flag(grid_path: Path, *, max_parallel: int | None = None) -> str:
    shard_count = watershed_tune_shard_count_for_grid_config(grid_path)
    cap = max_parallel if max_parallel is not None else 6
    return f"--array=1-{shard_count}%{cap}"


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


def test_submit_watershed_tuning_default_dry_run_submits_shard_array_and_merge(
    tmp_path: Path,
) -> None:
    """INTENT: dry-run chains predict → shard array → merge with afterok per variant."""
    grid_path = grid_path_from_axes(
        tmp_path,
        minimal_grid_axes(min_distance=[5, 9], boundary_dilate_iter=[0, 1]),
    )
    env = {**os.environ, "GRID_CONFIG": str(grid_path)}
    expected_array = _shard_array_flag(grid_path)
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
        env=env,
    )
    assert result.returncode == 0, result.stderr
    stdout = result.stdout
    assert "run_watershed_tune_predict.sh" in stdout
    assert str(run_watershed_tune_shard_script_path().name) in stdout
    assert str(run_watershed_tune_merge_script_path().name) in stdout
    assert expected_array in stdout

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


def test_submit_watershed_tuning_use_cached_preds_dry_run_submits_shard_array_and_merge(
    tmp_path: Path,
) -> None:
    """INTENT: --use-cached-preds dry-run skips predict but chains shard array → merge."""
    grid_path = grid_path_from_axes(
        tmp_path,
        minimal_grid_axes(min_distance=[5, 9], boundary_dilate_iter=[0, 1]),
    )
    env = {**os.environ, "GRID_CONFIG": str(grid_path)}
    expected_array = _shard_array_flag(grid_path)
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
    stdout = result.stdout
    assert "run_watershed_tune_predict.sh" not in stdout
    assert str(run_watershed_tune_shard_script_path().name) in stdout
    assert str(run_watershed_tune_merge_script_path().name) in stdout
    assert expected_array in stdout
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


def test_submit_watershed_tuning_dry_run_honors_shard_max_parallel_env(
    tmp_path: Path,
) -> None:
    """INTENT: WATERSHED_TUNE_SHARD_MAX_PARALLEL throttles the shard array percent cap."""
    grid_path = grid_path_from_axes(
        tmp_path,
        minimal_grid_axes(min_distance=[5, 9], boundary_dilate_iter=[0, 1]),
    )
    expected_array = _shard_array_flag(grid_path, max_parallel=2)
    env = {
        **os.environ,
        "GRID_CONFIG": str(grid_path),
        "WATERSHED_TUNE_SHARD_MAX_PARALLEL": "2",
    }
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
    assert expected_array in result.stdout
    assert _shard_array_flag(grid_path) not in result.stdout


def test_watershed_tune_walltime_scales_with_combo_count() -> None:
    """INTENT: tune wall time grows with scored combo count and applies a minimum floor."""
    assert watershed_tune_walltime_for_combo_count(5) == "01:40:00"
    assert watershed_tune_walltime_for_combo_count(12) == "03:07:30"
    assert watershed_tune_walltime_for_combo_count(84) == "18:07:30"


def test_watershed_tune_walltime_roles_derive_from_grid_yaml(tmp_path: Path) -> None:
    """INTENT: monolithic and shard wall times are derived from the grid YAML at submit time."""
    shard_grid = grid_path_from_axes(
        tmp_path,
        minimal_grid_axes(
            h_maxima=[0, 2, 4, 6, 8, 10],
            min_area_px=[0, 64],
        ),
    )
    monolithic_grid = tmp_path / "monolithic_grid.yaml"
    write_watershed_tune_grid_config(
        monolithic_grid,
        minimal_grid_axes(
            min_distance=[5, 9],
            boundary_dilate_iter=[0, 1],
            h_maxima=[0, 4, 8],
            min_area_px=[0, 64],
        ),
    )
    assert watershed_tune_shard_walltime_for_grid_config(shard_grid) == "03:07:30"
    assert (
        watershed_tune_monolithic_walltime_for_grid_config(monolithic_grid)
        == "05:37:30"
    )


def test_watershed_tune_walltimes_for_grid_config_matches_per_role_helpers(
    tmp_path: Path,
) -> None:
    """INTENT: batch walltime helper returns the same values as per-role lookups."""
    grid_path = grid_path_from_axes(
        tmp_path,
        minimal_grid_axes(h_maxima=[0, 4], min_area_px=[0]),
    )
    shard, monolithic, merge = watershed_tune_walltimes_for_grid_config(grid_path)
    assert shard == watershed_tune_shard_walltime_for_grid_config(grid_path)
    assert monolithic == watershed_tune_monolithic_walltime_for_grid_config(grid_path)
    assert merge == WATERSHED_TUNE_MERGE_WALLTIME


def test_watershed_tune_walltime_cli_all_prints_three_values_on_one_line(
    tmp_path: Path,
) -> None:
    """INTENT: walltime CLI --all emits shard, monolithic, and merge on one line."""
    grid_path = grid_path_from_axes(
        tmp_path,
        minimal_grid_axes(h_maxima=[0, 4], min_area_px=[0]),
    )
    expected = watershed_tune_walltimes_for_grid_config(grid_path)
    result = subprocess.run(
        [
            "uv",
            "run",
            "--directory",
            str(repo_root() / "src" / "unet"),
            "python",
            "-m",
            "unet.watershed_tune_walltime",
            "--all",
            "--grid-config",
            str(grid_path),
        ],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == " ".join(expected)


def test_submit_watershed_tuning_dry_run_passes_grid_derived_walltime(
    tmp_path: Path,
) -> None:
    """INTENT: submit script passes sbatch --time derived from the grid config."""
    grid_path = grid_path_from_axes(
        tmp_path,
        minimal_grid_axes(h_maxima=[0, 4], min_area_px=[0]),
    )
    expected_walltime = watershed_tune_shard_walltime_for_grid_config(grid_path)
    assert expected_walltime == "01:02:30"
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
    assert f"--time={expected_walltime}" in result.stdout
    assert f"--time={WATERSHED_TUNE_MERGE_WALLTIME}" in result.stdout
