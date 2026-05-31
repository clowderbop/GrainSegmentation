"""SLURM wrapper contract for post-eval reporting."""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.slurm import (
    SLURM_JOB_RESOURCES,
    build_report_argv,
    run_build_report_script_path,
)


def test_build_report_argv_minimal() -> None:
    root = Path("/scratch/example/GrainSeg")
    argv = build_report_argv(grainseg_root=root)
    assert argv == [
        "uv",
        "run",
        "--group",
        "analysis",
        "python",
        "-m",
        "analysis.build_report",
        "--grainseg-root",
        str(root),
    ]


def test_build_report_argv_optional_flags() -> None:
    root = Path("/scratch/example/GrainSeg")
    out = Path("/scratch/example/GrainSeg/eval/reporting")
    argv = build_report_argv(
        grainseg_root=root,
        output_dir=out,
        strict=True,
        no_figures=True,
    )
    assert "--output-dir" in argv
    assert str(out) in argv
    assert "--strict" in argv
    assert "--no-figures" in argv


def test_slurm_job_resources_are_cpu_only() -> None:
    assert "gpus" not in SLURM_JOB_RESOURCES
    assert SLURM_JOB_RESOURCES["mem"] == "8G"


def test_run_build_report_script_exists_and_matches_argv() -> None:
    script = run_build_report_script_path()
    assert script.is_file(), f"Missing SLURM job script: {script}"
    text = script.read_text(encoding="utf-8")
    assert "analysis.build_report" in text
    assert "#SBATCH --job-name=post_eval_report" in text
    assert "uv sync --group analysis" in text
    for key in ("mem", "cpus-per-task", "time"):
        assert key in SLURM_JOB_RESOURCES
        assert f"#SBATCH --{key}={SLURM_JOB_RESOURCES[key]}" in text
