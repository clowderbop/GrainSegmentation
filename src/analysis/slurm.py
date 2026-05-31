"""SLURM job helpers for post-eval reporting (testable command contract)."""

from __future__ import annotations

from pathlib import Path

from common.variants import repo_root

SLURM_JOB_RESOURCES: dict[str, str] = {
    "job-name": "post_eval_report",
    "output": "logs/post_eval_report-%j.log",
    "mem": "8G",
    "cpus-per-task": "4",
    "time": "00:15:00",
}


def run_build_report_script_path() -> Path:
    return repo_root() / "SLURM" / "analysis" / "run_build_report.sh"


def build_report_argv(
    *,
    grainseg_root: Path,
    output_dir: Path | None = None,
    strict: bool = False,
    no_figures: bool = False,
) -> list[str]:
    """Argv for `uv run` to execute the reporting CLI (workspace root cwd)."""
    argv = [
        "uv",
        "run",
        "--group",
        "analysis",
        "python",
        "-m",
        "analysis.build_report",
        "--grainseg-root",
        str(grainseg_root),
    ]
    if output_dir is not None:
        argv.extend(["--output-dir", str(output_dir)])
    if strict:
        argv.append("--strict")
    if no_figures:
        argv.append("--no-figures")
    return argv
