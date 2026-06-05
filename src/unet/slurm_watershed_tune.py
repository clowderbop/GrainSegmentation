"""SLURM job helpers for U-Net watershed predict-then-tune (testable contract)."""

from __future__ import annotations

from pathlib import Path

from common.variants import repo_root

WATERSHED_TUNE_PREDICT_RESOURCES: dict[str, str] = {
    "job-name": "PredWatershed",
    "mem": "256G",
    "cpus-per-task": "8",
    "gpus-per-node": "rtx_pro_6000:1",
    "time": "04:00:00",
}

WATERSHED_TUNE_TUNE_RESOURCES: dict[str, str] = {
    "job-name": "TuneWatershed",
    "mem": "256G",
    "cpus-per-task": "8",
    "time": "04:00:00",
}

WATERSHED_TUNE_RUNBOOK_REL = Path("docs/runbooks/unet.md")
WATERSHED_TUNE_PREDS_ROOT_REL = "runs/watershed_tune_preds"
WATERSHED_SEMANTIC_PREDS_DIR_NAME = "semantic"


def watershed_tune_preds_semantic_dir(grainseg_root: Path, variant_subdir: str) -> Path:
    return (
        grainseg_root
        / WATERSHED_TUNE_PREDS_ROOT_REL
        / variant_subdir
        / WATERSHED_SEMANTIC_PREDS_DIR_NAME
    )


def watershed_tune_runbook_path() -> Path:
    return repo_root() / WATERSHED_TUNE_RUNBOOK_REL


def run_watershed_tune_predict_script_path() -> Path:
    return repo_root() / "SLURM" / "unet" / "run_watershed_tune_predict.sh"


def run_watershed_tuning_script_path() -> Path:
    return repo_root() / "SLURM" / "unet" / "run_watershed_tuning.sh"


def submit_watershed_tuning_script_path() -> Path:
    return repo_root() / "SLURM" / "unet" / "submit_watershed_tuning.sh"
