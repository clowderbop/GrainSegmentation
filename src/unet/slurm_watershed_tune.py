"""SLURM job helpers for U-Net watershed predict-then-tune (testable contract)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from common.variants import repo_root

from unet.watershed_tune_grid import (
    WatershedTuneGrid,
    load_watershed_tune_grid,
    watershed_tune_candidate_count,
    watershed_tune_grid_path,
)
from unet.watershed_tune_grid_shard import (
    iter_watershed_tune_shards,
    watershed_tune_shard_combo_count,
)

WatershedTuneWalltimeRole = Literal["monolithic", "shard", "merge"]

# Jun 2026 train whole-section timing (52k×10k):
# - Pre-/low-h_maxima grids: ~10 min/combo (~2 h / 12 combos per shard).
# - Refinement h_maxima≥28: ~18 min/combo (multi-channel) to ~28 min/combo (PPL,
#   PPLPPXblend); use the high end so monolithic jobs do not time out.
# See docs/runbooks/unet.md#watershed-tuning.
WATERSHED_TUNE_WALLTIME_SETUP_SECONDS = 30 * 60
WATERSHED_TUNE_WALLTIME_SECONDS_PER_COMBO = 28 * 60
WATERSHED_TUNE_WALLTIME_HEADROOM = 1.25
WATERSHED_TUNE_WALLTIME_MIN_SECONDS = 60 * 60
WATERSHED_TUNE_WALLTIME_MAX_SECONDS = 48 * 60 * 60
WATERSHED_TUNE_MERGE_WALLTIME = "00:30:00"
WATERSHED_TUNE_PREDICT_WALLTIME = "00:15:00"

WATERSHED_TUNE_PREDICT_RESOURCES: dict[str, str] = {
    "job-name": "PredWatershed",
    "mem": "256G",
    "cpus-per-task": "8",
    "gpus-per-node": "rtx_pro_6000:1",
    "time": WATERSHED_TUNE_PREDICT_WALLTIME,
}

WATERSHED_TUNE_TUNE_RESOURCES: dict[str, str] = {
    "job-name": "TuneWatershed",
    "mem": "256G",
    "cpus-per-task": "8",
    "time": "12:00:00",
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


def run_watershed_tune_shard_script_path() -> Path:
    return repo_root() / "SLURM" / "unet" / "run_watershed_tune_shard.sh"


def run_watershed_tune_merge_script_path() -> Path:
    return repo_root() / "SLURM" / "unet" / "run_watershed_tune_merge.sh"


def format_slurm_walltime(total_seconds: int) -> str:
    """Format seconds as SLURM ``D-HH:MM:SS`` or ``HH:MM:SS``."""
    if total_seconds < 0:
        raise ValueError("total_seconds must be >= 0")
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def watershed_tune_walltime_seconds_for_combo_count(combo_count: int) -> int:
    if combo_count < 1:
        raise ValueError("combo_count must be >= 1")
    raw_seconds = (
        WATERSHED_TUNE_WALLTIME_SETUP_SECONDS
        + combo_count * WATERSHED_TUNE_WALLTIME_SECONDS_PER_COMBO
    )
    scaled = int(raw_seconds * WATERSHED_TUNE_WALLTIME_HEADROOM)
    return max(
        WATERSHED_TUNE_WALLTIME_MIN_SECONDS,
        min(scaled, WATERSHED_TUNE_WALLTIME_MAX_SECONDS),
    )


def watershed_tune_walltime_for_combo_count(combo_count: int) -> str:
    """Estimate tune-job wall time from scored combo count (train whole section)."""
    return format_slurm_walltime(
        watershed_tune_walltime_seconds_for_combo_count(combo_count)
    )


def watershed_tune_shard_combo_count_for_grid(grid: WatershedTuneGrid) -> int:
    first_shard = next(iter(iter_watershed_tune_shards(grid)))
    return watershed_tune_shard_combo_count(grid, first_shard)


def watershed_tune_monolithic_walltime_for_grid_config(
    grid_config: Path | None = None,
) -> str:
    grid = load_watershed_tune_grid(watershed_tune_grid_path(grid_config)).grid
    return watershed_tune_walltime_for_combo_count(watershed_tune_candidate_count(grid))


def watershed_tune_shard_walltime_for_grid_config(
    grid_config: Path | None = None,
) -> str:
    grid = load_watershed_tune_grid(watershed_tune_grid_path(grid_config)).grid
    return watershed_tune_walltime_for_combo_count(
        watershed_tune_shard_combo_count_for_grid(grid)
    )


def watershed_tune_walltimes_for_grid_config(
    grid_config: Path | None = None,
) -> tuple[str, str, str]:
    """Return ``(shard, monolithic, merge)`` SLURM walltimes for one grid YAML."""
    grid = load_watershed_tune_grid(watershed_tune_grid_path(grid_config)).grid
    shard = watershed_tune_walltime_for_combo_count(
        watershed_tune_shard_combo_count_for_grid(grid)
    )
    monolithic = watershed_tune_walltime_for_combo_count(
        watershed_tune_candidate_count(grid)
    )
    return shard, monolithic, WATERSHED_TUNE_MERGE_WALLTIME


def watershed_tune_walltime_for_role(
    role: WatershedTuneWalltimeRole,
    *,
    grid_config: Path | None = None,
) -> str:
    if role == "merge":
        return WATERSHED_TUNE_MERGE_WALLTIME
    shard, monolithic, _merge = watershed_tune_walltimes_for_grid_config(grid_config)
    if role == "monolithic":
        return monolithic
    if role == "shard":
        return shard
    raise ValueError(f"unsupported watershed tune walltime role: {role!r}")
