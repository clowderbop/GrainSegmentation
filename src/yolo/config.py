import os
from dataclasses import dataclass
from pathlib import Path

from common.variants import all_variant_names, get_variant

DATASET_ROOT = Path("GrainSeg") / "dataset" / "train" / "yolo"
RUN_ROOT = Path("GrainSeg") / "runs" / "yolo26-seg"

# Typical SLURM resource hints live in SLURM/yolo/*.sh (mem, GPUs, time).


@dataclass(frozen=True)
class VariantConfig:
    """YOLO-facing view of a registry variant (channels = stacked TIFF depth)."""

    name: str
    dataset_subdir: str
    yaml_name: str
    channels: int


def default_scratch_root(scratch_root: str | Path | None = None) -> Path:
    if scratch_root is not None:
        return Path(scratch_root)
    return Path(os.environ.get("SCRATCH", "/scratch"))


def default_dataset_root(scratch_root: str | Path | None = None) -> Path:
    return default_scratch_root(scratch_root) / DATASET_ROOT


def default_run_root(scratch_root: str | Path | None = None) -> Path:
    return default_scratch_root(scratch_root) / RUN_ROOT


def get_variant_config(name: str) -> VariantConfig:
    spec = get_variant(name)
    return VariantConfig(
        name=spec.name,
        dataset_subdir=spec.yolo.dataset_subdir,
        yaml_name=spec.yolo.yaml_name,
        channels=spec.yolo.input_channels,
    )


def variant_choices() -> tuple[str, ...]:
    return all_variant_names()
