from dataclasses import dataclass
import os
from pathlib import Path


DATASET_ROOT = Path("GrainSeg") / "dataset" / "train" / "yolo"
RUN_ROOT = Path("GrainSeg") / "runs" / "yolo26-seg"

# Typical SLURM resource hints live in SLURM/yolo/*.sh (mem, GPUs, time).

@dataclass(frozen=True)
class VariantConfig:
    name: str
    dataset_subdir: str
    yaml_name: str
    channels: int


VARIANT_CONFIGS: dict[str, VariantConfig] = {
    "PPL": VariantConfig(
        name="PPL",
        dataset_subdir="PPL",
        yaml_name="PPL.yaml",
        channels=1,
    ),
    "PPLPPXblend": VariantConfig(
        name="PPLPPXblend",
        dataset_subdir="PPLPPXblend",
        yaml_name="PPLPPXblend.yaml",
        channels=1,
    ),
    "PPL+PPXblend": VariantConfig(
        name="PPL+PPXblend",
        dataset_subdir="PPL+PPXblend",
        yaml_name="PPL_PPXblend.yaml",
        channels=6,
    ),
    "PPL+AllPPX": VariantConfig(
        name="PPL+AllPPX",
        dataset_subdir="PPL+AllPPX",
        yaml_name="PPL+AllPPX.yaml",
        channels=32,
    ),
}


def default_scratch_root(scratch_root: str | Path | None = None) -> Path:
    if scratch_root is not None:
        return Path(scratch_root)
    return Path(os.environ.get("SCRATCH", "/scratch"))


def default_dataset_root(scratch_root: str | Path | None = None) -> Path:
    return default_scratch_root(scratch_root) / DATASET_ROOT


def default_run_root(scratch_root: str | Path | None = None) -> Path:
    return default_scratch_root(scratch_root) / RUN_ROOT


def get_variant_config(name: str) -> VariantConfig:
    try:
        return VARIANT_CONFIGS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(VARIANT_CONFIGS))
        raise ValueError(
            f"Unknown YOLO variant {name!r}. Expected one of: {valid}"
        ) from exc


def variant_choices() -> tuple[str, ...]:
    return tuple(VARIANT_CONFIGS)
