"""Locate finished eval artifacts on a grainseg root (path conventions v1)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.variants import all_variant_names, unet_finetuned_eval_run_name


@dataclass(frozen=True)
class EvalRunRef:
    producer: str
    variant: str
    unit: str
    instance_metrics_path: Path
    mask_ap_metrics_path: Path | None = None
    patch_job_dir: Path | None = None


@dataclass(frozen=True)
class UltralyticsValRef:
    variant: str
    metrics_path: Path


class DiscoveryError(FileNotFoundError):
    """Raised when required eval artifacts are missing (strict discovery)."""


def yolo_whole_eval_dir(grainseg_root: Path, variant: str) -> Path:
    return grainseg_root / "eval" / f"yolo_{variant}"


def unet_whole_eval_dir(grainseg_root: Path, variant: str) -> Path:
    """Whole-section U-Net test eval dir (run name omits .keras from model basename)."""
    return grainseg_root / "eval" / "unet_test" / unet_finetuned_eval_run_name(variant)


def yolo_patch_variant_dir(grainseg_root: Path, variant: str) -> Path:
    return grainseg_root / "eval" / "yolo_patches" / variant


def unet_patch_variant_dir(grainseg_root: Path, variant: str) -> Path:
    return grainseg_root / "eval" / "unet_patches" / variant


def ultralytics_val_metrics_path(grainseg_root: Path, variant: str) -> Path:
    return grainseg_root / "runs" / "yolo26-seg-val" / variant / "test" / "metrics.json"


def latest_patch_job_dir(variant_dir: Path) -> Path:
    """Newest subdir under a patch eval variant folder that has instance_metrics.json."""
    if not variant_dir.is_dir():
        raise DiscoveryError(f"Patch eval directory not found: {variant_dir}")
    candidates: list[tuple[float, Path]] = []
    for child in variant_dir.iterdir():
        if not child.is_dir():
            continue
        metrics = child / "instance_metrics.json"
        if metrics.is_file():
            candidates.append((metrics.stat().st_mtime, child))
    if not candidates:
        raise DiscoveryError(
            f"No patch job with instance_metrics.json under {variant_dir}"
        )
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def discover_eval_runs(
    grainseg_root: Path,
    *,
    variants: tuple[str, ...] | None = None,
    strict: bool = False,
) -> list[EvalRunRef]:
    root = grainseg_root.resolve()
    variant_names = variants or all_variant_names()
    runs: list[EvalRunRef] = []
    missing: list[str] = []

    for variant in variant_names:
        yolo_whole = yolo_whole_eval_dir(root, variant) / "instance_metrics.json"
        if yolo_whole.is_file():
            mask_ap = yolo_whole_eval_dir(root, variant) / "mask_ap_metrics.json"
            runs.append(
                EvalRunRef(
                    producer="yolo",
                    variant=variant,
                    unit="whole",
                    instance_metrics_path=yolo_whole,
                    mask_ap_metrics_path=mask_ap if mask_ap.is_file() else None,
                )
            )
        else:
            missing.append(f"yolo whole {variant}: {yolo_whole}")

        unet_whole = unet_whole_eval_dir(root, variant) / "instance_metrics.json"
        if unet_whole.is_file():
            runs.append(
                EvalRunRef(
                    producer="unet",
                    variant=variant,
                    unit="whole",
                    instance_metrics_path=unet_whole,
                )
            )
        else:
            missing.append(f"unet whole {variant}: {unet_whole}")

        yolo_patch_root = yolo_patch_variant_dir(root, variant)
        if yolo_patch_root.is_dir():
            try:
                job_dir = latest_patch_job_dir(yolo_patch_root)
                metrics = job_dir / "instance_metrics.json"
                runs.append(
                    EvalRunRef(
                        producer="yolo",
                        variant=variant,
                        unit="patch",
                        instance_metrics_path=metrics,
                        patch_job_dir=job_dir,
                    )
                )
            except DiscoveryError as exc:
                missing.append(str(exc))

        unet_patch_root = unet_patch_variant_dir(root, variant)
        if unet_patch_root.is_dir():
            try:
                job_dir = latest_patch_job_dir(unet_patch_root)
                metrics = job_dir / "instance_metrics.json"
                runs.append(
                    EvalRunRef(
                        producer="unet",
                        variant=variant,
                        unit="patch",
                        instance_metrics_path=metrics,
                        patch_job_dir=job_dir,
                    )
                )
            except DiscoveryError as exc:
                missing.append(str(exc))

    if strict and missing:
        detail = "\n".join(f"  - {line}" for line in missing)
        raise DiscoveryError(f"Missing eval artifacts:\n{detail}")
    return runs


def discover_ultralytics_val(
    grainseg_root: Path,
    *,
    variants: tuple[str, ...] | None = None,
) -> list[UltralyticsValRef]:
    root = grainseg_root.resolve()
    variant_names = variants or all_variant_names()
    refs: list[UltralyticsValRef] = []
    for variant in variant_names:
        path = ultralytics_val_metrics_path(root, variant)
        if path.is_file():
            refs.append(UltralyticsValRef(variant=variant, metrics_path=path))
    return refs
