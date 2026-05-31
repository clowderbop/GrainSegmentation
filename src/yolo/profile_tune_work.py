"""Shared staging and subprocess helpers for YOLO profile tune."""

from __future__ import annotations

import subprocess
from pathlib import Path

from common.manifest_io import build_yolo_whole_manifest, write_dataset_manifest
from common.test_inference import load_test_inference_recipe, sahi_overlap_ratio
from yolo.config import default_run_root, default_scratch_root


def uv_run_cmd(
    project_dir: Path,
    module: str,
    *args: str,
    unbuffered: bool = False,
) -> list[str]:
    python_cmd = ["python", "-u"] if unbuffered else ["python"]
    return [
        "uv",
        "run",
        "--directory",
        str(project_dir),
        *python_cmd,
        "-m",
        module,
        *args,
    ]


def run_subprocess(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def weights_path(grainseg_root: Path, variant: str, run_root: Path) -> Path:
    return run_root / variant / "weights" / "best.pt"


def staged_manifest_path(work_root: Path, variant: str) -> Path:
    return work_root / variant / "staged" / "manifest.json"


def prepare_train_whole_manifest(
    grainseg_root: Path, variant: str, work_root: Path
) -> Path:
    manifest_path = work_root / variant / "yolo_whole_train.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_dataset_manifest(
        manifest_path,
        build_yolo_whole_manifest(
            split="train", variant=variant, grainseg_root=grainseg_root
        ),
    )
    return manifest_path


def ensure_staged_train_manifest(
    *,
    grainseg_root: Path,
    variant: str,
    work_root: Path,
    repo: Path,
) -> Path:
    canonical_manifest = prepare_train_whole_manifest(grainseg_root, variant, work_root)
    staged_root = work_root / variant / "staged"
    staged_manifest = staged_manifest_path(work_root, variant)
    if staged_manifest.is_file():
        return staged_manifest
    staged_root.mkdir(parents=True, exist_ok=True)
    run_subprocess(
        uv_run_cmd(
            repo / "src" / "common",
            "common.stage_manifest",
            "run",
            str(canonical_manifest),
            str(staged_root),
        )
    )
    if not staged_manifest.is_file():
        raise FileNotFoundError(f"Staged manifest missing: {staged_manifest}")
    return staged_manifest


def sahi_window_kwargs() -> dict[str, int | float]:
    recipe = load_test_inference_recipe()
    overlap = sahi_overlap_ratio(window=recipe.whole.window, stride=recipe.whole.stride)
    return {
        "slice_height": recipe.whole.window,
        "slice_width": recipe.whole.window,
        "overlap_height_ratio": overlap,
        "overlap_width_ratio": overlap,
    }


def default_grainseg_and_run_roots(
    grainseg_root: Path | None, run_root: Path | None
) -> tuple[Path, Path]:
    resolved_grainseg = grainseg_root or default_scratch_root() / "GrainSeg"
    resolved_run = run_root or default_run_root()
    return resolved_grainseg, resolved_run


def evaluate_variant_predictions(
    *,
    variant: str,
    variant_output_dir: Path,
    staged_manifest: Path,
    repo: Path,
) -> Path:
    """Write eval manifest and run instance AJI for prediction sets in variant_output_dir."""
    variant_output_dir.mkdir(parents=True, exist_ok=True)
    common_src = repo / "src" / "common"
    eval_manifest = variant_output_dir / "eval_manifest.json"
    run_subprocess(
        uv_run_cmd(
            common_src,
            "common.stage_manifest",
            "write-eval",
            "--source",
            str(staged_manifest),
            "--prediction-set-dir",
            str(variant_output_dir),
            "--output",
            str(eval_manifest),
        )
    )
    metrics_path = variant_output_dir / "instance_metrics.json"
    run_subprocess(
        uv_run_cmd(
            common_src,
            "common.evaluate_instances",
            "--unit",
            "whole",
            "--model-type",
            "yolo",
            "--variant",
            variant,
            "--manifest",
            str(eval_manifest),
            "--output-json",
            str(metrics_path),
            unbuffered=True,
        )
    )
    return metrics_path
