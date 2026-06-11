"""SLURM wrapper contracts for CC vs watershed train-section eval."""

from __future__ import annotations

import subprocess
from pathlib import Path

from common.variants import repo_root
from unet.extraction_tune_scoring import WatershedParamSet
from unet.tests.watershed_best_json_fixtures import write_watershed_best_json


def _sbatch_lines(stdout: str) -> list[str]:
    return [line for line in stdout.splitlines() if line.startswith("sbatch")]


def _run_build_cc_extract_args(
    *,
    model_path: Path,
    model_dir: Path,
    tune_root: Path,
    cc_min_area_px: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env_exports = ""
    if cc_min_area_px is not None:
        env_exports = f'export CC_MIN_AREA_PX="{cc_min_area_px}"\n'
    bash = f"""
set -euo pipefail
{env_exports}REPO_ROOT="{repo_root()}"
SLURM_ROOT="$REPO_ROOT/SLURM"
# shellcheck source=SLURM/utils/watershed.sh
source "$SLURM_ROOT/utils/watershed.sh"
MODEL_DIR="{model_dir}"
WATERSHED_TUNE_ROOT="{tune_root}"
export VARIANT=PPL
build_cc_extract_args "{model_path}" ""
printf 'resolved=%s\\n' "$RESOLVED_WATERSHED_JSON"
printf '%s\\n' "${{extract_args[*]}}"
"""
    return subprocess.run(
        ["bash", "-c", bash],
        capture_output=True,
        text=True,
        check=False,
    )


def test_submit_cc_vs_watershed_train_eval_dry_run_passes_tune_root_to_cc_job() -> None:
    """INTENT: CC train eval job receives --watershed-tune-root like the watershed job."""
    result = subprocess.run(
        [
            "bash",
            str(repo_root() / "SLURM/unet/submit_cc_vs_watershed_train_eval.sh"),
            "--dry-run",
        ],
        cwd=repo_root(),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    sbatch_lines = _sbatch_lines(result.stdout)
    cc_lines = [line for line in sbatch_lines if "instance_val_cc" in line]
    ws_lines = [line for line in sbatch_lines if "instance_val_watershed" in line]
    assert len(cc_lines) == 1
    assert len(ws_lines) == 1
    assert "--watershed-tune-root" in cc_lines[0]
    assert "--instance-method" in cc_lines[0] and " cc " in f" {cc_lines[0]} "


def test_build_cc_extract_args_uses_tune_json_min_area_px(tmp_path: Path) -> None:
    """INTENT: CC whole eval passes tuned min_area_px from watershed_best JSON to extract_instances."""
    tune_root = tmp_path / "watershed_tune"
    variant_dir = tune_root / "PPL"
    variant_dir.mkdir(parents=True)
    params = WatershedParamSet(5, 0, 1, 256, False, None)
    write_watershed_best_json(variant_dir / "watershed_best_1.json", params)

    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "unet_finetuned_PPL.keras"
    model_path.write_text("stub", encoding="utf-8")

    result = _run_build_cc_extract_args(
        model_path=model_path,
        model_dir=model_dir,
        tune_root=tune_root,
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].startswith("resolved=")
    assert lines[0].endswith("watershed_best_1.json")
    tokens = lines[1].split()
    assert tokens[:2] == ["--instance-method", "cc"]
    min_area_idx = tokens.index("--min-area-px")
    assert tokens[min_area_idx + 1] == "256"


def test_build_cc_extract_args_cc_min_area_px_override_skips_tune_root_resolution(
    tmp_path: Path,
) -> None:
    """INTENT: CC_MIN_AREA_PX override wins without requiring a valid watershed tune root."""
    tune_root = tmp_path / "missing_tune_root"
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "unet_finetuned_PPL.keras"
    model_path.write_text("stub", encoding="utf-8")

    result = _run_build_cc_extract_args(
        model_path=model_path,
        model_dir=model_dir,
        tune_root=tune_root,
        cc_min_area_px="128",
    )

    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "resolved="
    tokens = lines[1].split()
    assert tokens == ["--instance-method", "cc", "--min-area-px", "128"]
