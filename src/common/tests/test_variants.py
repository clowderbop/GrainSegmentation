"""Phase 1: common.variants registry loader and CLI helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from common.variants import (
    all_variant_names,
    default_grainseg_root,
    get_variant,
    load_registry,
    registry_path,
    repo_root,
    validate,
)

EXPECTED_VARIANTS = ("PPL", "PPLPPXblend", "PPL+PPXblend", "PPL+AllPPX")
EXPECTED_YOLO_CHANNELS = {
    "PPL": 3,
    "PPLPPXblend": 3,
    "PPL+PPXblend": 6,
    "PPL+AllPPX": 21,
}
EXPECTED_WATERSHED_JOB_SLUGS = {
    "PPL": "PPL",
    "PPLPPXblend": "PPLPPXblend",
    "PPL+PPXblend": "PPL_PlusPPXblend",
    "PPL+AllPPX": "PPL_AllPPX",
}


def test_registry_path_under_repo_root() -> None:
    assert registry_path() == repo_root() / "config" / "variants.yaml"
    assert registry_path().is_file()


def test_load_registry_has_four_variants() -> None:
    reg = load_registry()
    assert reg.schema_version == 1
    assert tuple(reg.variants) == EXPECTED_VARIANTS
    validate(reg)


@pytest.mark.parametrize("name", EXPECTED_VARIANTS)
def test_yolo_input_channels(name: str) -> None:
    spec = get_variant(name)
    assert spec.yolo.input_channels == EXPECTED_YOLO_CHANNELS[name]


@pytest.mark.parametrize("name", EXPECTED_VARIANTS)
def test_watershed_job_slug(name: str) -> None:
    assert get_variant(name).slugs.job == EXPECTED_WATERSHED_JOB_SLUGS[name]


@pytest.mark.parametrize("name", EXPECTED_VARIANTS)
def test_unet_suffix_count_matches_num_inputs(name: str) -> None:
    spec = get_variant(name)
    assert len(spec.unet.input_suffixes) == spec.unet.num_inputs


def test_resolve_paths_under_grainseg_root(tmp_path: Path) -> None:
    grainseg = tmp_path / "GrainSeg"
    spec = get_variant("PPL+AllPPX")
    resolved = spec.resolve_paths(grainseg)
    assert resolved.test_mosaic_stacked == grainseg / "dataset/test/test_PPL+AllPPX.tif"
    assert resolved.train_channel_path("_PPX3") == grainseg / "dataset/train/train_PPX3.tif"


def test_all_variant_names_matches_registry_order() -> None:
    assert all_variant_names() == EXPECTED_VARIANTS


def test_get_variant_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown microscopy variant"):
        get_variant("nope")


def _run_variants_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "python", "-m", "common.variants", *args],
        cwd=repo_root(),
        check=True,
        capture_output=True,
        text=True,
    )


def test_cli_env_exports_num_inputs() -> None:
    proc = _run_variants_cli(
        "--grainseg-root",
        str(default_grainseg_root("/scratch/example")),
        "env",
        "--variant",
        "PPL+AllPPX",
    )
    assert "export NUM_INPUTS=7" in proc.stdout
    assert "IMAGE_SUFFIXES=(_PPL _PPX1 _PPX2 _PPX3 _PPX4 _PPX5 _PPX6)" in proc.stdout
    assert "export YOLO_INPUT_CHANNELS=21" in proc.stdout


def test_cli_print_json() -> None:
    proc = _run_variants_cli("print-json", "--variant", "PPL")
    payload = json.loads(proc.stdout)
    assert payload["yolo"]["input_channels"] == 3


def test_yolo_config_reexports_registry_channels() -> None:
    from yolo.config import get_variant_config

    assert get_variant_config("PPL+AllPPX").channels == 21
    assert get_variant_config("PPL").channels == 3
