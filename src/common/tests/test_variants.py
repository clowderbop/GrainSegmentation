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
    variant_display_names_in_thesis_order,
    variants_in_thesis_order,
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
EXPECTED_DISPLAY_NAMES = {
    "PPL": "PPL",
    "PPL+AllPPX": "FullStack",
    "PPL+PPXblend": "PPL+XPLComp",
    "PPLPPXblend": "FullComp",
}
THESIS_VARIANT_ORDER = ("PPL", "PPL+AllPPX", "PPL+PPXblend", "PPLPPXblend")


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
    assert spec.unet.channels_per_input == 3


def test_ppl_plus_ppx_yaml_name_differs_from_subdir() -> None:
    yolo = get_variant("PPL+PPXblend").yolo
    assert yolo.dataset_subdir == "PPL+PPXblend"
    assert yolo.yaml_name == "PPL_PPXblend.yaml"


def test_stacked_mosaic_paths_use_literal_variant() -> None:
    paths = get_variant("PPL+AllPPX").paths
    assert paths.train_mosaic_stacked == "dataset/train/train_PPL+AllPPX.tif"
    assert paths.test_mosaic_stacked == "dataset/test/test_PPL+AllPPX.tif"


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


@pytest.mark.parametrize("name", EXPECTED_VARIANTS)
def test_display_name(name: str) -> None:
    assert get_variant(name).display_name == EXPECTED_DISPLAY_NAMES[name]


def test_variants_in_thesis_order() -> None:
    assert variants_in_thesis_order() == THESIS_VARIANT_ORDER


def test_variant_display_names_in_thesis_order() -> None:
    assert variant_display_names_in_thesis_order() == tuple(
        EXPECTED_DISPLAY_NAMES[k] for k in THESIS_VARIANT_ORDER
    )


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
