"""Phase 0: lightweight checks that config/variants.yaml matches known contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
VARIANTS_PATH = REPO_ROOT / "config" / "variants.yaml"

EXPECTED_VARIANTS = ("PPL", "PPLPPXblend", "PPL+PPXblend", "PPL+AllPPX")
EXPECTED_YOLO_CHANNELS = {
    "PPL": 3,
    "PPLPPXblend": 3,
    "PPL+PPXblend": 6,
    "PPL+AllPPX": 21,
}
EXPECTED_UNET_NUM_INPUTS = {
    "PPL": 1,
    "PPLPPXblend": 1,
    "PPL+PPXblend": 2,
    "PPL+AllPPX": 7,
}
EXPECTED_WATERSHED_JOB_SLUGS = {
    "PPL": "PPL",
    "PPLPPXblend": "PPLPPXblend",
    "PPL+PPXblend": "PPL_PlusPPXblend",
    "PPL+AllPPX": "PPL_AllPPX",
}


@pytest.fixture(scope="module")
def registry() -> dict:
    assert VARIANTS_PATH.is_file(), f"Missing registry: {VARIANTS_PATH}"
    with VARIANTS_PATH.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    assert isinstance(data, dict)
    return data


def test_schema_version(registry: dict) -> None:
    assert registry["schema_version"] == 1


def test_all_four_variants_present(registry: dict) -> None:
    variants = registry["variants"]
    assert set(variants) == set(EXPECTED_VARIANTS)


@pytest.mark.parametrize("name", EXPECTED_VARIANTS)
def test_yolo_input_channels_match_readme(registry: dict, name: str) -> None:
    spec = registry["variants"][name]
    assert spec["yolo"]["input_channels"] == EXPECTED_YOLO_CHANNELS[name]
    assert spec["unet"]["channels_per_input"] == 3


@pytest.mark.parametrize("name", EXPECTED_VARIANTS)
def test_unet_num_inputs(registry: dict, name: str) -> None:
    unet = registry["variants"][name]["unet"]
    assert unet["num_inputs"] == EXPECTED_UNET_NUM_INPUTS[name]
    assert len(unet["input_suffixes"]) == unet["num_inputs"]


@pytest.mark.parametrize("name", EXPECTED_VARIANTS)
def test_watershed_job_slug_matches_variants_sh(registry: dict, name: str) -> None:
    assert registry["variants"][name]["slugs"]["job"] == EXPECTED_WATERSHED_JOB_SLUGS[name]


def test_ppl_plus_ppx_yaml_name_differs_from_subdir(registry: dict) -> None:
    yolo = registry["variants"]["PPL+PPXblend"]["yolo"]
    assert yolo["dataset_subdir"] == "PPL+PPXblend"
    assert yolo["yaml_name"] == "PPL_PPXblend.yaml"


def test_stacked_mosaic_paths_use_literal_variant(registry: dict) -> None:
    spec = registry["variants"]["PPL+AllPPX"]["paths"]
    assert spec["train_mosaic_stacked"] == "dataset/train/train_PPL+AllPPX.tif"
    assert spec["test_mosaic_stacked"] == "dataset/test/test_PPL+AllPPX.tif"
