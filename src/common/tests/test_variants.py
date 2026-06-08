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


def test_load_registry_exposes_expected_variant_metadata() -> None:
    """INTENT: load_registry loads four variants with thesis ordering, channels, slugs, and display names."""
    reg = load_registry()
    assert reg.schema_version == 1
    assert tuple(reg.variants) == EXPECTED_VARIANTS
    validate(reg)
    assert registry_path().is_file()

    for name in EXPECTED_VARIANTS:
        spec = get_variant(name)
        assert spec.yolo.input_channels == EXPECTED_YOLO_CHANNELS[name]
        assert spec.unet.channels_per_input == 3
        assert len(spec.unet.input_suffixes) == spec.unet.num_inputs
        assert spec.slugs.job == EXPECTED_WATERSHED_JOB_SLUGS[name]
        assert spec.display_name == EXPECTED_DISPLAY_NAMES[name]

    assert all_variant_names() == EXPECTED_VARIANTS
    assert variants_in_thesis_order() == THESIS_VARIANT_ORDER
    assert variant_display_names_in_thesis_order() == tuple(
        EXPECTED_DISPLAY_NAMES[k] for k in THESIS_VARIANT_ORDER
    )

    ppl_ppx = get_variant("PPL+PPXblend")
    assert ppl_ppx.yolo.dataset_subdir == "PPL+PPXblend"
    assert ppl_ppx.yolo.yaml_name == "PPL_PPXblend.yaml"
    all_ppx_paths = get_variant("PPL+AllPPX").paths
    assert all_ppx_paths.train_mosaic_stacked == "dataset/train/train_PPL+AllPPX.tif"


def test_resolve_paths_under_grainseg_root(tmp_path: Path) -> None:
    """INTENT: resolve_paths joins variant path templates under a grainseg root directory."""
    grainseg = tmp_path / "GrainSeg"
    spec = get_variant("PPL+AllPPX")
    resolved = spec.resolve_paths(grainseg)
    assert resolved.test_mosaic_stacked == grainseg / "dataset/test/test_PPL+AllPPX.tif"
    assert (
        resolved.train_channel_path("_PPX3")
        == grainseg / "dataset/train/train_PPX3.tif"
    )


def test_get_variant_unknown_raises() -> None:
    """INTENT: get_variant raises ValueError for unregistered variant names."""
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
    """INTENT: the variants CLI env subcommand exports NUM_INPUTS and YOLO channel shell variables."""
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
    """INTENT: the variants CLI print-json subcommand emits JSON with YOLO input channel metadata."""
    proc = _run_variants_cli("print-json", "--variant", "PPL")
    payload = json.loads(proc.stdout)
    assert payload["yolo"]["input_channels"] == 3
