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


def test_load_registry_exposes_consistent_variant_metadata() -> None:
    """INTENT: load_registry loads variants with consistent channels, slugs, and thesis ordering."""
    reg = load_registry()
    assert reg.schema_version == 1
    validate(reg)
    assert registry_path().is_file()
    assert len(reg.variants) >= 1

    for name, spec in reg.variants.items():
        assert (
            spec.yolo.input_channels
            == spec.unet.channels_per_input * spec.unet.num_inputs
        )
        assert len(spec.unet.input_suffixes) == spec.unet.num_inputs
        assert spec.slugs.job
        assert spec.display_name
        assert spec.yolo.dataset_subdir
        assert spec.yolo.yaml_name.endswith(".yaml")

    names = all_variant_names()
    assert names == tuple(reg.variants)
    thesis_order = variants_in_thesis_order()
    assert set(thesis_order).issubset(set(reg.variants))
    assert variant_display_names_in_thesis_order() == tuple(
        reg.variants[name].display_name for name in thesis_order
    )


def test_resolve_paths_under_grainseg_root(tmp_path: Path) -> None:
    """INTENT: resolve_paths joins variant path templates under a grainseg root directory."""
    grainseg = tmp_path / "GrainSeg"
    spec = get_variant("PPL+AllPPX")
    resolved = spec.resolve_paths(grainseg)
    assert resolved.test_mosaic_stacked == grainseg / spec.paths.test_mosaic_stacked
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
    variant = "PPL+AllPPX"
    spec = get_variant(variant)
    proc = _run_variants_cli(
        "--grainseg-root",
        str(default_grainseg_root("/scratch/example")),
        "env",
        "--variant",
        variant,
    )
    assert f"export NUM_INPUTS={spec.unet.num_inputs}" in proc.stdout
    assert f"export YOLO_INPUT_CHANNELS={spec.yolo.input_channels}" in proc.stdout


def test_cli_print_json() -> None:
    """INTENT: the variants CLI print-json subcommand emits JSON with YOLO input channel metadata."""
    variant = "PPL"
    spec = get_variant(variant)
    proc = _run_variants_cli("print-json", "--variant", variant)
    payload = json.loads(proc.stdout)
    assert payload["yolo"]["input_channels"] == spec.yolo.input_channels
