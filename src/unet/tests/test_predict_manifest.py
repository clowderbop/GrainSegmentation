"""Tests for manifest-driven unet.predict."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from common.manifest_io import build_unet_whole_manifest, write_dataset_manifest
from unet.predict import _build_arg_parser, _resolve_predict_samples


def _write_rgb(path: Path, size: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    tifffile.imwrite(path, rgb, photometric="rgb")


def _grainseg_with_ppl(tmp_path: Path) -> Path:
    grainseg = tmp_path / "GrainSeg"
    spec_path = grainseg / "dataset/train/train_PPL.tif"
    _write_rgb(spec_path)
    (grainseg / "dataset/train/train_labels.tif").write_bytes(b"")
    (grainseg / "dataset/train/train_labels.gpkg").write_text("", encoding="utf-8")
    return grainseg


def test_resolve_predict_samples_from_manifest(tmp_path: Path) -> None:
    grainseg = _grainseg_with_ppl(tmp_path)
    manifest = build_unet_whole_manifest(
        split="train", variant="PPL", grainseg_root=grainseg
    )
    manifest_path = tmp_path / "manifest.json"
    write_dataset_manifest(manifest_path, manifest)

    parser = _build_arg_parser()
    args = parser.parse_args(
        ["--model-path", "m.keras", "--output-dir", str(tmp_path / "out"), "--manifest", str(manifest_path)]
    )
    samples = _resolve_predict_samples(args)
    assert len(samples) == 1
    assert samples[0]["id"] == "train"
    assert len(samples[0]["images"]) == 1
    assert args.num_inputs == 1


def test_manifest_is_required() -> None:
    parser = _build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model-path", "m.keras", "--output-dir", "out"])
