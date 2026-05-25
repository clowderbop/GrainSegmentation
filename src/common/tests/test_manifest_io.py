"""Tests for manifest_io and stage_manifest."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from common.evaluate_instances import collect_manifest_samples
from common.manifest_io import (
    DatasetManifest,
    ManifestSampleRow,
    build_eval_manifest,
    build_unet_whole_manifest,
    build_yolo_whole_manifest,
    whole_manifest_overlay_anchor,
    collect_manifest_unet_samples,
    load_dataset_manifest,
    resolve_row_path,
    validate_dataset_manifest,
    write_dataset_manifest,
)
from common.stage_manifest import stage_manifest, stage_manifest_to_file
from common.variants import all_variant_names, get_variant


def _write_rgb_tiff(path: Path, *, size: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    tifffile.imwrite(path, rgb, photometric="rgb")


def _synthetic_grainseg(tmp_path: Path) -> Path:
    grainseg = tmp_path / "GrainSeg"
    spec = get_variant("PPL+AllPPX")
    root = grainseg
    for suffix in spec.unet.input_suffixes:
        _write_rgb_tiff(root / spec.paths.train_channel_template.format(suffix=suffix))
    _write_rgb_tiff(root / spec.paths.train_labels_raster)
    (root / spec.paths.train_labels_gpkg).parent.mkdir(parents=True, exist_ok=True)
    (root / spec.paths.train_labels_gpkg).write_text("", encoding="utf-8")
    return grainseg


def test_build_yolo_whole_manifest(tmp_path: Path) -> None:
    grainseg = _synthetic_grainseg(tmp_path)
    spec = get_variant("PPL+PPXblend")
    stacked = grainseg / spec.paths.test_mosaic_stacked
    stacked.parent.mkdir(parents=True, exist_ok=True)
    stacked.write_bytes(b"")
    manifest = build_yolo_whole_manifest(
        split="test", variant="PPL+PPXblend", grainseg_root=grainseg
    )
    validate_dataset_manifest(manifest)
    assert manifest.samples[0].image == spec.paths.test_mosaic_stacked


def test_whole_manifest_overlay_anchor(tmp_path: Path) -> None:
    grainseg = _synthetic_grainseg(tmp_path)
    manifest = build_unet_whole_manifest(
        split="train", variant="PPL+AllPPX", grainseg_root=grainseg
    )
    sample_id, image_rel, mask_rel = whole_manifest_overlay_anchor(manifest)
    assert sample_id == "train"
    assert image_rel.endswith("_PPL.tif")
    assert mask_rel is not None


def test_build_and_validate_unet_whole_manifest(tmp_path: Path) -> None:
    grainseg = _synthetic_grainseg(tmp_path)
    manifest = build_unet_whole_manifest(
        split="train", variant="PPL+AllPPX", grainseg_root=grainseg
    )
    validate_dataset_manifest(manifest)
    assert len(manifest.samples) == 1
    assert manifest.samples[0].images is not None
    assert len(manifest.samples[0].images) == 7


def test_load_rejects_stacked_mosaic_in_images(tmp_path: Path) -> None:
    grainseg = _synthetic_grainseg(tmp_path)
    spec = get_variant("PPL+PPXblend")
    payload = {
        "schema_version": 1,
        "variant": "PPL+PPXblend",
        "unit": "whole",
        "grainseg_root": str(grainseg),
        "path_base": "grainseg_root",
        "samples": [
            {
                "sample_id": "train",
                "images": [
                    spec.paths.train_mosaic_stacked,
                    spec.paths.train_channel_template.format(suffix="_PPXblend"),
                ],
            }
        ],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="stacked YOLO mosaic"):
        load_dataset_manifest(path)


def test_collect_manifest_unet_samples_shape(tmp_path: Path) -> None:
    grainseg = _synthetic_grainseg(tmp_path)
    manifest = build_unet_whole_manifest(
        split="train", variant="PPL+AllPPX", grainseg_root=grainseg
    )
    out_path = tmp_path / "manifest.json"
    write_dataset_manifest(out_path, manifest)
    samples = collect_manifest_unet_samples(out_path)
    assert len(samples) == 1
    assert samples[0]["id"] == "train"
    assert len(samples[0]["images"]) == 7


def test_stage_manifest_copies_channel_files(tmp_path: Path) -> None:
    grainseg = _synthetic_grainseg(tmp_path)
    manifest = build_unet_whole_manifest(
        split="train", variant="PPL+AllPPX", grainseg_root=grainseg
    )
    work = tmp_path / "work"
    staged = stage_manifest(manifest, work)
    assert staged.path_base == "work_root"
    assert len(staged.samples) == 1
    row = staged.samples[0]
    assert row.images is not None
    for name in row.images:
        assert (work / name).is_file()
    assert len(list(work.glob("*.tif"))) == 8  # 7 channels + labels mask copied


@pytest.mark.integration
def test_stage_manifest_integration_file_count(tmp_path: Path) -> None:
    """Plan acceptance: PPL+AllPPX train whole manifest stages exactly 7 channel TIFFs."""
    grainseg = _synthetic_grainseg(tmp_path)
    manifest_path = grainseg / "dataset/train/manifests/PPL+AllPPX.whole.json"
    write_dataset_manifest(
        manifest_path,
        build_unet_whole_manifest(
            split="train", variant="PPL+AllPPX", grainseg_root=grainseg
        ),
    )
    work = tmp_path / "staged"
    staged_path = stage_manifest_to_file(manifest_path, work)
    loaded = load_dataset_manifest(staged_path)
    assert loaded.variant == "PPL+AllPPX"
    channel_files = [work / name for name in loaded.samples[0].images or ()]
    assert len(channel_files) == 7
    assert all(path.is_file() for path in channel_files)


def test_build_eval_manifest_gt_gpkg_overrides_source(tmp_path: Path) -> None:
    grainseg = _synthetic_grainseg(tmp_path)
    source = build_unet_whole_manifest(
        split="train", variant="PPL+AllPPX", grainseg_root=grainseg
    )
    assert source.samples[0].gt_gpkg is not None

    pred_root = tmp_path / "run_model"
    pred_root.mkdir()
    instances_dir = pred_root / "instances"
    instances_dir.mkdir()
    tifffile.imwrite(
        instances_dir / "train_instances.tif",
        np.zeros((8, 8), dtype=np.int32),
    )

    local_gt = tmp_path / "staged" / "train_labels.gpkg"
    local_gt.parent.mkdir(parents=True)
    local_gt.write_text("", encoding="utf-8")

    eval_doc = build_eval_manifest(
        source,
        pred_instances_dir=instances_dir,
        manifest_parent=pred_root,
        gt_gpkg=local_gt,
    )
    resolved = resolve_row_path(eval_doc, eval_doc.samples[0].gt_gpkg)
    assert resolved == local_gt.resolve()


def test_collect_manifest_samples_multi_input_anchor(tmp_path: Path) -> None:
    grainseg = _synthetic_grainseg(tmp_path)
    manifest = build_unet_whole_manifest(
        split="train", variant="PPL+AllPPX", grainseg_root=grainseg
    )
    out_path = tmp_path / "m.json"
    write_dataset_manifest(out_path, manifest)

    instances_dir = tmp_path / "instances"
    instances_dir.mkdir()
    pred = instances_dir / "train_instances.tif"
    tifffile.imwrite(pred, np.zeros((8, 8), dtype=np.int32))

    samples = collect_manifest_samples(
        out_path,
        pred_instances_dir=instances_dir,
        default_gt_gpkg=grainseg / get_variant("PPL+AllPPX").paths.train_labels_gpkg,
    )
    assert len(samples) == 1
    assert samples[0].sample_id == "train"
    assert samples[0].image_path.name == "train_PPL.tif"


@pytest.mark.parametrize("variant", all_variant_names())
def test_build_whole_manifest_all_variants(tmp_path: Path, variant: str) -> None:
    grainseg = tmp_path / "GrainSeg"
    spec = get_variant(variant)
    for suffix in spec.unet.input_suffixes:
        _write_rgb_tiff(
            grainseg / spec.paths.train_channel_template.format(suffix=suffix)
        )
        _write_rgb_tiff(
            grainseg / spec.paths.test_channel_template.format(suffix=suffix)
        )
    _write_rgb_tiff(grainseg / spec.paths.train_labels_raster)
    _write_rgb_tiff(grainseg / spec.paths.test_labels_raster)
    (grainseg / spec.paths.train_labels_gpkg).write_text("", encoding="utf-8")
    (grainseg / spec.paths.test_labels_gpkg).write_text("", encoding="utf-8")

    for split in ("train", "test"):
        manifest = build_unet_whole_manifest(
            split=split,  # type: ignore[arg-type]
            variant=variant,
            grainseg_root=grainseg,
        )
        validate_dataset_manifest(manifest)
