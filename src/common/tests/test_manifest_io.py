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
    collect_manifest_tune_samples,
    collect_manifest_unet_samples,
    load_dataset_manifest,
    resolve_row_path,
    validate_dataset_manifest,
    write_dataset_manifest,
)
from common.stage_manifest import (
    stage_manifest,
    stage_manifest_metadata,
    stage_manifest_metadata_to_file,
    stage_manifest_to_file,
)
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


def test_load_rejects_obsolete_pred_instances_field(tmp_path: Path) -> None:
    payload = {
        "schema_version": 1,
        "variant": "PPL",
        "unit": "patch",
        "grainseg_root": str(tmp_path),
        "path_base": "grainseg_root",
        "samples": [
            {
                "sample_id": "patch001",
                "image": "patch001.tif",
                "pred_instances": "instances/patch001_instances.tif",
            }
        ],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="obsolete field"):
        load_dataset_manifest(path)


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


def test_stage_manifest_copies_gt_txt_preserving_tree(tmp_path: Path) -> None:
    grainseg = tmp_path / "grainseg"
    rel_txt = "dataset/test/patches/PPL/labels/test/region_0001_y00000_x00000.txt"
    rel_image = "dataset/test/patches/PPL/images/test/region_0001_y00000_x00000.tif"
    (grainseg / rel_txt).parent.mkdir(parents=True, exist_ok=True)
    (grainseg / rel_txt).write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
    _write_rgb_tiff(grainseg / rel_image)

    source = DatasetManifest(
        schema_version=1,
        variant="PPL",
        unit="patch",
        grainseg_root=str(grainseg),
        path_base="grainseg_root",
        samples=(
            ManifestSampleRow(
                sample_id="region_0001_y00000_x00000",
                image=rel_image,
                gt_txt=rel_txt,
                gt_origin="patch_stem",
            ),
        ),
    )
    work = tmp_path / "work"
    staged = stage_manifest(source, work)
    assert staged.path_base == "work_root"
    resolved_txt = resolve_row_path(staged, staged.samples[0].gt_txt)
    assert resolved_txt is not None
    assert resolved_txt.is_file()
    assert resolved_txt == (work / rel_txt).resolve()


def test_stage_manifest_metadata_writes_manifest_without_copying_rasters(
    tmp_path: Path,
) -> None:
    grainseg = _synthetic_grainseg(tmp_path)
    manifest = build_unet_whole_manifest(
        split="train", variant="PPL+AllPPX", grainseg_root=grainseg
    )
    canonical = tmp_path / "canonical.json"
    write_dataset_manifest(canonical, manifest)
    work = tmp_path / "work"
    staged_path = stage_manifest_metadata_to_file(canonical, work)
    loaded = load_dataset_manifest(staged_path)
    assert loaded.path_base == "work_root"
    assert loaded.samples[0].images is not None
    assert len(loaded.samples[0].images) == 7
    assert list(work.glob("*.tif")) == []
    tune_samples = collect_manifest_tune_samples(loaded)
    assert tune_samples[0]["id"] == loaded.samples[0].sample_id
    assert tune_samples[0]["num_channels"] == 7


def test_stage_manifest_metadata_preserves_gt_paths_without_copying(
    tmp_path: Path,
) -> None:
    """Metadata-only staging is not GT-self-contained; GT is supplied via --gt-gpkg (ADR 0002)."""
    grainseg = _synthetic_grainseg(tmp_path)
    manifest = build_unet_whole_manifest(
        split="train", variant="PPL+AllPPX", grainseg_root=grainseg
    )
    canonical_gt = manifest.samples[0].gt_gpkg
    assert canonical_gt is not None
    work = tmp_path / "work"
    metadata = stage_manifest_metadata(manifest, work)
    assert metadata.samples[0].gt_gpkg == canonical_gt
    assert not (work / canonical_gt).is_file()


def test_stage_manifest_metadata_matches_full_staging_path_metadata(tmp_path: Path) -> None:
    grainseg = _synthetic_grainseg(tmp_path)
    manifest = build_unet_whole_manifest(
        split="train", variant="PPL+AllPPX", grainseg_root=grainseg
    )
    work_meta = tmp_path / "meta"
    work_full = tmp_path / "full"
    metadata = stage_manifest_metadata(manifest, work_meta)
    full = stage_manifest(manifest, work_full)
    assert metadata.samples[0].images == full.samples[0].images
    assert metadata.samples[0].sample_id == full.samples[0].sample_id


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


def test_build_eval_manifest_materializes_staged_assets_for_eval(tmp_path: Path) -> None:
    """Cluster layout: inference staging and run output are sibling directories."""
    grainseg = _synthetic_grainseg(tmp_path)
    spec = get_variant("PPL+PPXblend")
    stacked = grainseg / spec.paths.test_mosaic_stacked
    stacked.parent.mkdir(parents=True, exist_ok=True)
    _write_rgb_tiff(stacked)
    test_gpkg = grainseg / spec.paths.test_labels_gpkg
    test_gpkg.parent.mkdir(parents=True, exist_ok=True)
    test_gpkg.write_text("", encoding="utf-8")
    source = build_yolo_whole_manifest(
        split="test", variant="PPL+PPXblend", grainseg_root=grainseg
    )
    staged_work = tmp_path / "staged"
    staged = stage_manifest(source, staged_work)
    pred_root = tmp_path / "eval" / "yolo_PPL+PPXblend"
    pred_root.mkdir(parents=True)
    from common.prediction_set import prediction_set_path, save_prediction_set

    save_prediction_set(
        prediction_set_path(pred_root, "test"),
        {
            "schema_version": 1,
            "height": 8,
            "width": 8,
            "producer": "yolo",
            "detections": [],
        },
    )
    eval_doc = build_eval_manifest(
        staged,
        prediction_set_dir=pred_root,
        manifest_parent=pred_root,
    )
    eval_path = pred_root / "eval_manifest.json"
    write_dataset_manifest(eval_path, eval_doc)

    work_base = pred_root.resolve()
    image = resolve_row_path(eval_doc, eval_doc.samples[0].image)
    gt = resolve_row_path(eval_doc, eval_doc.samples[0].gt_gpkg)
    assert image is not None and gt is not None
    assert work_base in image.parents
    assert work_base in gt.parents
    assert image.is_file()
    assert gt.is_file()

    samples = collect_manifest_samples(eval_path)
    assert samples[0].image_path == image
    assert samples[0].gt_gpkg == gt


def test_build_eval_manifest_resolves_staged_gt_gpkg(tmp_path: Path) -> None:
    grainseg = _synthetic_grainseg(tmp_path)
    spec = get_variant("PPL+PPXblend")
    stacked = grainseg / spec.paths.test_mosaic_stacked
    stacked.parent.mkdir(parents=True, exist_ok=True)
    stacked.write_bytes(b"")
    test_gpkg = grainseg / spec.paths.test_labels_gpkg
    test_gpkg.parent.mkdir(parents=True, exist_ok=True)
    test_gpkg.write_text("", encoding="utf-8")
    source = build_yolo_whole_manifest(
        split="test", variant="PPL+PPXblend", grainseg_root=grainseg
    )
    work = tmp_path / "staged"
    staged = stage_manifest(source, work)
    pred_root = tmp_path / "run"
    pred_root.mkdir()
    from common.prediction_set import prediction_set_path, save_prediction_set

    save_prediction_set(
        prediction_set_path(pred_root, "test"),
        {
            "schema_version": 1,
            "height": 8,
            "width": 8,
            "producer": "yolo",
            "detections": [],
        },
    )
    eval_doc = build_eval_manifest(
        staged,
        prediction_set_dir=pred_root,
        manifest_parent=pred_root,
    )
    resolved = resolve_row_path(eval_doc, eval_doc.samples[0].gt_gpkg)
    assert resolved is not None
    assert resolved.is_file()


def test_build_eval_manifest_rejects_scratch_gt_gpkg(tmp_path: Path) -> None:
    grainseg = _synthetic_grainseg(tmp_path)
    source = build_unet_whole_manifest(
        split="train", variant="PPL+AllPPX", grainseg_root=grainseg
    )
    pred_root = tmp_path / "run"
    pred_root.mkdir()
    scratch_gt = grainseg / get_variant("PPL+AllPPX").paths.train_labels_gpkg

    with pytest.raises(ValueError, match="staged work"):
        build_eval_manifest(
            source,
            prediction_set_dir=pred_root,
            manifest_parent=pred_root,
            gt_gpkg=scratch_gt,
        )


def test_build_eval_manifest_gt_gpkg_overrides_source(tmp_path: Path) -> None:
    grainseg = _synthetic_grainseg(tmp_path)
    source = build_unet_whole_manifest(
        split="train", variant="PPL+AllPPX", grainseg_root=grainseg
    )
    assert source.samples[0].gt_gpkg is not None

    pred_root = tmp_path / "run_model"
    pred_root.mkdir()
    from common.prediction_set import prediction_set_path, save_prediction_set

    ps_path = prediction_set_path(pred_root, "train")
    save_prediction_set(
        ps_path,
        {
            "schema_version": 1,
            "height": 8,
            "width": 8,
            "producer": "unet",
            "detections": [],
        },
    )

    local_gt = pred_root / "train_labels.gpkg"
    local_gt.write_text("", encoding="utf-8")

    eval_doc = build_eval_manifest(
        source,
        prediction_set_dir=pred_root,
        manifest_parent=pred_root,
        gt_gpkg=local_gt,
    )
    assert eval_doc.samples[0].instance_prediction_set is not None
    assert eval_doc.samples[0].instance_prediction_set.endswith(
        "prediction_sets/train.json"
    )
    resolved = resolve_row_path(eval_doc, eval_doc.samples[0].gt_gpkg)
    assert resolved == local_gt.resolve()


def test_build_eval_manifest_preserves_gt_txt(tmp_path: Path) -> None:
    grainseg = tmp_path / "grainseg"
    grainseg.mkdir()
    image_path = grainseg / "patch001.tif"
    image_path.write_bytes(b"\x00" * 64)
    gt_txt = grainseg / "labels" / "patch001.txt"
    gt_txt.parent.mkdir(parents=True)
    gt_txt.write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    source = DatasetManifest(
        schema_version=1,
        variant="PPL",
        unit="patch",
        grainseg_root=str(grainseg),
        path_base="grainseg_root",
        samples=(
            ManifestSampleRow(
                sample_id="patch001",
                image="patch001.tif",
                gt_txt="labels/patch001.txt",
                gt_origin="patch_stem",
            ),
        ),
    )

    pred_root = tmp_path / "run"
    pred_root.mkdir()
    from common.prediction_set import prediction_set_path, save_prediction_set

    save_prediction_set(
        prediction_set_path(pred_root, "patch001"),
        {
            "schema_version": 1,
            "height": 16,
            "width": 16,
            "producer": "yolo",
            "detections": [],
        },
    )

    eval_doc = build_eval_manifest(
        source,
        prediction_set_dir=pred_root,
        manifest_parent=pred_root,
    )
    row = eval_doc.samples[0]
    assert row.gt_txt is not None
    assert row.gt_origin == "patch_stem"
    assert resolve_row_path(eval_doc, row.gt_txt) == (pred_root / "labels/patch001.txt").resolve()


def test_collect_manifest_samples_multi_input_anchor(tmp_path: Path) -> None:
    grainseg = _synthetic_grainseg(tmp_path)
    manifest = build_unet_whole_manifest(
        split="train", variant="PPL+AllPPX", grainseg_root=grainseg
    )
    out_path = tmp_path / "m.json"
    write_dataset_manifest(out_path, manifest)

    from common.prediction_set import prediction_set_path, save_prediction_set

    run_dir = tmp_path / "run"
    ps_path = prediction_set_path(run_dir, "train")
    save_prediction_set(
        ps_path,
        {
            "schema_version": 1,
            "height": 8,
            "width": 8,
            "producer": "unet",
            "detections": [],
        },
    )

    samples = collect_manifest_samples(
        out_path,
        prediction_set_dir=run_dir,
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
