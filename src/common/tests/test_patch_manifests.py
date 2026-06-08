"""Tests for YOLO / U-Net patch manifest builders."""

from __future__ import annotations

from pathlib import Path


from common.manifest_io import load_dataset_manifest
from common.patch_manifests import (
    build_unet_patch_manifest,
    build_yolo_patch_manifest,
    render_yolo_dataset_yaml,
    write_yolo_dataset_yaml_file,
)
from common.variants import get_variant


def _write_patch(
    root: Path,
    split_name: str,
    stem: str,
    *,
    with_label: bool = True,
) -> None:
    image_dir = root / "images" / split_name
    label_dir = root / "labels" / split_name
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    (image_dir / f"{stem}.tif").write_bytes(b"")
    if with_label:
        (label_dir / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")


def test_build_yolo_patch_manifest_train(tmp_path: Path) -> None:
    """INTENT: build_yolo_patch_manifest indexes train and val patch images with gt_txt and whole-section gpkg."""
    grainseg = tmp_path
    patch_root = grainseg / "dataset/train/patches/PPL"
    _write_patch(patch_root, "train", "region_0001_y00000_x00000")
    _write_patch(patch_root, "val", "region_0002_y00000_x00000")

    manifest = build_yolo_patch_manifest(
        variant="PPL",
        split="train",
        grainseg_root=grainseg,
    )
    assert manifest.unit == "patches"
    assert manifest.path_base == "grainseg_root"
    assert len(manifest.samples) == 2
    row = manifest.samples[0]
    assert (
        row.image
        == "dataset/train/patches/PPL/images/train/region_0001_y00000_x00000.tif"
    )
    assert (
        row.gt_txt
        == "dataset/train/patches/PPL/labels/train/region_0001_y00000_x00000.txt"
    )
    assert row.gt_gpkg == "dataset/train/train_labels.gpkg"
    assert row.gt_origin == "patch_stem"


def test_build_yolo_patch_manifest_test_split(tmp_path: Path) -> None:
    """INTENT: build_yolo_patch_manifest for test split points gt_gpkg at the test whole-section labels."""
    grainseg = tmp_path
    patch_root = grainseg / "dataset/test/patches/PPL"
    _write_patch(patch_root, "test", "region_0001_y00000_x00000")

    manifest = build_yolo_patch_manifest(
        variant="PPL",
        split="test",
        grainseg_root=grainseg,
        patch_root=patch_root,
    )
    assert len(manifest.samples) == 1
    assert manifest.samples[0].gt_gpkg == "dataset/test/test_labels.gpkg"


def test_build_unet_patch_manifest_single_input(tmp_path: Path) -> None:
    """INTENT: build_unet_patch_manifest maps YOLO patch rows to single-channel U-Net image and mask paths."""
    yolo = build_yolo_patch_manifest(
        variant="PPL",
        split="test",
        grainseg_root=tmp_path,
        patch_root=_make_yolo_tree(tmp_path, "test"),
    )
    unet = build_unet_patch_manifest(
        variant="PPL",
        split="test",
        grainseg_root=tmp_path,
        yolo_manifest=yolo,
        unet_root_rel="dataset/test/unet_from_yolo/PPL",
    )
    row = unet.samples[0]
    assert (
        row.image
        == "dataset/test/unet_from_yolo/PPL/images/region_0001_y00000_x00000_PPL.tif"
    )
    assert (
        row.mask
        == "dataset/test/unet_from_yolo/PPL/masks/region_0001_y00000_x00000_labels.tif"
    )
    assert row.gt_origin == "patch_stem"


def test_build_unet_patch_manifest_multi_input() -> None:
    """INTENT: build_unet_patch_manifest expands multi-input variants to one image path per channel."""
    from common.manifest_io import DatasetManifest, ManifestSampleRow

    yolo = DatasetManifest(
        schema_version=1,
        variant="PPL+AllPPX",
        unit="patches",
        grainseg_root="/scratch/GrainSeg",
        path_base="grainseg_root",
        samples=(
            ManifestSampleRow(
                sample_id="region_0001_y00000_x00000",
                image="dataset/test/patches/PPL+AllPPX/images/test/region_0001_y00000_x00000.tif",
                gt_gpkg="dataset/test/test_labels.gpkg",
                gt_origin="patch_stem",
            ),
        ),
    )
    unet = build_unet_patch_manifest(
        variant="PPL+AllPPX",
        split="test",
        grainseg_root="/scratch/GrainSeg",
        yolo_manifest=yolo,
        unet_root_rel="dataset/test/unet_from_yolo/PPL+AllPPX",
    )
    spec = get_variant("PPL+AllPPX")
    row = unet.samples[0]
    assert row.images is not None
    assert len(row.images) == spec.unet.num_inputs
    assert row.images[0].endswith("region_0001_y00000_x00000_PPL.tif")


def test_render_yolo_dataset_yaml_channels() -> None:
    """INTENT: render_yolo_dataset_yaml includes channels only for multi-channel variants when held out."""
    yaml = render_yolo_dataset_yaml("PPL+AllPPX", held_out=True)
    assert "channels: 21" in yaml
    assert "images/test" in yaml
    assert "channels:" not in render_yolo_dataset_yaml("PPL", held_out=False)


def test_write_yolo_dataset_yaml_file(tmp_path: Path) -> None:
    """INTENT: write_yolo_dataset_yaml_file writes a variant-named YAML without channels for single-input PPL."""
    patch_root = tmp_path / "PPL"
    patch_root.mkdir()
    path = write_yolo_dataset_yaml_file("PPL", patch_root, held_out=False)
    assert path.name == "PPL.yaml"
    assert "channels:" not in path.read_text()


def _make_yolo_tree(grainseg: Path, split_name: str) -> Path:
    patch_root = grainseg / "dataset/test/patches/PPL"
    _write_patch(patch_root, split_name, "region_0001_y00000_x00000")
    return patch_root


def test_collect_manifest_unet_samples_single_image_field(tmp_path: Path) -> None:
    """INTENT: collect_manifest_unet_samples reads single-image U-Net patch manifests into tune sample dicts."""
    from common.manifest_io import collect_manifest_unet_samples, write_dataset_manifest

    unet = build_unet_patch_manifest(
        variant="PPL",
        split="test",
        grainseg_root=tmp_path,
        yolo_manifest=build_yolo_patch_manifest(
            variant="PPL",
            split="test",
            grainseg_root=tmp_path,
            patch_root=_make_yolo_tree(tmp_path, "test"),
        ),
        unet_root_rel="dataset/test/unet_from_yolo/PPL",
    )
    manifest_path = tmp_path / "manifest.json"
    write_dataset_manifest(manifest_path, unet)
    doc = load_dataset_manifest(manifest_path)
    assert doc.samples[0].image is not None

    images_dir = tmp_path / "dataset/test/unet_from_yolo/PPL/images"
    images_dir.mkdir(parents=True, exist_ok=True)
    stem = "region_0001_y00000_x00000"
    (images_dir / f"{stem}_PPL.tif").write_bytes(b"")
    masks_dir = tmp_path / "dataset/test/unet_from_yolo/PPL/masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    (masks_dir / f"{stem}_labels.tif").write_bytes(b"")

    samples = collect_manifest_unet_samples(manifest_path)
    assert len(samples) == 1
    assert len(samples[0]["images"]) == 1
