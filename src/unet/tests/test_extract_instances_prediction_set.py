"""extract_instances writes instance prediction sets and run provenance."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile

from common.manifest_io import (
    DatasetManifest,
    ManifestSampleRow,
    write_dataset_manifest,
)
from common.prediction_set import load_prediction_set, prediction_set_path
from common.run_provenance import RUN_PROVENANCE_FILENAME, load_run_provenance
from unet.extract_instances import run_extract_instances


def _write_manifest(path: Path, sample_id: str) -> None:
    write_dataset_manifest(
        path,
        DatasetManifest(
            schema_version=1,
            variant="PPL",
            unit="patch",
            grainseg_root=str(path.parent),
            path_base="grainseg_root",
            samples=(
                ManifestSampleRow(
                    sample_id=sample_id,
                    image="dummy.tif",
                ),
            ),
        ),
    )


def test_extract_instances_writes_prediction_set_and_provenance(tmp_path: Path) -> None:
    """INTENT: CC extract_instances writes prediction sets and run provenance without legacy instance dirs."""
    sample_id = "patch001"
    semantic_dir = tmp_path / "semantic"
    semantic_dir.mkdir()
    semantic = np.zeros((12, 12), dtype=np.int32)
    semantic[2:6, 2:6] = 1
    tifffile.imwrite(semantic_dir / f"{sample_id}_pred.tif", semantic)

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, sample_id)

    output_dir = tmp_path / "out"
    args = argparse.Namespace(
        semantic_dir=semantic_dir,
        output_dir=output_dir,
        manifest=manifest_path,
        instance_method="cc",
        watershed_min_distance=1,
        watershed_boundary_dilate_iter=0,
        watershed_connectivity=1,
        watershed_min_area_px=0,
        watershed_exclude_border=False,
        watershed_ridge_level=None,
        min_area_px=0,
    )
    run_extract_instances(args)

    ps_path = prediction_set_path(output_dir, sample_id)
    assert ps_path.is_file()
    prediction_set = load_prediction_set(ps_path)
    assert prediction_set.producer == "unet"
    assert len(prediction_set.detections) == 1

    provenance = load_run_provenance(output_dir)
    assert provenance["producer"] == "unet"
    assert provenance["instance_method"] == "cc"
    assert provenance["min_area_px"] == 0
    assert not (output_dir / "instances").exists()
    assert not (output_dir / "instances" / ".extract_meta.json").exists()
    assert (output_dir / RUN_PROVENANCE_FILENAME).is_file()


def test_extract_instances_watershed_writes_prediction_set_and_provenance(
    tmp_path: Path,
) -> None:
    """INTENT: watershed extract_instances writes prediction sets and records watershed params in provenance."""
    sample_id = "patch001"
    semantic_dir = tmp_path / "semantic"
    semantic_dir.mkdir()
    semantic = np.zeros((32, 32), dtype=np.int32)
    semantic[8:20, 8:20] = 1
    semantic[22:30, 22:30] = 1
    tifffile.imwrite(semantic_dir / f"{sample_id}_pred.tif", semantic)

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, sample_id)

    output_dir = tmp_path / "out"
    args = argparse.Namespace(
        semantic_dir=semantic_dir,
        output_dir=output_dir,
        manifest=manifest_path,
        instance_method="watershed",
        watershed_min_distance=5,
        watershed_boundary_dilate_iter=0,
        watershed_connectivity=1,
        watershed_min_area_px=0,
        watershed_exclude_border=False,
        watershed_ridge_level=None,
        min_area_px=0,
    )
    run_extract_instances(args)

    ps_path = prediction_set_path(output_dir, sample_id)
    assert ps_path.is_file()
    prediction_set = load_prediction_set(ps_path)
    assert prediction_set.producer == "unet"
    assert len(prediction_set.detections) >= 2

    provenance = load_run_provenance(output_dir)
    assert provenance["producer"] == "unet"
    assert provenance["instance_method"] == "watershed"
    assert provenance["watershed_min_distance"] == 5
    assert provenance["watershed_min_area_px"] == 0


def test_extract_instances_cc_records_nonzero_min_area_px_in_provenance(
    tmp_path: Path,
) -> None:
    """INTENT: CC extract_instances records tuned min_area_px in run provenance."""
    sample_id = "patch001"
    semantic_dir = tmp_path / "semantic"
    semantic_dir.mkdir()
    semantic = np.zeros((20, 20), dtype=np.int32)
    semantic[2:12, 2:12] = 1
    semantic[14:16, 14:16] = 1
    tifffile.imwrite(semantic_dir / f"{sample_id}_pred.tif", semantic)

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, sample_id)

    output_dir = tmp_path / "out"
    args = argparse.Namespace(
        semantic_dir=semantic_dir,
        output_dir=output_dir,
        manifest=manifest_path,
        instance_method="cc",
        watershed_min_distance=1,
        watershed_boundary_dilate_iter=0,
        watershed_connectivity=1,
        watershed_min_area_px=0,
        watershed_exclude_border=False,
        watershed_ridge_level=None,
        min_area_px=64,
    )
    run_extract_instances(args)

    provenance = load_run_provenance(output_dir)
    assert provenance["instance_method"] == "cc"
    assert provenance["min_area_px"] == 64

    prediction_set = load_prediction_set(prediction_set_path(output_dir, sample_id))
    assert len(prediction_set.detections) == 1  # speckle below min_area_px is dropped
