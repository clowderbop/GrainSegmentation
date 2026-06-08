"""Whole-image YOLO predict uses manifest sample id for prediction set paths."""

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
from yolo.predict import _load_whole_predict_pairs


def _write_rgb_tiff(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(path, np.zeros((4, 4, 3), dtype=np.uint8), photometric="rgb")


def test_load_whole_predict_pairs_uses_manifest_sample_id(tmp_path: Path) -> None:
    """INTENT: whole predict pair loading uses manifest sample_id for output naming."""
    work = tmp_path / "staged"
    image_name = "test_PPL.tif"
    _write_rgb_tiff(work / image_name)
    manifest_path = work / "manifest.json"
    write_dataset_manifest(
        manifest_path,
        DatasetManifest(
            schema_version=1,
            variant="PPL",
            unit="whole",
            grainseg_root=str(work),
            path_base="work_root",
            samples=(
                ManifestSampleRow(
                    sample_id="test",
                    image=image_name,
                    gt_origin="whole_image",
                ),
            ),
        ),
    )

    pairs = _load_whole_predict_pairs(
        argparse.Namespace(manifest=manifest_path, image=None)
    )
    assert pairs == [(work / image_name, "test")]
