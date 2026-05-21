"""Tests for SAHI prediction visualization export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from common.instance_predictions import instance_map_path, write_instance_map_tiff
from yolo.export_sahi_visualization import (
    export_sample_visualization,
    write_mask_overlay_visual,
)


def test_write_mask_overlay_visual(tmp_path: Path) -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    pred_map = np.zeros((16, 16), dtype=np.int32)
    pred_map[2:8, 2:8] = 1
    out_path = tmp_path / "prediction_visual.tif"
    write_mask_overlay_visual(image, pred_map, out_path)
    assert out_path.is_file()
    loaded = tifffile.imread(out_path)
    assert loaded.shape == (16, 16, 3)


def test_export_sample_visualization_from_instance_map(tmp_path: Path) -> None:
    width, height = 32, 32
    image_path = tmp_path / "sample.tif"
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    tifffile.imwrite(image_path, rgb, photometric="rgb")

    pred_map = np.zeros((height, width), dtype=np.int32)
    pred_map[4:20, 4:20] = 1
    pred_path = instance_map_path(tmp_path, "sample")
    write_instance_map_tiff(pred_path, pred_map)

    sample_out = tmp_path / "out" / "sample"
    export_sample_visualization(
        image_path=image_path,
        pred_instances_path=pred_path,
        sample_out_dir=sample_out,
    )
    assert (sample_out / "prediction_visual.tif").is_file()
