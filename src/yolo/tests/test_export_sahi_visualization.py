"""Tests for SAHI prediction visualization export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from common.yolo_seg_labels import YoloSegPredRow, write_yolo_seg_pred_label_file
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


def test_export_sample_visualization_from_pred_txt(tmp_path: Path) -> None:
    width, height = 32, 32
    image_path = tmp_path / "sample.tif"
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    tifffile.imwrite(image_path, rgb, photometric="rgb")

    rows = [
        YoloSegPredRow(
            class_id=0,
            points=np.array([[4.0, 4.0], [20.0, 4.0], [12.0, 18.0]], dtype=np.float32),
            confidence=0.9,
        )
    ]
    label_path = tmp_path / "labels" / "sample.txt"
    label_path.parent.mkdir()
    write_yolo_seg_pred_label_file(
        label_path, rows, image_width=width, image_height=height
    )

    sample_out = tmp_path / "out" / "sample"
    export_sample_visualization(
        image_path=image_path,
        pred_label_path=label_path,
        sample_out_dir=sample_out,
    )
    assert (sample_out / "prediction_visual.tif").is_file()
    assert not (sample_out / "predicted_masks.gpkg").exists()
