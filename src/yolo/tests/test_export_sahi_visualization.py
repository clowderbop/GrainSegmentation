"""Tests for SAHI prediction visualization export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from common.tests.prediction_set_fixtures import yolo_prediction_set_from_masks
from common.prediction_set import merge_yolo_proposals_by_score, save_prediction_set
from yolo.export_sahi_visualization import (
    export_sample_visualization_from_prediction_set,
    write_mask_overlay_visual,
    write_prediction_set_overlay_visual,
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


def test_write_prediction_set_overlay_on_canonical_grain(tmp_path: Path) -> None:
    height, width = 16, 16
    masks = np.zeros((2, height, width), dtype=np.float32)
    masks[0, 4:12, 4:12] = 1.0
    masks[1, 4:12, 4:12] = 1.0
    proposals = yolo_prediction_set_from_masks(
        masks_hw=masks,
        scores=np.array([0.2, 0.9], dtype=np.float32),
        height=height,
        width=width,
    )
    prediction_set = merge_yolo_proposals_by_score(proposals)
    image = np.zeros((height, width, 3), dtype=np.uint8)
    out_path = tmp_path / "overlay.tif"
    write_prediction_set_overlay_visual(image, prediction_set, out_path)
    visual = tifffile.imread(out_path)
    assert visual[8, 8].any()


def test_export_sample_visualization_from_prediction_set(tmp_path: Path) -> None:
    height, width = 32, 32
    image_path = tmp_path / "sample.tif"
    tifffile.imwrite(
        image_path, np.zeros((height, width, 3), dtype=np.uint8), photometric="rgb"
    )
    masks = np.zeros((1, height, width), dtype=np.float32)
    masks[0, 4:20, 4:20] = 1.0
    ps_path = tmp_path / "prediction_sets" / "sample.json"
    save_prediction_set(
        ps_path,
        yolo_prediction_set_from_masks(
            masks_hw=masks,
            scores=np.array([0.8], dtype=np.float32),
            height=height,
            width=width,
        ),
    )
    sample_out = tmp_path / "out" / "sample"
    export_sample_visualization_from_prediction_set(
        image_path=image_path,
        prediction_set_path=ps_path,
        sample_out_dir=sample_out,
    )
    assert (sample_out / "prediction_visual.tif").is_file()
