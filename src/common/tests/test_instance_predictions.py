"""Tests for common.instance_predictions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from common.instance_predictions import (
    instance_map_from_masks,
    read_instance_map_tiff,
    write_instance_map_tiff,
    yolo_mask_npz_to_coco_dt,
)


def test_instance_map_round_trip(tmp_path: Path) -> None:
    label_map = np.zeros((32, 32), dtype=np.int32)
    label_map[4:12, 4:12] = 1
    label_map[18:26, 10:22] = 2
    path = tmp_path / "sample_instances.tif"
    write_instance_map_tiff(path, label_map)
    loaded = read_instance_map_tiff(path)
    np.testing.assert_array_equal(loaded, label_map)


def test_instance_map_from_masks_respects_confidence_order() -> None:
    masks = np.zeros((2, 8, 8), dtype=np.float32)
    masks[0, 2:6, 2:6] = 1.0
    masks[1, 2:6, 2:6] = 1.0
    low_conf = instance_map_from_masks(
        masks, np.array([0.2, 0.9]), height=8, width=8
    )
    assert int(low_conf[3, 3]) == 2


def test_yolo_mask_npz_to_coco_dt(tmp_path: Path) -> None:
    masks = np.zeros((1, 16, 16), dtype=np.float32)
    masks[0, 4:10, 4:10] = 1.0
    npz_path = tmp_path / "sample.npz"
    np.savez_compressed(
        npz_path,
        masks=masks,
        scores=np.array([0.87], dtype=np.float32),
        classes=np.array([0.0], dtype=np.float32),
        orig_shape=np.array([16, 16]),
        image_path=np.array("sample.tif"),
    )
    dt = yolo_mask_npz_to_coco_dt(npz_path, image_id=1, height=16, width=16)
    assert len(dt) == 1
    assert float(dt[0]["score"]) == pytest.approx(0.87)
    assert dt[0]["category_id"] == 0
