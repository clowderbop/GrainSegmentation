"""Tests for YOLO segmentation label I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.yolo_seg_labels import (
    YoloSegGtRow,
    YoloSegPredRow,
    instance_label_map_to_yolo_seg_pred_label_file,
    read_yolo_seg_gt_label_rows,
    read_yolo_seg_pred_label_rows,
    write_yolo_seg_gt_label_file,
    write_yolo_seg_pred_label_file,
    yolo_seg_pred_labels_to_coco_dt,
)


def test_gt_round_trip_no_confidence(tmp_path: Path) -> None:
    width, height = 100, 80
    rows = [
        YoloSegGtRow(
            class_id=0,
            points=np.array([[10.0, 10.0], [50.0, 10.0], [30.0, 40.0]], dtype=np.float32),
        )
    ]
    label_path = tmp_path / "sample.txt"
    write_yolo_seg_gt_label_file(
        label_path, rows, image_width=width, image_height=height
    )
    text = label_path.read_text(encoding="utf-8").strip()
    parts = text.split()
    assert len(parts) == 7
    assert (len(parts) - 1) % 2 == 0

    loaded = read_yolo_seg_gt_label_rows(
        label_path, image_width=width, image_height=height
    )
    assert len(loaded) == 1
    assert loaded[0].class_id == 0
    np.testing.assert_allclose(loaded[0].points, rows[0].points, rtol=1e-4, atol=1e-3)


def test_pred_round_trip_with_confidence(tmp_path: Path) -> None:
    width, height = 64, 64
    rows = [
        YoloSegPredRow(
            class_id=0,
            points=np.array([[5.0, 5.0], [20.0, 5.0], [12.0, 18.0]], dtype=np.float32),
            confidence=0.87,
        )
    ]
    label_path = tmp_path / "pred.txt"
    write_yolo_seg_pred_label_file(
        label_path, rows, image_width=width, image_height=height
    )
    text = label_path.read_text(encoding="utf-8").strip()
    assert text.endswith("0.87")

    loaded = read_yolo_seg_pred_label_rows(
        label_path, image_width=width, image_height=height
    )
    assert len(loaded) == 1
    assert abs(loaded[0].confidence - 0.87) < 1e-5


def test_pred_txt_to_coco_dt_scores(tmp_path: Path) -> None:
    width, height = 40, 40
    rows = [
        YoloSegPredRow(
            class_id=0,
            points=np.array([[2.0, 2.0], [18.0, 2.0], [10.0, 16.0]], dtype=np.float32),
            confidence=0.42,
        )
    ]
    label_path = tmp_path / "pred.txt"
    write_yolo_seg_pred_label_file(
        label_path, rows, image_width=width, image_height=height
    )
    dt = yolo_seg_pred_labels_to_coco_dt(
        label_path, width=width, height=height, image_id=1
    )
    assert len(dt) == 1
    assert abs(float(dt[0]["score"]) - 0.42) < 1e-6


def test_instance_map_writes_confidence_suffix(tmp_path: Path) -> None:
    width, height = 32, 32
    instance_map = np.zeros((height, width), dtype=np.int32)
    instance_map[5:15, 5:15] = 1
    instance_map[18:28, 10:25] = 2
    label_path = tmp_path / "instances.txt"
    instance_label_map_to_yolo_seg_pred_label_file(
        instance_map,
        label_path,
        default_confidence=1.0,
        min_area_px=1,
    )
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        assert float(parts[-1]) == 1.0
