"""Tests for YOLO segmentation label I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from common.yolo_seg_labels import (
    read_yolo_seg_gt_label_rows,
    read_yolo_seg_pred_label_rows,
)


def test_read_gt_labels_without_confidence(tmp_path: Path) -> None:
    width, height = 100, 80
    expected = np.array(
        [[10.0, 10.0], [50.0, 10.0], [30.0, 40.0]], dtype=np.float32
    )
    label_path = tmp_path / "sample.txt"
    label_path.write_text(
        "0 0.1 0.125 0.5 0.125 0.3 0.5\n",
        encoding="utf-8",
    )

    loaded = read_yolo_seg_gt_label_rows(
        label_path, image_width=width, image_height=height
    )
    assert len(loaded) == 1
    assert loaded[0].class_id == 0
    np.testing.assert_allclose(loaded[0].points, expected, rtol=1e-4, atol=1e-3)


def test_read_pred_labels_with_confidence(tmp_path: Path) -> None:
    width, height = 64, 64
    expected = np.array(
        [[5.0, 5.0], [20.0, 5.0], [12.0, 18.0]], dtype=np.float32
    )
    label_path = tmp_path / "pred.txt"
    label_path.write_text(
        "0 0.078125 0.078125 0.3125 0.078125 0.1875 0.28125 0.87\n",
        encoding="utf-8",
    )

    loaded = read_yolo_seg_pred_label_rows(
        label_path, image_width=width, image_height=height
    )
    assert len(loaded) == 1
    assert loaded[0].class_id == 0
    np.testing.assert_allclose(loaded[0].points, expected, rtol=1e-4, atol=1e-3)
    assert abs(loaded[0].confidence - 0.87) < 1e-5
