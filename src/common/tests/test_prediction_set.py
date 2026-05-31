"""Tests for common.prediction_set (instance prediction set v1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from common.mask_ops import instance_map_from_masks
from common.prediction_set import (
    PredictionSet,
    build_yolo_prediction_set_from_ultralytics,
    load_prediction_set,
    prediction_set_path,
    prediction_set_to_merged_instance_view,
    save_prediction_set,
    segmentation_to_binary_mask,
    validate_prediction_set,
)
from common.tests.prediction_set_fixtures import yolo_prediction_set_from_masks


def test_prediction_set_save_load_round_trip(tmp_path: Path) -> None:
    data = {
        "schema_version": 1,
        "height": 8,
        "width": 8,
        "producer": "yolo",
        "detections": [
            {
                "segmentation": {
                    "size": [8, 8],
                    "counts": "0",
                },
                "score": 0.9,
                "category_id": 0,
            }
        ],
    }
    path = prediction_set_path(tmp_path, "patch001")
    save_prediction_set(path, data)
    loaded = load_prediction_set(path)
    assert loaded.schema_version == 1
    assert loaded.producer == "yolo"
    assert len(loaded.detections) == 1
    assert loaded.detections[0]["score"] == pytest.approx(0.9)


def test_validate_rejects_unet_detection_with_score() -> None:
    with pytest.raises(ValueError, match="score"):
        validate_prediction_set(
            {
                "schema_version": 1,
                "height": 4,
                "width": 4,
                "producer": "unet",
                "detections": [
                    {
                        "segmentation": {"size": [4, 4], "counts": "0"},
                        "score": 0.5,
                        "category_id": 0,
                    }
                ],
            }
        )


def test_validate_rejects_yolo_detection_without_score() -> None:
    with pytest.raises(ValueError, match="requires score"):
        validate_prediction_set(
            {
                "schema_version": 1,
                "height": 4,
                "width": 4,
                "producer": "yolo",
                "detections": [
                    {
                        "segmentation": {"size": [4, 4], "counts": "0"},
                        "category_id": 0,
                    }
                ],
            }
        )


def test_yolo_merged_instance_view_decodes_one_mask_at_a_time() -> None:
    masks = np.zeros((4, 8, 8), dtype=np.float32)
    masks[0, 1:4, 1:4] = 1.0
    masks[1, 2:5, 2:5] = 1.0
    masks[2, 3:6, 3:6] = 1.0
    masks[3, 4:7, 4:7] = 1.0
    scores = np.array([0.1, 0.4, 0.7, 0.9], dtype=np.float32)
    prediction_set = yolo_prediction_set_from_masks(
        masks_hw=masks,
        scores=scores,
        height=8,
        width=8,
    )

    live_decodes = 0
    peak_decodes = 0

    def counting_decode(segmentation: dict) -> np.ndarray:
        nonlocal live_decodes, peak_decodes
        live_decodes += 1
        peak_decodes = max(peak_decodes, live_decodes)
        try:
            return segmentation_to_binary_mask(segmentation)
        finally:
            live_decodes -= 1

    with patch("common.prediction_set.segmentation_to_binary_mask", counting_decode):
        with patch("numpy.stack", side_effect=AssertionError("must not stack all masks")):
            merged = prediction_set_to_merged_instance_view(prediction_set)

    expected = instance_map_from_masks(masks, scores, height=8, width=8)
    np.testing.assert_array_equal(merged, expected)
    assert peak_decodes == 1


def test_merged_instance_view_matches_score_painting() -> None:
    masks = np.zeros((2, 8, 8), dtype=np.float32)
    masks[0, 2:6, 2:6] = 1.0
    masks[1, 2:6, 2:6] = 1.0
    scores = np.array([0.2, 0.9], dtype=np.float32)
    expected = instance_map_from_masks(masks, scores, height=8, width=8)

    ps = yolo_prediction_set_from_masks(
        masks_hw=masks,
        scores=scores,
        height=8,
        width=8,
    )
    merged = prediction_set_to_merged_instance_view(ps)
    np.testing.assert_array_equal(merged, expected)


def test_prediction_set_path_layout() -> None:
    root = Path("/run/out")
    assert prediction_set_path(root, "abc") == root / "prediction_sets" / "abc.json"


class _PerMaskTensorStack:
    """Tensor stack that forbids bulk ``.cpu().numpy()`` (index-only access)."""

    def __init__(self, planes: list[torch.Tensor]) -> None:
        self._planes = planes

    def __len__(self) -> int:
        return len(self._planes)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._planes[index]

    def cpu(self) -> None:
        raise AssertionError("must not materialize full mask stack")


def test_build_yolo_prediction_set_from_ultralytics_per_mask() -> None:
    height, width = 8, 8
    plane = torch.zeros((height, width), dtype=torch.float32)
    plane[2:6, 2:6] = 1.0
    masks = MagicMock()
    masks.__len__.return_value = 1
    masks.data = _PerMaskTensorStack([plane])
    result = MagicMock()
    result.masks = masks
    result.boxes.conf = torch.tensor([0.75], dtype=torch.float32)

    from_ultralytics = build_yolo_prediction_set_from_ultralytics(
        result, height=height, width=width
    )
    expected = yolo_prediction_set_from_masks(
        masks_hw=plane.numpy()[None, ...],
        scores=np.array([0.75], dtype=np.float32),
        height=height,
        width=width,
    )
    assert len(from_ultralytics.detections) == len(expected.detections) == 1
    assert from_ultralytics.detections[0]["score"] == pytest.approx(0.75)
    np.testing.assert_array_equal(
        prediction_set_to_merged_instance_view(from_ultralytics),
        prediction_set_to_merged_instance_view(expected),
    )
