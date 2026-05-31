"""Tests for shared whole-section sliced detection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from yolo.sliced_detection import (
    merge_sliced_object_predictions,
    run_whole_sliced_detection,
)


def test_run_whole_sliced_detection_skips_slice_merge_postprocess() -> None:
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    detection_model = MagicMock()
    pred_a = MagicMock()
    pred_a.get_shifted_object_prediction.return_value = "pred_a"
    pred_b = MagicMock()
    pred_b.get_shifted_object_prediction.return_value = "pred_b"
    detection_model.object_prediction_list = [pred_a, pred_b]

    mock_postprocess_cls = MagicMock()

    with (
        patch(
            "sahi.slicing.get_slice_bboxes",
            return_value=[(0, 0, 16, 16), (8, 8, 24, 24)],
        ),
        patch("sahi.predict.filter_predictions", side_effect=lambda preds, **_: preds),
        patch(
            "yolo.sliced_detection.perform_ultralytics_inference_preserve_channels"
        ),
        patch.dict(
            "sahi.predict.POSTPROCESS_NAME_TO_CLASS",
            {"NMM": mock_postprocess_cls},
            clear=False,
        ),
    ):
        proposals = run_whole_sliced_detection(
            image,
            detection_model,
            slice_height=16,
            slice_width=16,
            overlap_height_ratio=0.5,
            overlap_width_ratio=0.5,
        )

    assert proposals == ["pred_a", "pred_b", "pred_a", "pred_b"]
    mock_postprocess_cls.assert_not_called()


def test_merge_sliced_object_predictions_applies_postprocess() -> None:
    mock_postprocess = MagicMock(side_effect=lambda preds: preds[:1])
    mock_postprocess_cls = MagicMock(return_value=mock_postprocess)
    with patch.dict(
        "sahi.predict.POSTPROCESS_NAME_TO_CLASS",
        {"NMM": mock_postprocess_cls},
        clear=False,
    ):
        merged = merge_sliced_object_predictions(
            ["a", "b"],
            postprocess_type="NMM",
            match_metric="IOU",
            match_threshold=0.6,
        )
    mock_postprocess_cls.assert_called_once_with(
        match_threshold=0.6,
        match_metric="IOU",
        class_agnostic=False,
    )
    mock_postprocess.assert_called_once_with(["a", "b"])
    assert merged == ["a"]
