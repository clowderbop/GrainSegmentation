"""Whole SAHI predict writes score-merged canonical instance prediction sets."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from common.prediction_set import load_prediction_set, merge_yolo_proposals_by_score
from common.run_provenance import load_run_provenance
from common.tests.prediction_set_fixtures import (
    assert_yolo_canonical_sets_equal,
    yolo_prediction_set_from_masks,
)
from common.tests.test_prediction_set_sahi import (
    _FakeCategory,
    _FakeMask,
    _FakeSahiPrediction,
    _FakeScore,
)
from yolo import predict as predict_module


@patch("yolo.predict._load_whole_predict_pairs")
@patch("yolo.predict.load_image_for_yolo")
@patch("yolo.predict._get_sliced_prediction_preserve_channels")
@patch("sahi.AutoDetectionModel.from_pretrained")
def test_whole_predict_writes_score_merged_canonical_set(
    mock_from_pretrained: MagicMock,
    mock_sliced: MagicMock,
    mock_load_image: MagicMock,
    mock_pairs: MagicMock,
    tmp_path: Path,
) -> None:
    height, width = 16, 16
    image_path = tmp_path / "test_PPL.tif"
    image_path.write_bytes(b"\x00" * 64)
    mock_pairs.return_value = [(image_path, "test")]
    mock_load_image.return_value = np.zeros((height, width, 3), dtype=np.uint8)

    masks = np.zeros((2, height, width), dtype=np.float32)
    masks[0, 4:12, 4:12] = 1.0
    masks[1, 4:12, 4:12] = 1.0
    predictions = [
        _FakeSahiPrediction(
            mask=_FakeMask(bool_mask=masks[0].astype(bool)),
            score=_FakeScore(value=0.2),
            category=_FakeCategory(id=0),
        ),
        _FakeSahiPrediction(
            mask=_FakeMask(bool_mask=masks[1].astype(bool)),
            score=_FakeScore(value=0.9),
            category=_FakeCategory(id=0),
        ),
    ]
    mock_result = MagicMock()
    mock_result.object_prediction_list = predictions
    mock_sliced.return_value = mock_result
    mock_from_pretrained.return_value = MagicMock()

    weights = tmp_path / "best.pt"
    weights.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"
    args = Namespace(
        weights=weights,
        output_dir=out_dir,
        variant="PPL",
        imgsz=1024,
        conf=0.25,
        device="cpu",
        slice_height=1024,
        slice_width=1024,
        overlap_height_ratio=0.5,
        overlap_width_ratio=0.5,
        manifest=None,
        image=None,
    )
    predict_module.run_whole_predict(args)

    loaded = load_prediction_set(out_dir / "prediction_sets" / "test.json")
    proposals = yolo_prediction_set_from_masks(
        masks_hw=masks,
        scores=np.array([0.2, 0.9], dtype=np.float32),
        height=height,
        width=width,
    )
    expected = merge_yolo_proposals_by_score(proposals)
    assert_yolo_canonical_sets_equal(loaded, expected)


@patch("yolo.predict._load_whole_predict_pairs", return_value=[])
@patch("sahi.AutoDetectionModel.from_pretrained")
def test_whole_run_provenance_records_score_merge_at_predict(
    mock_from_pretrained: MagicMock,
    mock_pairs: MagicMock,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "test_PPL.tif"
    image_path.write_bytes(b"\x00" * 64)
    weights = tmp_path / "best.pt"
    weights.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"
    mock_from_pretrained.return_value = MagicMock()
    args = Namespace(
        weights=weights,
        output_dir=out_dir,
        variant="PPL",
        imgsz=1024,
        conf=0.25,
        device="cpu",
        slice_height=1024,
        slice_width=1024,
        overlap_height_ratio=0.5,
        overlap_width_ratio=0.5,
        manifest=None,
        image=image_path,
    )
    predict_module.run_whole_predict(args)
    provenance = load_run_provenance(out_dir)
    assert provenance.get("score_merge_at_predict") is True
