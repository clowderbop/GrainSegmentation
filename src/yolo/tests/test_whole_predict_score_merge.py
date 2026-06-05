"""Whole predict writes cross-tile canonical instance prediction sets."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from common.prediction_set import load_prediction_set
from common.run_provenance import load_run_provenance
from common.tests.prediction_set_fixtures import assert_yolo_canonical_sets_equal
from yolo import predict as predict_module
from yolo.cross_tile_postprocess import prediction_set_from_tiled_proposal_records
from yolo.tests.profile_tune_fixtures import tiled_proposal_records_disjoint_via_collector


@patch("yolo.predict._load_whole_predict_pairs")
@patch("yolo.predict.load_image_for_yolo")
@patch("yolo.predict.collect_tiled_detector_proposals")
@patch("sahi.AutoDetectionModel.from_pretrained")
def test_whole_predict_writes_cross_tile_canonical_set(
    mock_from_pretrained: MagicMock,
    mock_collect: MagicMock,
    mock_load_image: MagicMock,
    mock_pairs: MagicMock,
    tmp_path: Path,
) -> None:
    height, width = 16, 16
    image_path = tmp_path / "test_PPL.tif"
    image_path.write_bytes(b"\x00" * 64)
    mock_pairs.return_value = [(image_path, "test")]
    mock_load_image.return_value = np.zeros((height, width, 3), dtype=np.uint8)
    records = tiled_proposal_records_disjoint_via_collector(height, width, mask_threshold=0.5)
    mock_collect.return_value = records
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
        mask_threshold=0.5,
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
    expected = prediction_set_from_tiled_proposal_records(
        records, height=height, width=width
    )
    assert_yolo_canonical_sets_equal(loaded, expected)


@patch("yolo.predict._load_whole_predict_pairs", return_value=[])
@patch("sahi.AutoDetectionModel.from_pretrained")
def test_whole_run_provenance_records_cross_tile_association(
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
        mask_threshold=0.5,
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
    assert provenance.get("cross_tile_association_at_predict") is True
