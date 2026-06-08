"""Whole predict writes cross-tile canonical instance prediction sets."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from common.prediction_set import (
    assert_yolo_grains_non_overlapping,
    load_prediction_set,
)
from common.run_provenance import load_run_provenance
from common.tests.prediction_set_fixtures import assert_yolo_canonical_sets_equal
from yolo import predict as predict_module
from yolo.cross_tile_postprocess import prediction_set_from_tiled_proposal_records
from yolo.tests.profile_tune_fixtures import tiled_proposal_records_disjoint_via_collector


def _whole_predict_args(
    tmp_path: Path,
    *,
    out_dir: Path | None = None,
) -> Namespace:
    weights = tmp_path / "best.pt"
    weights.write_text("", encoding="utf-8")
    return Namespace(
        weights=weights,
        output_dir=out_dir or (tmp_path / "out"),
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
    """INTENT: whole predict persists a cross-tile-associated canonical prediction set."""
    height, width = 16, 16
    image_path = tmp_path / "test_PPL.tif"
    image_path.write_bytes(b"\x00" * 64)
    mock_pairs.return_value = [(image_path, "test")]
    mock_load_image.return_value = np.zeros((height, width, 3), dtype=np.uint8)
    records = tiled_proposal_records_disjoint_via_collector(height, width, mask_threshold=0.5)
    mock_collect.return_value = records
    mock_from_pretrained.return_value = MagicMock()

    out_dir = tmp_path / "out"
    args = _whole_predict_args(tmp_path, out_dir=out_dir)
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
    """INTENT: whole predict run provenance records cross-tile association at predict time."""
    image_path = tmp_path / "test_PPL.tif"
    image_path.write_bytes(b"\x00" * 64)
    out_dir = tmp_path / "out"
    mock_from_pretrained.return_value = MagicMock()
    args = _whole_predict_args(tmp_path, out_dir=out_dir)
    args.image = image_path
    predict_module.run_whole_predict(args)
    provenance = load_run_provenance(out_dir)
    assert provenance.get("cross_tile_association_at_predict") is True


@patch("yolo.predict._load_whole_predict_pairs")
@patch("yolo.predict.load_image_for_yolo")
@patch("yolo.predict.collect_tiled_detector_proposals")
@patch("sahi.AutoDetectionModel.from_pretrained")
def test_whole_predict_calls_scaled_associate_tiled_proposals(
    mock_from_pretrained: MagicMock,
    mock_collect: MagicMock,
    mock_load_image: MagicMock,
    mock_pairs: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """INTENT: whole predict fuses tiled proposals through scaled cross-tile association."""
    import yolo.cross_tile_postprocess as postprocess
    from yolo.cross_tile_association import associate_tiled_proposals as real_associate

    calls: list[tuple[int, int, int]] = []

    def spy(proposals, *, height: int, width: int, log_timings: bool = False):
        calls.append((len(proposals), height, width))
        return real_associate(
            proposals, height=height, width=width, log_timings=log_timings
        )

    monkeypatch.setattr(postprocess, "associate_tiled_proposals", spy)

    height, width = 16, 16
    image_path = tmp_path / "test_PPL.tif"
    image_path.write_bytes(b"\x00" * 64)
    mock_pairs.return_value = [(image_path, "test")]
    mock_load_image.return_value = np.zeros((height, width, 3), dtype=np.uint8)
    records = tiled_proposal_records_disjoint_via_collector(height, width, mask_threshold=0.5)
    mock_collect.return_value = records
    mock_from_pretrained.return_value = MagicMock()

    out_dir = tmp_path / "out"
    predict_module.run_whole_predict(_whole_predict_args(tmp_path, out_dir=out_dir))

    assert calls == [(len(records), height, width)]

    loaded = load_prediction_set(out_dir / "prediction_sets" / "test.json")
    assert loaded.producer == "yolo"
    assert all("score" in det for det in loaded.detections)
    assert_yolo_grains_non_overlapping(loaded)


@patch("yolo.predict._load_whole_predict_pairs")
@patch("yolo.predict.load_image_for_yolo")
@patch("yolo.predict.collect_tiled_detector_proposals")
@patch("sahi.AutoDetectionModel.from_pretrained")
def test_whole_predict_does_not_use_score_merge(
    mock_from_pretrained: MagicMock,
    mock_collect: MagicMock,
    mock_load_image: MagicMock,
    mock_pairs: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """INTENT: whole predict never invokes score-merge proposal fusion."""
    def fail_score_merge(*_args: object, **_kwargs: object):
        raise AssertionError(
            "whole predict must not call merge_yolo_proposals_by_score"
        )

    monkeypatch.setattr(predict_module, "merge_yolo_proposals_by_score", fail_score_merge)

    height, width = 16, 16
    image_path = tmp_path / "test_PPL.tif"
    image_path.write_bytes(b"\x00" * 64)
    mock_pairs.return_value = [(image_path, "test")]
    mock_load_image.return_value = np.zeros((height, width, 3), dtype=np.uint8)
    records = tiled_proposal_records_disjoint_via_collector(height, width, mask_threshold=0.5)
    mock_collect.return_value = records
    mock_from_pretrained.return_value = MagicMock()

    predict_module.run_whole_predict(_whole_predict_args(tmp_path))


@patch("yolo.predict._load_whole_predict_pairs")
@patch("yolo.predict.load_image_for_yolo")
@patch("yolo.predict.collect_tiled_detector_proposals")
@patch("sahi.AutoDetectionModel.from_pretrained")
def test_whole_predict_logs_sliding_window_and_association_phases(
    mock_from_pretrained: MagicMock,
    mock_collect: MagicMock,
    mock_load_image: MagicMock,
    mock_pairs: MagicMock,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """INTENT: whole predict logs sliding-window and cross-tile association phase progress."""
    from yolo.tests.phase_logging_assertions import (
        assert_done_timing_lines,
        assert_substrings_in_order,
    )

    height, width = 16, 16
    image_path = tmp_path / "test_PPL.tif"
    image_path.write_bytes(b"\x00" * 64)
    mock_pairs.return_value = [(image_path, "test")]
    mock_load_image.return_value = np.zeros((height, width, 3), dtype=np.uint8)
    records = tiled_proposal_records_disjoint_via_collector(height, width, mask_threshold=0.5)
    mock_collect.return_value = records
    mock_from_pretrained.return_value = MagicMock()

    predict_module.run_whole_predict(_whole_predict_args(tmp_path))
    out = capsys.readouterr().out
    assert_substrings_in_order(
        out,
        "Predicting tiles …",
        "Predicting tiles done",
        "Merging predictions …",
        "Cross-tile association done",
    )
    assert_done_timing_lines(out, min_count=2)
