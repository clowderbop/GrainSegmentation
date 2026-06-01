"""Tests for YOLO profile tune detector cache CLI."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from common.test_inference import load_test_inference_recipe
from yolo.profile_tune_detector import write_detector_proposal_cache
from yolo.tests.profile_tune_fixtures import disjoint_tile_local_proposals
from yolo.tiled_proposal_cache import (
    detector_cache_expected_record,
    load_tiled_proposals,
    proposal_cache_dir,
    proposal_cache_record,
    recipe_whole_window_fingerprint,
    tiled_proposal_record_from_binary_mask,
    weights_sha256,
    write_tiled_proposals,
)


@pytest.fixture
def detector_tune_layout(tmp_path: Path) -> dict[str, Path]:
    grainseg = tmp_path / "GrainSeg"
    run_root = tmp_path / "runs" / "yolo26-seg"
    variant = "PPL"
    weights = run_root / variant / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights-bytes")

    work_root = tmp_path / "tune_out" / "_work"
    staged_manifest = work_root / variant / "staged" / "manifest.json"
    staged_manifest.parent.mkdir(parents=True)
    staged_manifest.write_text(
        '{"samples": [{"id": "train", "image": "train.tif"}]}',
        encoding="utf-8",
    )

    return {
        "output_dir": tmp_path / "tune_out",
        "grainseg_root": grainseg,
        "run_root": run_root,
        "work_root": work_root,
        "repo": tmp_path / "repo",
        "variant": variant,
        "weights": weights,
    }


def test_write_detector_proposal_cache_skips_sliced_detection_when_cache_valid(
    detector_tune_layout: dict[str, Path],
) -> None:
    layout = detector_tune_layout
    conf, mask_threshold = 0.25, 0.5
    cache_dir = proposal_cache_dir(
        layout["work_root"] / layout["variant"], conf=conf, mask_threshold=mask_threshold
    )
    recipe = load_test_inference_recipe()
    height, width = 16, 16
    mask = np.zeros((height, width), dtype=bool)
    mask[2:6, 2:6] = True
    record = proposal_cache_record(
        variant=layout["variant"],
        weights_sha256=weights_sha256(layout["weights"]),
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=conf,
        mask_threshold=mask_threshold,
        sample_id="train",
        height=height,
        width=width,
    )
    write_tiled_proposals(
        cache_dir, [tiled_proposal_record_from_binary_mask(mask, score=0.5)], record
    )

    detect_calls: list[int] = []

    def fake_sliced_detection(*_args, **_kwargs) -> list[dict]:
        detect_calls.append(1)
        return [{"computed": True}]

    common_kwargs = dict(
        variant=layout["variant"],
        conf=conf,
        mask_threshold=mask_threshold,
        output_dir=layout["output_dir"],
        grainseg_root=layout["grainseg_root"],
        run_root=layout["run_root"],
        work_root=layout["work_root"],
        device="0",
        repo=layout["repo"],
    )

    with (
        patch(
            "yolo.profile_tune_detector.ensure_staged_train_manifest",
            return_value=layout["work_root"] / layout["variant"] / "staged" / "manifest.json",
        ),
        patch(
            "yolo.profile_tune_detector.collect_manifest_image_paths",
            return_value=[(Path("/fake/train.tif"), "train")],
        ),
        patch(
            "yolo.profile_tune_detector.collect_tiled_detector_proposals",
            side_effect=fake_sliced_detection,
        ),
    ):
        write_detector_proposal_cache(**common_kwargs)

    assert detect_calls == []


def test_write_detector_proposal_cache_persists_crop_local_masks_on_disk(
    detector_tune_layout: dict[str, Path],
) -> None:
    """Detector write path stores v2 crop-local RLE on disk, not full-section dense masks."""
    layout = detector_tune_layout
    conf, mask_threshold = 0.2, 0.45
    height, width = 64, 64
    image = np.zeros((height, width, 3), dtype=np.uint8)
    tile_preds = disjoint_tile_local_proposals(height, width)

    def fake_iter(_img, _model, *, full_shape, **_kwargs):
        assert full_shape is None
        yield 0, 0, width, height, tile_preds

    common_kwargs = dict(
        variant=layout["variant"],
        conf=conf,
        mask_threshold=mask_threshold,
        output_dir=layout["output_dir"],
        grainseg_root=layout["grainseg_root"],
        run_root=layout["run_root"],
        work_root=layout["work_root"],
        device="0",
        repo=layout["repo"],
    )
    train_tif = layout["work_root"] / "train.tif"
    train_tif.write_bytes(b"\x00")

    with (
        patch(
            "yolo.profile_tune_detector.ensure_staged_train_manifest",
            return_value=layout["work_root"] / layout["variant"] / "staged" / "manifest.json",
        ),
        patch(
            "yolo.profile_tune_detector.collect_manifest_image_paths",
            return_value=[(train_tif, "train")],
        ),
        patch(
            "yolo.profile_tune_detector.load_image_for_yolo",
            return_value=image,
        ),
        patch(
            "yolo.profile_tune_detector.AutoDetectionModel.from_pretrained",
            return_value=MagicMock(),
        ),
        patch(
            "yolo.sliced_detection.iter_whole_slice_predictions",
            side_effect=fake_iter,
        ),
    ):
        cache_dir = write_detector_proposal_cache(**common_kwargs)

    expected = detector_cache_expected_record(
        variant=layout["variant"],
        weights_path=layout["weights"],
        conf=conf,
        mask_threshold=mask_threshold,
        sample_id="train",
    )
    loaded, meta = load_tiled_proposals(cache_dir, expected=expected)
    assert meta["schema_version"] == 2
    assert len(loaded) == 2
    for record in loaded:
        crop_h, crop_w = record["segmentation"]["size"]
        assert crop_h < height
        assert crop_w < width
        assert crop_h * crop_w < height * width

    with (cache_dir / "proposals.pkl").open("rb") as handle:
        on_disk = pickle.load(handle)
    assert all(isinstance(entry, dict) for entry in on_disk)
    assert not any(isinstance(entry, np.ndarray) for entry in on_disk)
