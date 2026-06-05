"""Tests for tiled detector proposal cache (ADR 0005)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from common.test_inference import load_test_inference_recipe
from yolo.tests.profile_tune_fixtures import (
    disjoint_sahi_proposals,
    write_on_disk_v1_proposal_cache,
)
from common.tests.profile_tune_fixtures import FakeBbox, FakeSahiPrediction
from common.tests.test_prediction_set_sahi import _FakeCategory, _FakeMask, _FakeScore
from yolo.tests.profile_tune_fixtures import disjoint_tile_local_proposals
from yolo.tiled_proposal_cache import (
    TILED_PROPOSAL_CACHE_SCHEMA_VERSION,
    collect_tiled_detector_proposals,
    load_or_write_tiled_proposals,
    load_tiled_proposals,
    proposal_cache_dir,
    proposal_cache_fingerprint,
    proposal_cache_record,
    recipe_whole_window_fingerprint,
    sahi_predictions_from_tiled_proposal_records,
    tiled_proposal_record_from_binary_mask,
    tiled_proposal_record_from_tile_mask,
    tiled_proposal_records_from_tile_predictions,
    validate_tiled_proposal_cache,
    weights_sha256,
    write_tiled_proposals,
)


def test_weights_sha256_matches_file_bytes(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    expected = hashlib.sha256(b"fake-weights").hexdigest()
    assert weights_sha256(weights) == expected


def test_tiled_proposal_cache_schema_version_is_three() -> None:
    assert TILED_PROPOSAL_CACHE_SCHEMA_VERSION == 3


def test_proposal_cache_dir_uses_conf_only_path(tmp_path: Path) -> None:
    cache_dir = proposal_cache_dir(tmp_path / "PPL", conf=0.25)
    assert cache_dir == tmp_path / "PPL" / "tiled_proposals" / "c0.25"
    assert "t0." not in cache_dir.name


def test_proposal_cache_record_includes_fingerprint_fields() -> None:
    recipe = load_test_inference_recipe()
    record = proposal_cache_record(
        variant="PPL",
        weights_sha256="abc",
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=0.25,
        mask_threshold=0.5,
        sample_id="train",
        height=100,
        width=200,
    )
    assert record["schema_version"] == TILED_PROPOSAL_CACHE_SCHEMA_VERSION
    assert record["variant"] == "PPL"
    assert record["sample_id"] == "train"
    assert record["conf"] == 0.25
    assert record["mask_threshold"] == 0.5
    assert record["height"] == 100
    assert record["width"] == 200


def test_load_tiled_proposals_rejects_on_disk_v1_sahi_pickle_cache(
    tmp_path: Path,
) -> None:
    """v1 layout on disk (dense SAHI pickle + schema_version 1 meta) must not load."""
    recipe = load_test_inference_recipe()
    height, width = 32, 32
    cache_dir = proposal_cache_dir(tmp_path / "PPL", conf=0.25, mask_threshold=0.5)
    v1_meta = proposal_cache_record(
        variant="PPL",
        weights_sha256="a",
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=0.25,
        mask_threshold=0.5,
        sample_id="train",
        height=height,
        width=width,
    )
    write_on_disk_v1_proposal_cache(cache_dir, meta=v1_meta)
    expected = proposal_cache_fingerprint(
        variant="PPL",
        weights_sha256="a",
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=0.25,
        mask_threshold=0.5,
        sample_id="train",
    )
    with pytest.raises(ValueError, match="schema_version 1"):
        load_tiled_proposals(cache_dir, expected=expected)


def test_validate_tiled_proposal_cache_rejects_schema_version_two() -> None:
    recipe = load_test_inference_recipe()
    expected = proposal_cache_record(
        variant="PPL",
        weights_sha256="a",
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=0.25,
        mask_threshold=0.5,
        sample_id="train",
        height=16,
        width=16,
    )
    v2_meta = dict(expected)
    v2_meta["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version 2"):
        validate_tiled_proposal_cache(v2_meta, expected)


def test_validate_tiled_proposal_cache_rejects_schema_version_one() -> None:
    recipe = load_test_inference_recipe()
    expected = proposal_cache_record(
        variant="PPL",
        weights_sha256="a",
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=0.25,
        mask_threshold=0.5,
        sample_id="train",
        height=16,
        width=16,
    )
    v1_meta = dict(expected)
    v1_meta["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version 1"):
        validate_tiled_proposal_cache(v1_meta, expected)


def test_write_and_load_tiled_proposals_round_trip(tmp_path: Path) -> None:
    recipe = load_test_inference_recipe()
    height, width = 16, 16
    record = proposal_cache_record(
        variant="PPL",
        weights_sha256="deadbeef",
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=0.2,
        mask_threshold=0.45,
        sample_id="train",
        height=height,
        width=width,
    )
    cache_dir = proposal_cache_dir(tmp_path / "PPL", conf=0.2, mask_threshold=0.45)
    mask = np.zeros((height, width), dtype=bool)
    mask[4:10, 4:10] = True
    proposals = [
        tiled_proposal_record_from_binary_mask(
            mask,
            score=0.9,
            tile_y0=0,
            tile_x0=0,
            tile_y1=height,
            tile_x1=width,
        )
    ]
    write_tiled_proposals(cache_dir, proposals, record)
    loaded, meta = load_tiled_proposals(cache_dir, expected=record)
    assert loaded == proposals
    assert loaded[0]["tile_y0"] == 0
    assert loaded[0]["tile_y1"] == height
    assert meta["variant"] == "PPL"
    assert meta["height"] == height
    assert meta["width"] == width
    crop = loaded[0]
    assert crop["score"] == 0.9
    assert crop["offset_y"] == 4
    assert crop["offset_x"] == 4
    assert crop["segmentation"]["size"] == [6, 6]
    assert crop["bbox"] == [4.0, 4.0, 10.0, 10.0]


def test_load_or_write_tiled_proposals_reuses_valid_cache_without_compute(
    tmp_path: Path,
) -> None:
    recipe = load_test_inference_recipe()
    record = proposal_cache_record(
        variant="PPL",
        weights_sha256="deadbeef",
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=0.2,
        mask_threshold=0.45,
        sample_id="train",
        height=16,
        width=16,
    )
    cache_dir = proposal_cache_dir(tmp_path / "PPL", conf=0.2, mask_threshold=0.45)
    mask = np.zeros((16, 16), dtype=bool)
    mask[2:6, 2:6] = True
    cached = [tiled_proposal_record_from_binary_mask(mask, score=0.9)]
    write_tiled_proposals(cache_dir, cached, record)
    compute_calls: list[int] = []

    def compute() -> list[dict]:
        compute_calls.append(1)
        return [{"id": 99}]

    loaded, from_cache = load_or_write_tiled_proposals(
        cache_dir, expected=record, compute_fn=compute
    )
    assert loaded == cached
    assert from_cache is True
    assert compute_calls == []


def test_load_or_write_tiled_proposals_computes_when_cache_missing(
    tmp_path: Path,
) -> None:
    recipe = load_test_inference_recipe()
    record = proposal_cache_record(
        variant="PPL",
        weights_sha256="deadbeef",
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=0.2,
        mask_threshold=0.45,
        sample_id="train",
        height=16,
        width=16,
    )
    cache_dir = proposal_cache_dir(tmp_path / "PPL", conf=0.2, mask_threshold=0.45)
    mask = np.zeros((16, 16), dtype=bool)
    mask[8:12, 8:12] = True
    fresh = [tiled_proposal_record_from_binary_mask(mask, score=0.7)]

    loaded, from_cache = load_or_write_tiled_proposals(
        cache_dir,
        expected=record,
        compute_fn=lambda: fresh,
    )
    assert loaded == fresh
    assert from_cache is False
    on_disk, _meta = load_tiled_proposals(cache_dir, expected=record)
    assert on_disk == fresh


def test_load_or_write_tiled_proposals_recomputes_on_fingerprint_mismatch(
    tmp_path: Path,
) -> None:
    recipe = load_test_inference_recipe()
    record = proposal_cache_record(
        variant="PPL",
        weights_sha256="deadbeef",
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=0.2,
        mask_threshold=0.45,
        sample_id="train",
        height=16,
        width=16,
    )
    cache_dir = proposal_cache_dir(tmp_path / "PPL", conf=0.2, mask_threshold=0.45)
    stale = dict(record)
    stale["conf"] = 0.99
    mask = np.zeros((16, 16), dtype=bool)
    mask[0:4, 0:4] = True
    write_tiled_proposals(cache_dir, [tiled_proposal_record_from_binary_mask(mask, score=0.1)], stale)
    mask2 = np.zeros((16, 16), dtype=bool)
    mask2[12:16, 12:16] = True
    updated = [tiled_proposal_record_from_binary_mask(mask2, score=0.5)]

    loaded, from_cache = load_or_write_tiled_proposals(
        cache_dir,
        expected=record,
        compute_fn=lambda: updated,
    )
    assert loaded == updated
    assert from_cache is False


def test_collect_tiled_detector_proposals_never_allocates_whole_section_plane() -> None:
    """ADR 0005: encode must not materialize (H,W) masks per proposal."""
    section_h, section_w = 10_000, 52_000
    slice_h, slice_w = 64, 64
    tile_mask = np.zeros((slice_h, slice_w), dtype=bool)
    tile_mask[8:20, 8:20] = True
    pred = FakeSahiPrediction(
        mask=_FakeMask(bool_mask=tile_mask),
        score=_FakeScore(value=0.5),
        category=_FakeCategory(id=0),
        bbox=FakeBbox(8.0, 8.0, 20.0, 20.0),
    )
    resize_calls: list[tuple[int, int]] = []
    original_zeros = np.zeros

    def guarded_zeros(shape, *args, **kwargs):
        if len(shape) >= 2 and (int(shape[0]), int(shape[1])) == (section_h, section_w):
            raise AssertionError("encode must not allocate whole-section plane")
        return original_zeros(shape, *args, **kwargs)

    def tracked_resize(mask, height, width):
        resize_calls.append((int(height), int(width)))
        if (height, width) == (section_h, section_w):
            raise AssertionError("encode must not resize masks to whole-section shape")
        from common.mask_ops import resize_mask_nearest

        return resize_mask_nearest(mask, height, width)

    image = np.zeros((section_h, section_w, 3), dtype=np.uint8)

    def fake_iter(_image, _model, *, full_shape, **_kwargs):
        assert full_shape is None
        yield 0, 0, slice_w, slice_h, [pred]

    with (
        patch("yolo.sliced_detection.iter_whole_slice_predictions", side_effect=fake_iter),
        patch("numpy.zeros", side_effect=guarded_zeros),
        patch("common.mask_ops.resize_mask_nearest", side_effect=tracked_resize),
    ):
        records = collect_tiled_detector_proposals(
            image,
            MagicMock(),
            slice_height=slice_h,
            slice_width=slice_w,
            overlap_height_ratio=0.25,
            overlap_width_ratio=0.25,
            mask_threshold=0.5,
        )

    assert len(records) == 1
    crop_h, crop_w = records[0]["segmentation"]["size"]
    assert crop_h < section_h and crop_w < section_w
    assert not any(h == section_h and w == section_w for h, w in resize_calls)


def test_tile_encode_rejects_whole_section_bool_mask() -> None:
    section_h, section_w = 10_000, 52_000
    slice_h, slice_w = 64, 64
    huge = np.zeros((section_h, section_w), dtype=bool)
    huge[0:4, 0:4] = True
    pred = FakeSahiPrediction(
        mask=_FakeMask(bool_mask=huge),
        score=_FakeScore(value=0.5),
        category=_FakeCategory(id=0),
    )
    with pytest.raises(ValueError, match="exceeds slice"):
        tiled_proposal_records_from_tile_predictions(
            [pred],
            tlx=0,
            tly=0,
            tile_y1=slice_h,
            tile_x1=slice_w,
            slice_height=slice_h,
            slice_width=slice_w,
            mask_threshold=0.5,
        )


def test_tiled_proposal_records_from_tile_predictions_crop_local() -> None:
    height, width = 16, 16
    sahi_preds = disjoint_tile_local_proposals(height, width)
    records = tiled_proposal_records_from_tile_predictions(
        sahi_preds,
        tlx=0,
        tly=0,
        tile_y1=height,
        tile_x1=width,
        slice_height=height,
        slice_width=width,
        mask_threshold=0.5,
    )
    assert len(records) == 2
    for record in records:
        assert record["segmentation"]["size"][0] < height
        assert record["segmentation"]["size"][1] < width
        assert record["tile_y0"] == 0
        assert record["tile_y1"] == height


def test_load_tiled_proposals_rejects_v2_records_without_tile_bounds(
    tmp_path: Path,
) -> None:
    """On-disk v2 proposal records (no tile_y0…) must not load under schema 3."""
    recipe = load_test_inference_recipe()
    height, width = 16, 16
    record = proposal_cache_record(
        variant="PPL",
        weights_sha256="deadbeef",
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=0.2,
        mask_threshold=0.45,
        sample_id="train",
        height=height,
        width=width,
    )
    cache_dir = proposal_cache_dir(tmp_path / "PPL", conf=0.2, mask_threshold=0.45)
    mask = np.zeros((height, width), dtype=bool)
    mask[4:10, 4:10] = True
    stale_v2 = {
        "score": 0.9,
        "bbox": [4.0, 4.0, 10.0, 10.0],
        "segmentation": tiled_proposal_record_from_binary_mask(mask, score=0.9)[
            "segmentation"
        ],
        "offset_y": 4,
        "offset_x": 4,
    }
    import json
    import pickle

    cache_dir.mkdir(parents=True, exist_ok=True)
    with (cache_dir / "proposals.pkl").open("wb") as handle:
        pickle.dump([stale_v2], handle, protocol=pickle.HIGHEST_PROTOCOL)
    (cache_dir / "proposals.meta.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="tile bounds"):
        load_tiled_proposals(cache_dir, expected=record)


def test_sahi_predictions_from_tiled_records_use_crop_local_masks() -> None:
    """ADR 0005: scoring adapter must not materialize full-section bool_mask planes."""
    section_h, section_w = 10_000, 52_000
    crop = np.zeros((12, 12), dtype=bool)
    crop[2:10, 2:10] = True
    record = tiled_proposal_record_from_tile_mask(
        crop,
        score=0.6,
        offset_y=100,
        offset_x=200,
        tile_y0=90,
        tile_x0=190,
        tile_y1=120,
        tile_x1=220,
    )
    original_zeros = np.zeros

    def guarded_zeros(shape, *args, **kwargs):
        if len(shape) >= 2 and (int(shape[0]), int(shape[1])) == (section_h, section_w):
            raise AssertionError("adapter must not allocate whole-section plane")
        return original_zeros(shape, *args, **kwargs)

    with patch("numpy.zeros", side_effect=guarded_zeros):
        predictions = sahi_predictions_from_tiled_proposal_records(
            [record], height=section_h, width=section_w
        )

    assert len(predictions) == 1
    mask = predictions[0].mask.bool_mask
    assert mask.shape == crop.shape
    assert np.array_equal(mask, crop)
    assert predictions[0].bbox.to_xyxy() == record["bbox"]


def test_validate_tiled_proposal_cache_rejects_fingerprint_mismatch() -> None:
    recipe = load_test_inference_recipe()
    expected = proposal_cache_record(
        variant="PPL",
        weights_sha256="a",
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=0.25,
        mask_threshold=0.5,
        sample_id="train",
        height=16,
        width=16,
    )
    stale = dict(expected)
    stale["conf"] = 0.99
    with pytest.raises(ValueError, match="conf"):
        validate_tiled_proposal_cache(stale, expected)
