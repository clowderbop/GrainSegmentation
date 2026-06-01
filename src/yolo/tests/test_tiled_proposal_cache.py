"""Tests for tiled detector proposal cache (ADR 0005)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from common.test_inference import load_test_inference_recipe
from yolo.tests.profile_tune_fixtures import (
    disjoint_sahi_proposals,
    write_on_disk_v1_proposal_cache,
)
from yolo.tiled_proposal_cache import (
    TILED_PROPOSAL_CACHE_SCHEMA_VERSION,
    load_or_write_tiled_proposals,
    load_tiled_proposals,
    proposal_cache_dir,
    proposal_cache_fingerprint,
    proposal_cache_record,
    recipe_whole_window_fingerprint,
    tiled_proposal_record_from_binary_mask,
    tiled_proposal_records_from_sahi_predictions,
    validate_tiled_proposal_cache,
    weights_sha256,
    write_tiled_proposals,
)


def test_weights_sha256_matches_file_bytes(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    expected = hashlib.sha256(b"fake-weights").hexdigest()
    assert weights_sha256(weights) == expected


def test_tiled_proposal_cache_schema_version_is_two() -> None:
    assert TILED_PROPOSAL_CACHE_SCHEMA_VERSION == 2


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
    proposals = [tiled_proposal_record_from_binary_mask(mask, score=0.9)]
    write_tiled_proposals(cache_dir, proposals, record)
    loaded, meta = load_tiled_proposals(cache_dir, expected=record)
    assert loaded == proposals
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


def test_tiled_proposal_records_from_sahi_round_trip_mask() -> None:
    height, width = 16, 16
    sahi_preds = disjoint_sahi_proposals(height, width)
    records = tiled_proposal_records_from_sahi_predictions(
        sahi_preds, height=height, width=width, mask_threshold=0.5
    )
    assert len(records) == 2
    for record in records:
        assert record["segmentation"]["size"][0] < height
        assert record["segmentation"]["size"][1] < width


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
