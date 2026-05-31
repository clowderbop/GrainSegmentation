"""Tests for tiled detector proposal cache (ADR 0005)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from common.test_inference import load_test_inference_recipe
from yolo.tiled_proposal_cache import (
    TILED_PROPOSAL_CACHE_SCHEMA_VERSION,
    load_or_write_tiled_proposals,
    load_tiled_proposals,
    proposal_cache_dir,
    proposal_cache_record,
    recipe_whole_window_fingerprint,
    validate_tiled_proposal_cache,
    weights_sha256,
    write_tiled_proposals,
)


def test_weights_sha256_matches_file_bytes(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake-weights")
    expected = hashlib.sha256(b"fake-weights").hexdigest()
    assert weights_sha256(weights) == expected


def test_proposal_cache_record_includes_fingerprint_fields() -> None:
    recipe = load_test_inference_recipe()
    record = proposal_cache_record(
        variant="PPL",
        weights_sha256="abc",
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=0.25,
        mask_threshold=0.5,
        sample_id="train",
    )
    assert record["schema_version"] == TILED_PROPOSAL_CACHE_SCHEMA_VERSION
    assert record["variant"] == "PPL"
    assert record["sample_id"] == "train"
    assert record["conf"] == 0.25
    assert record["mask_threshold"] == 0.5


def test_write_and_load_tiled_proposals_round_trip(tmp_path: Path) -> None:
    recipe = load_test_inference_recipe()
    record = proposal_cache_record(
        variant="PPL",
        weights_sha256="deadbeef",
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=0.2,
        mask_threshold=0.45,
        sample_id="train",
    )
    cache_dir = proposal_cache_dir(tmp_path / "PPL", conf=0.2, mask_threshold=0.45)
    proposals = [{"id": 1, "score": 0.9}]
    write_tiled_proposals(cache_dir, proposals, record)
    loaded, meta = load_tiled_proposals(cache_dir, expected=record)
    assert loaded == proposals
    assert meta["variant"] == "PPL"


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
    )
    cache_dir = proposal_cache_dir(tmp_path / "PPL", conf=0.2, mask_threshold=0.45)
    cached = [{"id": 1, "score": 0.9}]
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
    )
    cache_dir = proposal_cache_dir(tmp_path / "PPL", conf=0.2, mask_threshold=0.45)
    fresh = [{"id": 2, "score": 0.7}]

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
    )
    cache_dir = proposal_cache_dir(tmp_path / "PPL", conf=0.2, mask_threshold=0.45)
    stale = dict(record)
    stale["conf"] = 0.99
    write_tiled_proposals(cache_dir, [{"stale": True}], stale)
    updated = [{"id": 3}]

    loaded, from_cache = load_or_write_tiled_proposals(
        cache_dir,
        expected=record,
        compute_fn=lambda: updated,
    )
    assert loaded == updated
    assert from_cache is False


def test_validate_tiled_proposal_cache_rejects_fingerprint_mismatch() -> None:
    recipe = load_test_inference_recipe()
    expected = proposal_cache_record(
        variant="PPL",
        weights_sha256="a",
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=0.25,
        mask_threshold=0.5,
        sample_id="train",
    )
    stale = dict(expected)
    stale["conf"] = 0.99
    with pytest.raises(ValueError, match="conf"):
        validate_tiled_proposal_cache(stale, expected)
