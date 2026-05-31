"""Tests for profile selection candidate worker (ADR 0005)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from common.test_inference import YoloInferenceProfileCandidate
from yolo.inference_profile_tune import load_profile_selection_row, profile_selection_row_path
from yolo.profile_tune_candidate import (
    build_profile_selection_row,
    row_fingerprint_matches,
    score_profile_selection_candidate,
)


def test_build_profile_selection_row_includes_per_variant_aji() -> None:
    candidate = YoloInferenceProfileCandidate(
        postprocess_type="GREEDYNMM",
        match_metric="IOS",
        match_threshold=0.5,
        conf=0.25,
        mask_threshold=0.5,
    )
    row = build_profile_selection_row(
        candidate=candidate,
        per_variant_aji={"PPL": 0.8, "PPL+AllPPX": 0.6},
        fingerprint={"candidate_id": candidate.candidate_id()},
    )
    assert row["mean_aji"] == pytest.approx(0.7)
    assert row["aji__PPL"] == pytest.approx(0.8)
    assert row["candidate_id"] == candidate.candidate_id()


def test_score_profile_selection_candidate_writes_row_json(tmp_path: Path) -> None:
    candidate = YoloInferenceProfileCandidate(
        postprocess_type="GREEDYNMM",
        match_metric="IOS",
        match_threshold=0.5,
        conf=0.25,
        mask_threshold=0.5,
    )
    output_dir = tmp_path / "run"
    work_root = output_dir / "_work"
    grainseg_root = tmp_path / "grainseg"
    run_root = grainseg_root / "runs" / "yolo26-seg"

    with (
        patch(
            "yolo.profile_tune_candidate.candidate_row_fingerprint",
            return_value={"candidate_id": candidate.candidate_id()},
        ),
        patch(
            "yolo.profile_tune_candidate.score_variant_train_aji_from_cache",
            side_effect=lambda **kwargs: 0.9 if kwargs["variant"] == "PPL" else 0.5,
        ),
    ):
        row_path = score_profile_selection_candidate(
            candidate=candidate,
            variants=("PPL", "PPL+AllPPX"),
            output_dir=output_dir,
            grainseg_root=grainseg_root,
            run_root=run_root,
            work_root=work_root,
            grid_config=None,
            resume=False,
        )

    assert row_path == profile_selection_row_path(output_dir / "grid", candidate.candidate_id())
    row = load_profile_selection_row(row_path)
    assert row["mean_aji"] == pytest.approx(0.7)
    assert row["aji__PPL"] == pytest.approx(0.9)


def test_score_profile_selection_candidate_skips_when_fingerprint_matches(
    tmp_path: Path,
) -> None:
    candidate = YoloInferenceProfileCandidate(
        postprocess_type="NMM",
        match_metric="IOU",
        match_threshold=0.6,
        conf=0.35,
        mask_threshold=0.6,
    )
    output_dir = tmp_path / "run"
    grid_dir = output_dir / "grid"
    row_path = profile_selection_row_path(grid_dir, candidate.candidate_id())
    fingerprint = {"candidate_id": candidate.candidate_id(), "tune_grid_fingerprint": "abc"}
    row_path.parent.mkdir(parents=True)
    row_path.write_text(
        json.dumps(
            {
                "candidate_id": candidate.candidate_id(),
                **candidate.to_dict(),
                "mean_aji": 0.99,
                "aji__PPL": 0.99,
                "fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def _fake_score(**kwargs) -> float:
        calls.append(kwargs["variant"])
        return 0.1

    with (
        patch(
            "yolo.profile_tune_candidate.candidate_row_fingerprint",
            return_value=fingerprint,
        ),
        patch(
            "yolo.profile_tune_candidate.score_variant_train_aji_from_cache",
            side_effect=_fake_score,
        ),
    ):
        score_profile_selection_candidate(
            candidate=candidate,
            variants=("PPL",),
            output_dir=output_dir,
            grainseg_root=tmp_path / "grainseg",
            run_root=tmp_path / "grainseg" / "runs" / "yolo26-seg",
            work_root=output_dir / "_work",
            grid_config=None,
            resume=True,
        )

    assert calls == []
    assert load_profile_selection_row(row_path)["mean_aji"] == pytest.approx(0.99)


def test_row_fingerprint_matches_requires_exact_equality() -> None:
    assert row_fingerprint_matches({"a": 1}, {"a": 1})
    assert not row_fingerprint_matches({"a": 1}, {"a": 2})
