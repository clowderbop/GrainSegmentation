"""Tests for YOLO profile tune detector cache CLI."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yolo.profile_tune_detector import write_detector_proposal_cache
from yolo.tiled_proposal_cache import (
    detector_cache_expected_record,
    proposal_cache_dir,
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
    record = detector_cache_expected_record(
        variant=layout["variant"],
        weights_path=layout["weights"],
        conf=conf,
        mask_threshold=mask_threshold,
        sample_id="train",
    )
    write_tiled_proposals(cache_dir, [{"cached": True}], record)

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
            "yolo.profile_tune_detector.run_whole_sliced_detection",
            side_effect=fake_sliced_detection,
        ),
    ):
        write_detector_proposal_cache(**common_kwargs)

    assert detect_calls == []
