"""Tests for profile selection candidate cache staging (ADR 0005)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from common.profile_tune_gt_cache import gt_cache_dir, write_gt_instance_map_cache
from common.test_inference import YoloInferenceProfileCandidate
from yolo.tiled_proposal_cache import proposal_cache_dir, write_tiled_proposals


def _write_gt_cache(scratch_cache: Path) -> None:
    gt_map = np.zeros((4, 4), dtype=np.int32)
    fingerprint = {"schema_version": 2, "sample_id": "train", "width": 4, "height": 4}
    write_gt_instance_map_cache(
        gt_cache_dir(scratch_cache), gt_map, fingerprint=fingerprint
    )


def _write_proposal_cache(
    scratch_cache: Path,
    *,
    variant: str,
    conf: float,
    mask_threshold: float,
) -> None:
    cache_dir = proposal_cache_dir(
        scratch_cache / variant, conf=conf, mask_threshold=mask_threshold
    )
    write_tiled_proposals(
        cache_dir,
        [],
        {
            "schema_version": 2,
            "variant": variant,
            "conf": conf,
            "mask_threshold": mask_threshold,
            "sample_id": "train",
            "height": 4,
            "width": 4,
        },
    )


def test_stage_candidate_work_copies_gt_cache_preserving_layout(tmp_path: Path) -> None:
    from yolo.profile_tune_cache_stage import stage_candidate_work

    scratch_cache = tmp_path / "scratch" / ".cache"
    _write_gt_cache(scratch_cache)
    candidate = YoloInferenceProfileCandidate(
        postprocess_type="GREEDYNMM",
        match_metric="IOS",
        match_threshold=0.4,
        conf=0.2,
        mask_threshold=0.45,
    )
    _write_proposal_cache(
        scratch_cache,
        variant="PPL",
        conf=candidate.conf,
        mask_threshold=candidate.mask_threshold,
    )
    tmp_work = tmp_path / "tmpdir" / "work"

    work_root, timings = stage_candidate_work(
        scratch_cache_root=scratch_cache,
        tmp_work_root=tmp_work,
        candidate=candidate,
        variants=("PPL",),
    )

    assert work_root == tmp_work
    assert (work_root / "gt_cache" / "train" / "instance_map.npz").is_file()
    assert (work_root / "gt_cache" / "train" / "fingerprint.json").is_file()
    assert timings.copy_gt_s >= 0.0


def test_stage_candidate_work_copies_only_candidate_proposal_subtrees(
    tmp_path: Path,
) -> None:
    from yolo.profile_tune_cache_stage import stage_candidate_work

    scratch_cache = tmp_path / "scratch" / ".cache"
    _write_gt_cache(scratch_cache)
    candidate = YoloInferenceProfileCandidate(
        postprocess_type="GREEDYNMM",
        match_metric="IOS",
        match_threshold=0.4,
        conf=0.2,
        mask_threshold=0.45,
    )
    other = YoloInferenceProfileCandidate(
        postprocess_type="GREEDYNMM",
        match_metric="IOS",
        match_threshold=0.4,
        conf=0.5,
        mask_threshold=0.6,
    )
    for variant in ("PPL", "PPLPPXblend"):
        _write_proposal_cache(
            scratch_cache,
            variant=variant,
            conf=candidate.conf,
            mask_threshold=candidate.mask_threshold,
        )
        _write_proposal_cache(
            scratch_cache,
            variant=variant,
            conf=other.conf,
            mask_threshold=other.mask_threshold,
        )

    work_root, _timings = stage_candidate_work(
        scratch_cache_root=scratch_cache,
        tmp_work_root=tmp_path / "tmpdir" / "work",
        candidate=candidate,
        variants=("PPL", "PPLPPXblend"),
    )

    wanted = proposal_cache_dir(
        Path("PPL"), conf=candidate.conf, mask_threshold=candidate.mask_threshold
    )
    assert (work_root / wanted / "proposals.pkl").is_file()
    assert not (
        work_root
        / proposal_cache_dir(
            Path("PPL"), conf=other.conf, mask_threshold=other.mask_threshold
        )
    ).exists()
    blend_wanted = proposal_cache_dir(
        Path("PPLPPXblend"),
        conf=candidate.conf,
        mask_threshold=candidate.mask_threshold,
    )
    assert (work_root / blend_wanted / "proposals.pkl").is_file()


def test_stage_candidate_work_missing_gt_cache_fails_fast(tmp_path: Path) -> None:
    from yolo.profile_tune_cache_stage import stage_candidate_work

    scratch_cache = tmp_path / "scratch" / ".cache"
    candidate = YoloInferenceProfileCandidate(
        postprocess_type="GREEDYNMM",
        match_metric="IOS",
        match_threshold=0.4,
        conf=0.2,
        mask_threshold=0.45,
    )

    with pytest.raises(FileNotFoundError, match="gt_cache/train"):
        stage_candidate_work(
            scratch_cache_root=scratch_cache,
            tmp_work_root=tmp_path / "tmpdir" / "work",
            candidate=candidate,
            variants=("PPL",),
        )


def test_stage_candidate_work_missing_proposal_cache_points_at_scratch_path(
    tmp_path: Path,
) -> None:
    from yolo.profile_tune_cache_stage import stage_candidate_work

    scratch_cache = tmp_path / "scratch" / ".cache"
    _write_gt_cache(scratch_cache)
    candidate = YoloInferenceProfileCandidate(
        postprocess_type="GREEDYNMM",
        match_metric="IOS",
        match_threshold=0.4,
        conf=0.2,
        mask_threshold=0.45,
    )

    with pytest.raises(FileNotFoundError, match="PPL/tiled_proposals/c0.2"):
        stage_candidate_work(
            scratch_cache_root=scratch_cache,
            tmp_work_root=tmp_path / "tmpdir" / "work",
            candidate=candidate,
            variants=("PPL",),
        )


def test_stage_candidate_work_returns_proposal_timing_breakdown(tmp_path: Path) -> None:
    from yolo.profile_tune_cache_stage import stage_candidate_work

    scratch_cache = tmp_path / "scratch" / ".cache"
    _write_gt_cache(scratch_cache)
    candidate = YoloInferenceProfileCandidate(
        postprocess_type="GREEDYNMM",
        match_metric="IOS",
        match_threshold=0.4,
        conf=0.2,
        mask_threshold=0.45,
    )
    for variant in ("PPL", "PPLPPXblend"):
        _write_proposal_cache(
            scratch_cache,
            variant=variant,
            conf=candidate.conf,
            mask_threshold=candidate.mask_threshold,
        )

    _work_root, timings = stage_candidate_work(
        scratch_cache_root=scratch_cache,
        tmp_work_root=tmp_path / "tmpdir" / "work",
        candidate=candidate,
        variants=("PPL", "PPLPPXblend"),
    )

    assert timings.copy_gt_s >= 0.0
    assert timings.copy_proposals_s >= 0.0
    assert set(timings.per_variant_proposals_s) == {"PPL", "PPLPPXblend"}
    assert all(s >= 0.0 for s in timings.per_variant_proposals_s.values())


def test_stage_detector_train_image_copies_only_train_mosaic(tmp_path: Path) -> None:
    from common.variants import get_variant
    from yolo.profile_tune_cache_stage import stage_detector_train_image

    grainseg = tmp_path / "GrainSeg"
    variant = "PPL"
    spec = get_variant(variant)
    mosaic = grainseg / spec.paths.train_mosaic_stacked
    mosaic.parent.mkdir(parents=True)
    mosaic.write_bytes(b"stacked-tiff-bytes")

    tmp_dir = tmp_path / "tmpdir" / "det"
    staged = stage_detector_train_image(
        grainseg_root=grainseg, variant=variant, tmp_dir=tmp_dir
    )

    assert staged.sample_id == "train"
    assert staged.image_path == tmp_dir / mosaic.name
    assert staged.image_path.read_bytes() == b"stacked-tiff-bytes"
    assert staged.copy_s >= 0.0
    assert not (tmp_path / "staged").exists()


def test_format_candidate_stage_timings_lists_aggregate_and_per_variant() -> None:
    from yolo.profile_tune_cache_stage import (
        CandidateStageTimings,
        format_candidate_stage_timings,
    )

    line = format_candidate_stage_timings(
        CandidateStageTimings(
            copy_gt_s=1.2,
            copy_proposals_s=3.4,
            per_variant_proposals_s={"PPL": 1.0, "PPLPPXblend": 2.4},
        )
    )
    assert "copy_gt_s=1.2" in line
    assert "copy_proposals_s=3.4" in line
    assert "copy_proposals_PPL_s=1.0" in line
    assert "copy_proposals_PPLPPXblend_s=2.4" in line
