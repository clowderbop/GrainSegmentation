"""Tests for profile selection candidate worker (ADR 0005)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from common.merged_view_pq import MERGED_VIEW_PQ_RESULT_KEYS
from common.test_inference import (
    YoloInferenceProfileCandidate,
    profile_tune_candidate_from_conf,
)
from yolo.inference_profile_tune import (
    load_profile_selection_row,
    profile_selection_row_path,
    tune_grid_fingerprint,
    variant_metric_column,
)
from yolo.profile_tune_candidate import (
    build_profile_selection_row,
    row_fingerprint_matches,
    score_profile_selection_candidate,
)
from yolo.tests.profile_tune_fixtures import constant_merged_view_pq_result


def test_build_profile_selection_row_includes_per_variant_pq_results() -> None:
    candidate = profile_tune_candidate_from_conf(0.25)
    row = build_profile_selection_row(
        candidate=candidate,
        per_variant_pq_results={
            "PPL": constant_merged_view_pq_result(0.8),
            "PPL+AllPPX": constant_merged_view_pq_result(0.6),
        },
        fingerprint={"candidate_id": candidate.candidate_id()},
    )
    assert row["mean_pq"] == pytest.approx(0.7)
    assert row[variant_metric_column("pq", "PPL")] == pytest.approx(0.8)
    assert row[variant_metric_column("dq", "PPL+AllPPX")] == pytest.approx(0.6)
    assert row["candidate_id"] == candidate.candidate_id()
    assert "aji_plus__PPL" not in row
    assert "f1_iou75__PPL" not in row
    for key in MERGED_VIEW_PQ_RESULT_KEYS:
        assert f"mean_{key}" in row
        assert variant_metric_column(key, "PPL") in row


def test_score_profile_selection_candidate_writes_row_json(tmp_path: Path) -> None:
    candidate = profile_tune_candidate_from_conf(0.25)
    output_dir = tmp_path / "run"
    work_root = output_dir / ".cache"
    grainseg_root = tmp_path / "grainseg"
    run_root = grainseg_root / "runs" / "yolo26-seg"

    with (
        patch(
            "yolo.profile_tune_candidate.candidate_row_fingerprint",
            return_value={"candidate_id": candidate.candidate_id()},
        ),
        patch(
            "yolo.profile_tune_candidate.load_shared_train_gt_map",
            return_value=np.zeros((16, 16), dtype=np.int32),
        ),
        patch(
            "yolo.profile_tune_candidate.score_variant_train_metrics_from_cache",
            side_effect=lambda **kwargs: constant_merged_view_pq_result(
                0.9 if kwargs["variant"] == "PPL" else 0.5
            ),
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
    assert row["mean_pq"] == pytest.approx(0.7)
    assert row[variant_metric_column("pq", "PPL")] == pytest.approx(0.9)


def test_score_profile_selection_candidate_skips_when_fingerprint_matches(
    tmp_path: Path,
) -> None:
    candidate = profile_tune_candidate_from_conf(0.35)
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
                "mean_pq": 0.99,
                variant_metric_column("pq", "PPL"): 0.99,
                "fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    def _fake_score(**kwargs) -> dict[str, float | int]:
        calls.append(kwargs["variant"])
        return constant_merged_view_pq_result(0.1)

    with (
        patch(
            "yolo.profile_tune_candidate.candidate_row_fingerprint",
            return_value=fingerprint,
        ),
        patch(
            "yolo.profile_tune_candidate.score_variant_train_metrics_from_cache",
            side_effect=_fake_score,
        ),
    ):
        score_profile_selection_candidate(
            candidate=candidate,
            variants=("PPL",),
            output_dir=output_dir,
            grainseg_root=tmp_path / "grainseg",
            run_root=tmp_path / "grainseg" / "runs" / "yolo26-seg",
            work_root=output_dir / ".cache",
            grid_config=None,
            resume=True,
        )

    assert calls == []
    assert load_profile_selection_row(row_path)["mean_pq"] == pytest.approx(0.99)


def test_row_fingerprint_matches_requires_exact_equality() -> None:
    assert row_fingerprint_matches({"a": 1}, {"a": 1})
    assert not row_fingerprint_matches({"a": 1}, {"a": 2})


def test_score_profile_selection_candidate_loads_gt_once(tmp_path: Path) -> None:
    candidate = profile_tune_candidate_from_conf(0.25)
    output_dir = tmp_path / "run"
    gt_loads: list[int] = []
    fake_gt = np.zeros((16, 16), dtype=np.int32)

    def _track_gt_load(**kwargs: object) -> np.ndarray:
        gt_loads.append(1)
        return fake_gt

    with (
        patch(
            "yolo.profile_tune_candidate.candidate_row_fingerprint",
            return_value={"candidate_id": candidate.candidate_id()},
        ),
        patch(
            "yolo.profile_tune_candidate.load_shared_train_gt_map",
            side_effect=_track_gt_load,
        ),
        patch(
            "yolo.profile_tune_candidate.score_variant_train_metrics_from_cache",
            side_effect=lambda **kwargs: constant_merged_view_pq_result(0.8),
        ),
    ):
        score_profile_selection_candidate(
            candidate=candidate,
            variants=("PPL", "PPL+AllPPX"),
            output_dir=output_dir,
            grainseg_root=tmp_path / "grainseg",
            run_root=tmp_path / "grainseg" / "runs" / "yolo26-seg",
            work_root=output_dir / ".cache",
            grid_config=None,
            resume=False,
        )

    assert gt_loads == [1]


def test_candidate_row_fingerprint_includes_proposal_schema_v2(
    tmp_path: Path,
) -> None:
    import tifffile

    from yolo.profile_tune_candidate import candidate_row_fingerprint
    from yolo.tiled_proposal_cache import TILED_PROPOSAL_CACHE_SCHEMA_VERSION

    candidate = profile_tune_candidate_from_conf(0.25)
    grainseg_root = tmp_path / "grainseg"
    run_root = grainseg_root / "runs" / "yolo26-seg"
    weights = run_root / "PPL" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    labels_gpkg = grainseg_root / "dataset" / "train" / "train_labels.gpkg"
    labels_gpkg.parent.mkdir(parents=True)
    labels_gpkg.write_bytes(b"labels")
    anchor = grainseg_root / "dataset" / "train" / "train_PPL.tif"
    tifffile.imwrite(anchor, np.zeros((16, 16, 3), dtype=np.uint8))

    fingerprint = candidate_row_fingerprint(
        candidate=candidate,
        variants=("PPL",),
        grainseg_root=grainseg_root,
        run_root=run_root,
        work_root=tmp_path / "work",
        grid_config=None,
    )
    assert (
        fingerprint["variants"]["PPL"]["proposal_schema_version"]
        == TILED_PROPOSAL_CACHE_SCHEMA_VERSION
    )


def test_stale_pre_adr0006_row_fingerprint_triggers_rescore(tmp_path: Path) -> None:
    import tifffile

    from yolo.tiled_proposal_cache import weights_sha256

    candidate = profile_tune_candidate_from_conf(0.25)
    output_dir = tmp_path / "run"
    grainseg_root = tmp_path / "GrainSeg"
    run_root = grainseg_root / "runs" / "yolo26-seg"
    weights = run_root / "PPL" / "weights" / "best.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    labels_gpkg = grainseg_root / "dataset" / "train" / "train_labels.gpkg"
    labels_gpkg.parent.mkdir(parents=True)
    labels_gpkg.write_bytes(b"labels")
    anchor = grainseg_root / "dataset" / "train" / "train_PPL.tif"
    tifffile.imwrite(anchor, np.zeros((16, 16, 3), dtype=np.uint8))

    row_path = profile_selection_row_path(output_dir / "grid", candidate.candidate_id())
    legacy_fingerprint = {
        "candidate_id": candidate.candidate_id(),
        "tune_grid_fingerprint": tune_grid_fingerprint(None),
        "variants": {
            "PPL": {
                "weights_sha256": weights_sha256(weights),
                "gt_cache_fingerprint": {
                    "schema_version": 1,
                    "variant": "PPL",
                    "sample_id": "train",
                    "train_labels_gpkg_sha256": "legacy",
                },
                "proposal_cache_key": (
                    f"c{candidate.conf:g}_t{candidate.mask_threshold:g}"
                ),
            }
        },
    }
    row_path.parent.mkdir(parents=True)
    row_path.write_text(
        json.dumps(
            {
                "candidate_id": candidate.candidate_id(),
                **candidate.to_dict(),
                "mean_pq": 0.99,
                variant_metric_column("pq", "PPL"): 0.99,
                "fingerprint": legacy_fingerprint,
            }
        ),
        encoding="utf-8",
    )
    score_calls: list[str] = []

    def _fake_score(**kwargs: object) -> dict[str, float | int]:
        score_calls.append(str(kwargs["variant"]))
        return constant_merged_view_pq_result(0.42)

    with (
        patch(
            "yolo.profile_tune_candidate.load_shared_train_gt_map",
            return_value=np.zeros((16, 16), dtype=np.int32),
        ),
        patch(
            "yolo.profile_tune_candidate.score_variant_train_metrics_from_cache",
            side_effect=_fake_score,
        ),
    ):
        score_profile_selection_candidate(
            candidate=candidate,
            variants=("PPL",),
            output_dir=output_dir,
            grainseg_root=grainseg_root,
            run_root=run_root,
            work_root=output_dir / ".cache",
            grid_config=None,
            resume=True,
        )

    assert score_calls == ["PPL"]
    row = load_profile_selection_row(row_path)
    assert row["mean_pq"] == pytest.approx(0.42)
    assert "gt_cache_fingerprint" in row["fingerprint"]
    assert row["fingerprint"]["gt_cache_fingerprint"]["schema_version"] == 2
