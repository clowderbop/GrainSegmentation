"""Tests for YOLO profile tune detector cache CLI."""

from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from common.test_inference import load_test_inference_recipe
from yolo.profile_tune_detector import write_detector_proposal_cache
from common.tests.profile_tune_fixtures import disjoint_tile_local_proposals
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

    work_root = tmp_path / "tune_out" / ".cache"
    grainseg.mkdir(parents=True)
    train_mosaic = grainseg / "dataset" / "train" / "train_PPL.tif"
    train_mosaic.parent.mkdir(parents=True)
    train_mosaic.write_bytes(b"mosaic")

    return {
        "output_dir": tmp_path / "tune_out",
        "grainseg_root": grainseg,
        "run_root": run_root,
        "work_root": work_root,
        "repo": tmp_path / "repo",
        "variant": variant,
        "weights": weights,
        "train_mosaic": train_mosaic,
    }


def test_write_detector_proposal_cache_does_not_create_staged_manifest_tree(
    detector_tune_layout: dict[str, Path],
) -> None:
    """INTENT: write_detector_proposal_cache stages train TIFF locally without a persistent manifest tree."""
    layout = detector_tune_layout
    staging_dir = layout["output_dir"] / "tmpdir"
    staged_root = layout["work_root"] / layout["variant"] / "staged"

    with (
        patch(
            "yolo.profile_tune_detector_cache.collect_tiled_detector_proposals",
            return_value=[],
        ),
        patch(
            "yolo.profile_tune_detector_cache.load_image_for_yolo",
            return_value=np.zeros((8, 8, 3), dtype=np.uint8),
        ),
        patch(
            "yolo.profile_tune_detector_cache.AutoDetectionModel.from_pretrained",
            return_value=MagicMock(),
        ),
    ):
        write_detector_proposal_cache(
            variant=layout["variant"],
            conf=0.25,
            mask_threshold=0.5,
            grainseg_root=layout["grainseg_root"],
            run_root=layout["run_root"],
            work_root=layout["work_root"],
            device="0",
            train_image_staging_dir=staging_dir,
        )

    assert not staged_root.exists()
    assert (staging_dir / layout["train_mosaic"].name).is_file()


def test_write_detector_proposal_cache_skips_sliced_detection_when_cache_valid(
    detector_tune_layout: dict[str, Path],
) -> None:
    """INTENT: write_detector_proposal_cache reuses a valid on-disk proposal cache without running detection."""
    layout = detector_tune_layout
    conf = 0.25
    fixed_mask = load_test_inference_recipe().yolo.profile.mask_threshold
    cache_dir = proposal_cache_dir(layout["work_root"] / layout["variant"], conf=conf)
    recipe = load_test_inference_recipe()
    height, width = 16, 16
    mask = np.zeros((height, width), dtype=bool)
    mask[2:6, 2:6] = True
    record = proposal_cache_record(
        variant=layout["variant"],
        weights_sha256=weights_sha256(layout["weights"]),
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=conf,
        mask_threshold=fixed_mask,
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
        mask_threshold=fixed_mask,
        grainseg_root=layout["grainseg_root"],
        run_root=layout["run_root"],
        work_root=layout["work_root"],
        device="0",
    )

    with (
        patch(
            "yolo.profile_tune_detector_cache.collect_tiled_detector_proposals",
            side_effect=fake_sliced_detection,
        ),
        patch(
            "yolo.profile_tune_detector_cache.load_image_for_yolo",
            return_value=np.zeros((16, 16, 3), dtype=np.uint8),
        ),
    ):
        write_detector_proposal_cache(
            **common_kwargs,
            local_train_image=layout["train_mosaic"],
        )

    assert detect_calls == []


def test_write_detector_proposal_cache_persists_crop_local_masks_on_disk(
    detector_tune_layout: dict[str, Path],
) -> None:
    """INTENT: write_detector_proposal_cache persists crop-local RLE proposals, not full-section dense masks."""
    layout = detector_tune_layout
    conf = 0.2
    fixed_mask = load_test_inference_recipe().yolo.profile.mask_threshold
    height, width = 64, 64
    image = np.zeros((height, width, 3), dtype=np.uint8)
    tile_preds = disjoint_tile_local_proposals(height, width)

    def fake_iter(_img, _model, *, full_shape, **_kwargs):
        assert full_shape is None
        yield 0, 0, width, height, tile_preds

    common_kwargs = dict(
        variant=layout["variant"],
        conf=conf,
        mask_threshold=fixed_mask,
        grainseg_root=layout["grainseg_root"],
        run_root=layout["run_root"],
        work_root=layout["work_root"],
        device="0",
    )
    with (
        patch(
            "yolo.profile_tune_detector_cache.load_image_for_yolo",
            return_value=image,
        ),
        patch(
            "yolo.profile_tune_detector_cache.AutoDetectionModel.from_pretrained",
            return_value=MagicMock(),
        ),
        patch(
            "yolo.sliced_detection.iter_whole_slice_predictions",
            side_effect=fake_iter,
        ),
    ):
        cache_dir = write_detector_proposal_cache(
            **common_kwargs,
            local_train_image=layout["train_mosaic"],
        )

    expected = detector_cache_expected_record(
        variant=layout["variant"],
        weights_path=layout["weights"],
        conf=conf,
        mask_threshold=fixed_mask,
        sample_id="train",
    )
    loaded, meta = load_tiled_proposals(cache_dir, expected=expected)
    assert meta["schema_version"] == 3
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


def test_profile_tune_detector_main_runs_variant_bundle_for_array_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INTENT: profile_tune_detector CLI dispatches the variant bundle matching --array-index."""
    import yaml

    from yolo.profile_tune_detector import main

    grid_path = tmp_path / "grid.yaml"
    grid_path.write_text(
        yaml.safe_dump({"grid": {"conf": [0.2, 0.3]}}),
        encoding="utf-8",
    )
    output_dir = tmp_path / "run"
    captured: dict[str, object] = {}

    def fake_bundle(**kwargs: object) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "yolo.profile_tune_detector.run_detector_variant_bundle", fake_bundle
    )

    main(
        [
            "--output-dir",
            str(output_dir),
            "--grid-config",
            str(grid_path),
            "--array-index",
            "1",
            "--variants",
            "PPL",
            "--grainseg-root",
            str(tmp_path / "grainseg"),
            "--run-root",
            str(tmp_path / "runs"),
        ]
    )

    assert captured["variant"] == "PPL"


def test_run_detector_variant_bundle_skips_valid_caches_and_fail_fast(
    detector_tune_layout: dict[str, Path],
) -> None:
    """INTENT: run_detector_variant_bundle skips conf values with valid caches and computes only missing ones."""
    from yolo.inference_profile_tune import load_tune_grid
    from yolo.profile_tune_detector import run_detector_variant_bundle

    layout = detector_tune_layout
    grid_path = layout["repo"] / "grid.yaml"
    grid_path.parent.mkdir(parents=True, exist_ok=True)
    grid_path.write_text(
        '{"grid": {"conf": [0.2, 0.3]}}',
        encoding="utf-8",
    )
    spec = load_tune_grid(grid_path)
    conf = 0.2
    fixed_mask = load_test_inference_recipe().yolo.profile.mask_threshold
    cache_dir = proposal_cache_dir(layout["work_root"] / layout["variant"], conf=conf)
    recipe = load_test_inference_recipe()
    height, width = 8, 8
    record = proposal_cache_record(
        variant=layout["variant"],
        weights_sha256=weights_sha256(layout["weights"]),
        recipe_window_fingerprint=recipe_whole_window_fingerprint(recipe),
        conf=conf,
        mask_threshold=fixed_mask,
        sample_id="train",
        height=height,
        width=width,
    )
    write_tiled_proposals(cache_dir, [], record)

    compute_calls: list[tuple[float, float]] = []

    class _FakeDetectionModel:
        confidence_threshold: float = 0.0
        mask_threshold: float = 0.0

    def fake_from_pretrained(**kwargs: object) -> _FakeDetectionModel:
        model = _FakeDetectionModel()
        model.confidence_threshold = float(kwargs["confidence_threshold"])
        model.mask_threshold = float(kwargs["mask_threshold"])
        return model

    def fake_collect(_image, model: _FakeDetectionModel, **kwargs: object) -> list:
        compute_calls.append(
            (float(model.confidence_threshold), float(kwargs["mask_threshold"]))
        )
        return []

    with (
        patch(
            "yolo.profile_tune_detector_cache.load_image_for_yolo",
            return_value=np.zeros((height, width, 3), dtype=np.uint8),
        ),
        patch(
            "yolo.profile_tune_detector_cache.AutoDetectionModel.from_pretrained",
            side_effect=fake_from_pretrained,
        ),
        patch(
            "yolo.profile_tune_detector_cache.collect_tiled_detector_proposals",
            side_effect=fake_collect,
        ),
    ):
        run_detector_variant_bundle(
            variant=layout["variant"],
            spec=spec,
            output_dir=layout["output_dir"],
            grainseg_root=layout["grainseg_root"],
            run_root=layout["run_root"],
            work_root=layout["work_root"],
            device="0",
            local_train_image=layout["train_mosaic"],
        )

    assert compute_calls == [(0.3, fixed_mask)]
