"""Tests for common.prediction_set (instance prediction set v1)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from common.instance_maps import yolo_detections_to_instance_map_by_score
from common.prediction_set import (
    PredictionSet,
    assert_yolo_grains_non_overlapping,
    binary_mask_to_segmentation,
    build_yolo_prediction_set_from_ultralytics,
    load_prediction_set,
    merge_yolo_proposals_by_score,
    prediction_set_path,
    prediction_set_to_merged_instance_view,
    save_prediction_set,
    segmentation_to_binary_mask,
    validate_prediction_set,
    yolo_detection_mask_in_section,
)
from common.tests.prediction_set_fixtures import (
    assert_instance_map_partitions_equal,
    assert_yolo_canonical_sets_equal,
    yolo_prediction_set_from_masks,
)


def test_prediction_set_save_load_round_trip(tmp_path: Path) -> None:
    data = {
        "schema_version": 1,
        "height": 8,
        "width": 8,
        "producer": "yolo",
        "detections": [
            {
                "segmentation": {
                    "size": [8, 8],
                    "counts": "0",
                },
                "score": 0.9,
                "category_id": 0,
            }
        ],
    }
    path = prediction_set_path(tmp_path, "patch001")
    save_prediction_set(path, data)
    loaded = load_prediction_set(path)
    assert loaded.schema_version == 1
    assert loaded.producer == "yolo"
    assert len(loaded.detections) == 1
    assert loaded.detections[0]["score"] == pytest.approx(0.9)


def test_validate_rejects_unet_detection_with_score() -> None:
    with pytest.raises(ValueError, match="score"):
        validate_prediction_set(
            {
                "schema_version": 1,
                "height": 4,
                "width": 4,
                "producer": "unet",
                "detections": [
                    {
                        "segmentation": {"size": [4, 4], "counts": "0"},
                        "score": 0.5,
                        "category_id": 0,
                    }
                ],
            }
        )


def test_validate_rejects_yolo_detection_without_score() -> None:
    with pytest.raises(ValueError, match="requires score"):
        validate_prediction_set(
            {
                "schema_version": 1,
                "height": 4,
                "width": 4,
                "producer": "yolo",
                "detections": [
                    {
                        "segmentation": {"size": [4, 4], "counts": "0"},
                        "category_id": 0,
                    }
                ],
            }
        )


def test_yolo_detection_mask_in_section_fast_path_skips_section_plane() -> None:
    """Full-section RLE at (0, 0) returns the decoded mask without allocating a section plane."""
    height, width = 8, 8
    mask = np.zeros((height, width), dtype=bool)
    mask[2:5, 2:5] = True
    det = {
        "segmentation": binary_mask_to_segmentation(mask, height=height, width=width),
        "score": 0.5,
        "category_id": 0,
    }
    with patch("numpy.zeros", side_effect=AssertionError("must not allocate section plane")):
        placed = yolo_detection_mask_in_section(det, height=height, width=width)

    np.testing.assert_array_equal(placed, mask)


def test_yolo_detection_mask_in_section_places_crop_at_offset() -> None:
    crop = np.zeros((3, 4), dtype=bool)
    crop[1, 2] = True
    det = {
        "segmentation": binary_mask_to_segmentation(crop, height=3, width=4),
        "offset_y": 2,
        "offset_x": 1,
        "score": 0.6,
        "category_id": 0,
    }
    placed = yolo_detection_mask_in_section(det, height=8, width=8)
    expected = np.zeros((8, 8), dtype=bool)
    expected[3, 3] = True
    np.testing.assert_array_equal(placed, expected)


def test_yolo_detection_mask_in_section_rejects_negative_offset() -> None:
    det = {
        "segmentation": binary_mask_to_segmentation(
            np.ones((2, 2), dtype=bool), height=2, width=2
        ),
        "offset_y": -1,
        "offset_x": 0,
        "score": 0.5,
        "category_id": 0,
    }
    with pytest.raises(ValueError, match="Invalid mask offset"):
        yolo_detection_mask_in_section(det, height=8, width=8)


def test_yolo_detection_mask_in_section_rejects_crop_outside_section() -> None:
    det = {
        "segmentation": binary_mask_to_segmentation(
            np.ones((3, 3), dtype=bool), height=3, width=3
        ),
        "offset_y": 6,
        "offset_x": 6,
        "score": 0.5,
        "category_id": 0,
    }
    with pytest.raises(ValueError, match="extends outside section"):
        yolo_detection_mask_in_section(det, height=8, width=8)


def test_validate_rejects_yolo_detection_with_only_one_offset_field() -> None:
    with pytest.raises(ValueError, match="offset_y and offset_x must both be set"):
        validate_prediction_set(
            {
                "schema_version": 1,
                "height": 8,
                "width": 8,
                "producer": "yolo",
                "detections": [
                    {
                        "segmentation": {"size": [2, 2], "counts": "0"},
                        "offset_y": 1,
                        "score": 0.5,
                        "category_id": 0,
                    }
                ],
            }
        )


def test_validate_yolo_offset_fields_save_load_roundtrip(tmp_path: Path) -> None:
    crop = np.zeros((3, 4), dtype=bool)
    crop[1, 2] = True
    payload = {
        "schema_version": 1,
        "height": 8,
        "width": 8,
        "producer": "yolo",
        "detections": [
            {
                "segmentation": binary_mask_to_segmentation(crop, height=3, width=4),
                "offset_y": 2,
                "offset_x": 1,
                "score": 0.75,
                "category_id": 0,
            }
        ],
    }
    path = prediction_set_path(tmp_path, "offset_crop")
    save_prediction_set(path, payload)
    loaded = load_prediction_set(path)
    det = loaded.detections[0]
    assert det["offset_y"] == 2
    assert det["offset_x"] == 1
    placed = yolo_detection_mask_in_section(det, height=8, width=8)
    merged = prediction_set_to_merged_instance_view(loaded)
    assert int(placed.sum()) == int((merged > 0).sum()) == 1


def test_yolo_merged_instance_view_decodes_one_mask_at_a_time() -> None:
    masks = np.zeros((4, 8, 8), dtype=np.float32)
    masks[0, 1:4, 1:4] = 1.0
    masks[1, 2:5, 2:5] = 1.0
    masks[2, 3:6, 3:6] = 1.0
    masks[3, 4:7, 4:7] = 1.0
    scores = np.array([0.1, 0.4, 0.7, 0.9], dtype=np.float32)
    prediction_set = yolo_prediction_set_from_masks(
        masks_hw=masks,
        scores=scores,
        height=8,
        width=8,
    )

    live_decodes = 0
    peak_decodes = 0

    def counting_decode(segmentation: dict) -> np.ndarray:
        nonlocal live_decodes, peak_decodes
        live_decodes += 1
        peak_decodes = max(peak_decodes, live_decodes)
        try:
            return segmentation_to_binary_mask(segmentation)
        finally:
            live_decodes -= 1

    with patch("common.prediction_set.segmentation_to_binary_mask", counting_decode):
        with patch("numpy.stack", side_effect=AssertionError("must not stack all masks")):
            canonical = merge_yolo_proposals_by_score(prediction_set)
            merged = prediction_set_to_merged_instance_view(canonical)

    pre_change_eval = yolo_detections_to_instance_map_by_score(
        prediction_set.detections,
        height=8,
        width=8,
        decode_segmentation=segmentation_to_binary_mask,
    )
    assert_instance_map_partitions_equal(merged, pre_change_eval)
    assert peak_decodes == 1


def test_merge_yolo_proposals_keeps_disjoint_grains() -> None:
    height, width = 8, 8
    masks = np.zeros((2, height, width), dtype=np.float32)
    masks[0, 1:3, 1:3] = 1.0
    masks[1, 5:7, 5:7] = 1.0
    scores = np.array([0.3, 0.8], dtype=np.float32)
    proposals = yolo_prediction_set_from_masks(
        masks_hw=masks, scores=scores, height=height, width=width
    )

    merged = merge_yolo_proposals_by_score(proposals)

    assert len(merged.detections) == 2
    merged_scores = sorted(float(det["score"]) for det in merged.detections)
    assert merged_scores == [pytest.approx(0.3), pytest.approx(0.8)]
    assert_yolo_grains_non_overlapping(merged)


def test_assert_yolo_grains_non_overlapping_rejects_overlapping_proposals() -> None:
    masks = np.zeros((2, 8, 8), dtype=np.float32)
    masks[0, 2:6, 2:6] = 1.0
    masks[1, 2:6, 2:6] = 1.0
    proposals = yolo_prediction_set_from_masks(
        masks_hw=masks,
        scores=np.array([0.2, 0.9], dtype=np.float32),
        height=8,
        width=8,
    )
    with pytest.raises(ValueError, match="overlapping"):
        assert_yolo_grains_non_overlapping(proposals)


def test_merge_yolo_proposals_collapses_overlapping_proposals_to_one_grain() -> None:
    masks = np.zeros((2, 8, 8), dtype=np.float32)
    masks[0, 2:6, 2:6] = 1.0
    masks[1, 2:6, 2:6] = 1.0
    proposals = yolo_prediction_set_from_masks(
        masks_hw=masks,
        scores=np.array([0.2, 0.9], dtype=np.float32),
        height=8,
        width=8,
    )
    canonical = merge_yolo_proposals_by_score(proposals)
    assert len(proposals.detections) == 2
    assert len(canonical.detections) == 1
    assert canonical.detections[0]["score"] == pytest.approx(0.9)
    assert_yolo_grains_non_overlapping(canonical)
    assert_yolo_canonical_sets_equal(canonical, merge_yolo_proposals_by_score(proposals))


def test_merge_yolo_proposals_save_load_preserves_canonical_rles(tmp_path: Path) -> None:
    masks = np.zeros((2, 8, 8), dtype=np.float32)
    masks[0, 2:6, 2:6] = 1.0
    masks[1, 2:6, 2:6] = 1.0
    proposals = yolo_prediction_set_from_masks(
        masks_hw=masks,
        scores=np.array([0.2, 0.9], dtype=np.float32),
        height=8,
        width=8,
    )
    canonical = merge_yolo_proposals_by_score(proposals)
    path = prediction_set_path(tmp_path, "sample")
    save_prediction_set(path, canonical)
    reloaded = load_prediction_set(path)
    assert_yolo_canonical_sets_equal(reloaded, canonical)


def test_merge_yolo_proposals_matches_pre_change_eval_score_merge() -> None:
    height, width = 8, 8
    masks = np.zeros((2, height, width), dtype=np.float32)
    masks[0, 2:6, 2:6] = 1.0
    masks[1, 2:6, 2:6] = 1.0
    scores = np.array([0.2, 0.9], dtype=np.float32)

    proposals = yolo_prediction_set_from_masks(
        masks_hw=masks,
        scores=scores,
        height=height,
        width=width,
    )
    canonical = merge_yolo_proposals_by_score(proposals)
    pre_change_eval = yolo_detections_to_instance_map_by_score(
        proposals.detections,
        height=height,
        width=width,
        decode_segmentation=segmentation_to_binary_mask,
    )
    post_eval = prediction_set_to_merged_instance_view(canonical)

    assert_instance_map_partitions_equal(post_eval, pre_change_eval)
    assert_yolo_canonical_sets_equal(canonical, merge_yolo_proposals_by_score(proposals))


def test_prediction_set_path_layout() -> None:
    root = Path("/run/out")
    assert prediction_set_path(root, "abc") == root / "prediction_sets" / "abc.json"


class _PerMaskTensorStack:
    """Tensor stack that forbids bulk ``.cpu().numpy()`` (index-only access)."""

    def __init__(self, planes: list[torch.Tensor]) -> None:
        self._planes = planes

    def __len__(self) -> int:
        return len(self._planes)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self._planes[index]

    def cpu(self) -> None:
        raise AssertionError("must not materialize full mask stack")


def test_build_yolo_prediction_set_from_ultralytics_per_mask() -> None:
    height, width = 8, 8
    plane = torch.zeros((height, width), dtype=torch.float32)
    plane[2:6, 2:6] = 1.0
    masks = MagicMock()
    masks.__len__.return_value = 1
    masks.data = _PerMaskTensorStack([plane])
    result = MagicMock()
    result.masks = masks
    result.boxes.conf = torch.tensor([0.75], dtype=torch.float32)

    from_ultralytics = build_yolo_prediction_set_from_ultralytics(
        result, height=height, width=width
    )
    expected = yolo_prediction_set_from_masks(
        masks_hw=plane.numpy()[None, ...],
        scores=np.array([0.75], dtype=np.float32),
        height=height,
        width=width,
    )
    assert len(from_ultralytics.detections) == len(expected.detections) == 1
    assert from_ultralytics.detections[0]["score"] == pytest.approx(0.75)
    np.testing.assert_array_equal(
        prediction_set_to_merged_instance_view(
            merge_yolo_proposals_by_score(from_ultralytics)
        ),
        prediction_set_to_merged_instance_view(merge_yolo_proposals_by_score(expected)),
    )
