"""U-Net instance label map ↔ prediction set round trip."""

from __future__ import annotations

import time

import numpy as np

from common.prediction_set import (
    build_unet_prediction_set_from_instance_map,
    build_yolo_prediction_set_from_instance_map,
    prediction_set_to_merged_instance_view,
)
from common.semantic_instance import semantic_to_instance_label_map


def _contiguous_instance_map(instance_map: np.ndarray) -> np.ndarray:
    out = np.zeros_like(instance_map)
    labels = sorted(int(x) for x in np.unique(instance_map) if x != 0)
    for index, label_id in enumerate(labels):
        out[instance_map == label_id] = index + 1
    return out


def test_unet_prediction_set_round_trip_matches_contiguous_label_map() -> None:
    """INTENT: U-Net prediction set encode/decode yields a contiguous merged instance map without scores."""
    instance_map = np.zeros((16, 16), dtype=np.int32)
    instance_map[2:8, 2:8] = 3
    instance_map[9:14, 9:14] = 7

    prediction_set = build_unet_prediction_set_from_instance_map(instance_map)
    assert prediction_set.producer == "unet"
    assert prediction_set.height == 16
    assert prediction_set.width == 16
    assert len(prediction_set.detections) == 2
    assert all("score" not in det for det in prediction_set.detections)

    merged = prediction_set_to_merged_instance_view(prediction_set)
    expected = _contiguous_instance_map(instance_map)
    np.testing.assert_array_equal(merged, expected)


def test_cc_extraction_fixture_round_trips_through_prediction_set() -> None:
    """INTENT: semantic_to_instance_label_map output round-trips through U-Net prediction set I/O."""
    semantic = np.zeros((16, 16), dtype=np.int32)
    semantic[2:8, 2:8] = 1
    instance_map = semantic_to_instance_label_map(semantic)

    prediction_set = build_unet_prediction_set_from_instance_map(instance_map)
    merged = prediction_set_to_merged_instance_view(prediction_set)
    np.testing.assert_array_equal(merged, _contiguous_instance_map(instance_map))


def _sequential_single_pixel_labels(n: int, height: int, width: int) -> np.ndarray:
    if n > height * width:
        raise ValueError("n exceeds raster capacity")
    labeled = np.zeros((height, width), dtype=np.int32)
    rows, cols = np.divmod(np.arange(n), width)
    labeled[rows, cols] = np.arange(1, n + 1, dtype=np.int32)
    return labeled


def test_unet_instance_map_encode_scales_with_pixels_not_instance_count() -> None:
    """INTENT: encoding many small labels on a train-aspect canvas stays pixel-linear, not instances×pixels."""
    n_labels = 5_000
    height, width = 1_000, 5_200
    instance_map = _sequential_single_pixel_labels(n_labels, height=height, width=width)

    t0 = time.perf_counter()
    prediction_set = build_unet_prediction_set_from_instance_map(instance_map)
    merged = prediction_set_to_merged_instance_view(prediction_set)
    elapsed = time.perf_counter() - t0

    assert len(prediction_set.detections) == n_labels
    assert all("score" not in det for det in prediction_set.detections)
    np.testing.assert_array_equal(merged, _contiguous_instance_map(instance_map))
    assert elapsed < 15.0, (
        f"encode+round-trip took {elapsed:.2f}s; expected O(pixels), not O(instances×pixels)"
    )


def test_unet_prediction_set_round_trip_same_label_disjoint_regions() -> None:
    """INTENT: one label id spanning disjoint blobs encodes as a single detection that round-trips."""
    instance_map = np.zeros((32, 32), dtype=np.int32)
    instance_map[2:5, 2:5] = 42
    instance_map[20:23, 20:23] = 42
    instance_map[8:12, 24:28] = 99

    prediction_set = build_unet_prediction_set_from_instance_map(instance_map)
    assert len(prediction_set.detections) == 2
    assert all("score" not in det for det in prediction_set.detections)

    merged = prediction_set_to_merged_instance_view(prediction_set)
    np.testing.assert_array_equal(merged, _contiguous_instance_map(instance_map))


def test_unet_prediction_set_round_trip_uses_crop_offsets_on_large_canvas() -> None:
    """INTENT: small grains on a large canvas round-trip via crop-local RLE with bbox offsets."""
    instance_map = np.zeros((64, 64), dtype=np.int32)
    instance_map[10:14, 10:14] = 5
    instance_map[40:44, 50:54] = 1_000

    prediction_set = build_unet_prediction_set_from_instance_map(instance_map)
    assert len(prediction_set.detections) == 2
    assert any(
        "offset_y" in det and "offset_x" in det for det in prediction_set.detections
    )

    merged = prediction_set_to_merged_instance_view(prediction_set)
    np.testing.assert_array_equal(merged, _contiguous_instance_map(instance_map))


def test_yolo_prediction_set_round_trip_gapped_labels_with_scores() -> None:
    """INTENT: YOLO instance-map encoding round-trips gapped label ids and preserves per-label scores."""
    instance_map = np.zeros((32, 32), dtype=np.int32)
    instance_map[4:10, 4:10] = 7
    instance_map[18:24, 18:24] = 50
    scores = {7: 0.25, 50: 0.9}

    prediction_set = build_yolo_prediction_set_from_instance_map(
        instance_map,
        score_for_label=lambda label_id: scores[label_id],
    )
    assert prediction_set.producer == "yolo"
    assert len(prediction_set.detections) == 2
    assert {float(det["score"]) for det in prediction_set.detections} == {0.25, 0.9}

    merged = prediction_set_to_merged_instance_view(prediction_set)
    np.testing.assert_array_equal(merged, _contiguous_instance_map(instance_map))


def test_yolo_prediction_set_round_trip_same_label_disjoint_regions() -> None:
    """INTENT: YOLO encoding merges disjoint blobs under one label id into one scored detection."""
    instance_map = np.zeros((32, 32), dtype=np.int32)
    instance_map[2:5, 2:5] = 42
    instance_map[20:23, 20:23] = 42
    instance_map[8:12, 24:28] = 99
    scores = {42: 0.4, 99: 0.8}

    prediction_set = build_yolo_prediction_set_from_instance_map(
        instance_map,
        score_for_label=lambda label_id: scores[label_id],
    )
    assert len(prediction_set.detections) == 2
    assert {float(det["score"]) for det in prediction_set.detections} == {0.4, 0.8}

    merged = prediction_set_to_merged_instance_view(prediction_set)
    np.testing.assert_array_equal(merged, _contiguous_instance_map(instance_map))


def test_yolo_prediction_set_round_trip_uses_crop_offsets_on_large_canvas() -> None:
    """INTENT: YOLO grains on a large canvas round-trip via crop-local RLE with bbox offsets."""
    instance_map = np.zeros((64, 64), dtype=np.int32)
    instance_map[10:14, 10:14] = 5
    instance_map[40:44, 50:54] = 1_000
    scores = {5: 0.15, 1_000: 0.95}

    prediction_set = build_yolo_prediction_set_from_instance_map(
        instance_map,
        score_for_label=lambda label_id: scores[label_id],
    )
    assert len(prediction_set.detections) == 2
    assert any(
        "offset_y" in det and "offset_x" in det for det in prediction_set.detections
    )
    assert {float(det["score"]) for det in prediction_set.detections} == {0.15, 0.95}

    merged = prediction_set_to_merged_instance_view(prediction_set)
    np.testing.assert_array_equal(merged, _contiguous_instance_map(instance_map))
