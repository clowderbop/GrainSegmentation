"""U-Net instance label map ↔ prediction set round trip."""

from __future__ import annotations

import numpy as np

from common.prediction_set import (
    build_unet_prediction_set_from_instance_map,
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
