"""Mask AP COCO detections from instance prediction sets."""

from __future__ import annotations

import numpy as np
import pytest
from pycocotools import mask as mask_utils

from common.prediction_set import (
    PredictionSet,
    binary_mask_to_segmentation,
    merge_yolo_proposals_by_score,
    save_prediction_set,
    yolo_prediction_set_to_coco_dt,
)
from common.tests.prediction_set_fixtures import yolo_prediction_set_from_masks


def test_yolo_prediction_set_to_coco_dt_from_rle_fixture(tmp_path) -> None:
    height, width = 16, 16
    binary = np.zeros((height, width), dtype=bool)
    binary[4:10, 4:10] = True
    score = 0.87
    segmentation = binary_mask_to_segmentation(binary, height=height, width=width)

    prediction_set = PredictionSet(
        schema_version=1,
        height=height,
        width=width,
        producer="yolo",
        detections=(
            {
                "segmentation": segmentation,
                "score": score,
                "category_id": 0,
            },
        ),
    )
    ps_path = tmp_path / "prediction_sets" / "sample.json"
    save_prediction_set(ps_path, prediction_set)

    from_ps = yolo_prediction_set_to_coco_dt(
        prediction_set, image_id=1, height=height, width=width
    )

    rle = mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    ys, xs = np.where(binary)
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    expected = [
        {
            "image_id": 1,
            "category_id": 0,
            "segmentation": {"size": [height, width], "counts": counts},
            "bbox": [x0, y0, x1 - x0 + 1.0, y1 - y0 + 1.0],
            "score": score,
        }
    ]

    assert len(from_ps) == len(expected) == 1
    assert from_ps[0]["category_id"] == expected[0]["category_id"]
    assert float(from_ps[0]["score"]) == pytest.approx(float(expected[0]["score"]))
    assert from_ps[0]["segmentation"] == expected[0]["segmentation"]
    assert from_ps[0]["bbox"] == pytest.approx(expected[0]["bbox"], rel=1e-6)


def test_yolo_prediction_set_to_coco_dt_rejects_unet_producer() -> None:
    ps = PredictionSet(
        schema_version=1,
        height=16,
        width=16,
        producer="unet",
        detections=(
            {
                "segmentation": {"size": [16, 16], "counts": "0"},
                "category_id": 0,
            },
        ),
    )
    with pytest.raises(ValueError, match="producer"):
        yolo_prediction_set_to_coco_dt(ps, image_id=1, height=16, width=16)


def test_yolo_mask_ap_uses_score_merged_canonical_detection() -> None:
    height, width = 16, 16
    masks = np.zeros((2, height, width), dtype=np.float32)
    masks[0, 4:12, 4:12] = 1.0
    masks[1, 4:12, 4:12] = 1.0
    proposals = yolo_prediction_set_from_masks(
        masks_hw=masks,
        scores=np.array([0.2, 0.9], dtype=np.float32),
        height=height,
        width=width,
    )
    canonical = merge_yolo_proposals_by_score(proposals)

    from_proposals = yolo_prediction_set_to_coco_dt(
        proposals, image_id=1, height=height, width=width
    )
    from_canonical = yolo_prediction_set_to_coco_dt(
        canonical, image_id=1, height=height, width=width
    )

    assert len(from_proposals) == 2
    assert len(from_canonical) == 1
    assert float(from_canonical[0]["score"]) == pytest.approx(0.9)


def test_yolo_prediction_set_to_coco_dt_matches_mask_planes_fixture() -> None:
    masks = np.zeros((1, 16, 16), dtype=np.float32)
    masks[0, 4:10, 4:10] = 1.0
    scores = np.array([0.87], dtype=np.float32)
    prediction_set = yolo_prediction_set_from_masks(
        masks_hw=masks,
        scores=scores,
        height=16,
        width=16,
    )
    from_ps = yolo_prediction_set_to_coco_dt(
        prediction_set, image_id=1, height=16, width=16
    )
    assert len(from_ps) == 1
    assert float(from_ps[0]["score"]) == pytest.approx(0.87)
