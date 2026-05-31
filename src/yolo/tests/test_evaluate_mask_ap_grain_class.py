"""Mask AP requires GT and predictions on grain class 0 (ADR 0003)."""

from __future__ import annotations

import numpy as np
from shapely.geometry import box

from common.coco_annotations import build_gt_annotations
from common.prediction_set import GRAIN_CLASS_ID, yolo_prediction_set_to_coco_dt
from common.tests.prediction_set_fixtures import yolo_prediction_set_from_masks
from yolo.coco_instance_ap import evaluate_mask_ap


def test_evaluate_mask_ap_positive_when_gt_and_dt_share_grain_class() -> None:
    height, width = 64, 64
    gt_poly = box(12, 12, 52, 52)
    gt_anns = build_gt_annotations(
        [gt_poly],
        image_id=1,
        height=height,
        width=width,
    )
    assert gt_anns
    assert all(ann["category_id"] == GRAIN_CLASS_ID for ann in gt_anns)

    masks = np.zeros((1, height, width), dtype=np.float32)
    masks[0, 12:52, 12:52] = 1.0
    prediction_set = yolo_prediction_set_from_masks(
        masks_hw=masks,
        scores=np.array([0.95], dtype=np.float32),
        height=height,
        width=width,
    )
    dt_anns = yolo_prediction_set_to_coco_dt(
        prediction_set, image_id=1, height=height, width=width
    )

    summary = evaluate_mask_ap(
        image_id=1,
        file_name="fixture.tif",
        height=height,
        width=width,
        gt_annotations=gt_anns,
        dt_annotations=dt_anns,
    )

    assert summary.ap_50 > 0.0
