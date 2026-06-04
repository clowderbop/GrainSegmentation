"""Shared fixtures for instance evaluation integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from common.evaluate_instances import InstanceEvalSample
from common.prediction_set import prediction_set_path, save_prediction_set
from common.tests.prediction_set_fixtures import yolo_prediction_set_from_masks
from common.yolo_seg_labels import yolo_seg_labels_to_instance_map


def write_blank_image(path: Path, width: int, height: int) -> None:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    tifffile.imwrite(path, rgb, photometric="rgb")


def perfect_match_eval_sample(
    tmp_path: Path,
    *,
    sample_id: str = "sample",
    width: int = 48,
    height: int = 48,
) -> InstanceEvalSample:
    """One grain with a matching prediction set (PQ == 1 on the patch/whole unit)."""
    image_path = tmp_path / f"{sample_id}_PPL.tif"
    write_blank_image(image_path, width, height)
    gt_txt = tmp_path / f"{sample_id}.txt"
    gt_txt.write_text(
        "0 "
        + " ".join(
            f"{v:.8g}"
            for v in (
                4 / width,
                4 / height,
                20 / width,
                4 / height,
                20 / width,
                20 / height,
                4 / width,
                20 / height,
            )
        )
        + "\n",
        encoding="utf-8",
    )
    gt_map = yolo_seg_labels_to_instance_map(
        gt_txt, image_width=width, image_height=height
    )
    ps = yolo_prediction_set_from_masks(
        masks_hw=gt_map.astype(np.float32)[None, ...],
        scores=np.array([0.99], dtype=np.float32),
        height=height,
        width=width,
    )
    pred_path = prediction_set_path(tmp_path, sample_id)
    save_prediction_set(pred_path, ps)
    return InstanceEvalSample(
        sample_id=sample_id,
        image_path=image_path,
        instance_prediction_set=pred_path,
        gt_txt=gt_txt,
    )
