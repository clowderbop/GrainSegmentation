"""Patch instance eval reports include patch metric aggregates in extras."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from common.evaluate_instances import InstanceEvalSample, evaluate_instance_samples
from common.prediction_set import prediction_set_path, save_prediction_set
from common.tests.prediction_set_fixtures import yolo_prediction_set_from_masks


def _write_blank_image(path: Path, width: int, height: int) -> None:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    tifffile.imwrite(path, rgb, photometric="rgb")


def test_patch_eval_report_includes_grainy_aggregates(tmp_path: Path) -> None:
    width, height = 32, 32
    samples: list[InstanceEvalSample] = []
    for index in (1, 2):
        image_path = tmp_path / f"patch{index}_PPL.tif"
        _write_blank_image(image_path, width, height)
        gt_map = np.zeros((height, width), dtype=np.int32)
        gt_map[4:20, 4:20] = 1
        ps = yolo_prediction_set_from_masks(
            masks_hw=gt_map.astype(np.float32)[None, ...],
            scores=np.array([0.99], dtype=np.float32),
            height=height,
            width=width,
        )
        pred_path = prediction_set_path(tmp_path, f"patch{index}")
        save_prediction_set(pred_path, ps)
        gt_txt = tmp_path / f"patch{index}.txt"
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
        samples.append(
            InstanceEvalSample(
                sample_id=f"patch{index}",
                image_path=image_path,
                instance_prediction_set=pred_path,
                gt_txt=gt_txt,
            )
        )

    report = evaluate_instance_samples(
        samples,
        model_type="yolo",
        variant="PPL",
        unit="patch",
    )
    extras = report["extras"]
    assert extras["n_patches"] == 2
    assert extras["mean_aji_grainy"] == 1.0
    assert "mean_aji_weighted" in extras
