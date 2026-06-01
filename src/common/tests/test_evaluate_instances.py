"""Tests for common.evaluate_instances."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from common.evaluate_instances import (
    InstanceEvalSample,
    collect_manifest_samples,
    evaluate_instance_samples,
)
from common.prediction_set import prediction_set_path, save_prediction_set
from common.tests.prediction_set_fixtures import yolo_prediction_set_from_masks
from common.yolo_seg_labels import yolo_seg_labels_to_instance_map


def _write_blank_image(path: Path, width: int, height: int) -> None:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    tifffile.imwrite(path, rgb, photometric="rgb")


def test_collect_manifest_samples_patch(tmp_path: Path) -> None:
    image_path = tmp_path / "patch001_PPL.tif"
    _write_blank_image(image_path, 64, 64)
    ps = yolo_prediction_set_from_masks(
        masks_hw=np.ones((1, 64, 64), dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        height=64,
        width=64,
    )
    pred_path = prediction_set_path(tmp_path, "patch001")
    save_prediction_set(pred_path, ps)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variant": "PPL",
                "unit": "patch",
                "grainseg_root": str(tmp_path),
                "path_base": "work_root",
                "samples": [
                    {
                        "sample_id": "patch001",
                        "image": str(image_path),
                        "instance_prediction_set": str(pred_path),
                        "gt_txt": str(tmp_path / "patch001.txt"),
                        "gt_origin": "patch_stem",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "patch001.txt").write_text("0 0 0 0 0 0 0 0\n", encoding="utf-8")

    samples = collect_manifest_samples(manifest_path)
    assert len(samples) == 1
    assert samples[0].sample_id == "patch001"


def test_evaluate_two_synthetic_instance_maps(tmp_path: Path) -> None:
    width, height = 48, 48
    image_path = tmp_path / "sample_PPL.tif"
    _write_blank_image(image_path, width, height)

    gt_txt = tmp_path / "sample.txt"
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
    pred_path = prediction_set_path(tmp_path, "sample")
    save_prediction_set(pred_path, ps)

    sample = InstanceEvalSample(
        sample_id="sample",
        image_path=image_path,
        instance_prediction_set=pred_path,
        gt_txt=gt_txt,
    )
    report = evaluate_instance_samples(
        [sample],
        model_type="yolo",
        variant="PPL",
        unit="patch",
    )
    assert report["samples"][0]["aji"] == 1.0
