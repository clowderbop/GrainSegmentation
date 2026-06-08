"""Instance evaluation driven by instance prediction set JSON."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from common.evaluate_instances import (
    collect_manifest_samples,
    evaluate_instance_samples,
)
from common.prediction_set import (
    build_unet_prediction_set_from_instance_map,
    merge_yolo_proposals_by_score,
    prediction_set_path,
    save_prediction_set,
)
from common.tests.prediction_set_fixtures import yolo_prediction_set_from_masks
from common.yolo_seg_labels import yolo_seg_labels_to_instance_map


def _write_blank_image(path: Path, width: int, height: int) -> None:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    tifffile.imwrite(path, rgb, photometric="rgb")


def test_evaluate_instances_from_prediction_set_manifest(tmp_path: Path) -> None:
    """INTENT: YOLO prediction-set manifest evaluation yields unit PQ for matching GT and predictions."""
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

    masks = gt_map.astype(np.float32)[None, ...]
    ps = merge_yolo_proposals_by_score(
        yolo_prediction_set_from_masks(
            masks_hw=masks,
            scores=np.array([0.99], dtype=np.float32),
            height=height,
            width=width,
        )
    )
    ps_path = prediction_set_path(tmp_path, "sample")
    save_prediction_set(ps_path, ps)

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
                        "sample_id": "sample",
                        "image": str(image_path),
                        "instance_prediction_set": str(ps_path),
                        "gt_txt": str(gt_txt),
                        "gt_origin": "patch_stem",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    samples = collect_manifest_samples(manifest_path)
    report = evaluate_instance_samples(
        samples,
        model_type="yolo",
        variant="PPL",
        unit="patch",
    )
    assert report["samples"][0]["pq"] == 1.0


def test_evaluate_instances_from_unet_prediction_set_manifest(tmp_path: Path) -> None:
    """INTENT: U-Net prediction-set manifest evaluation yields unit PQ for matching GT and predictions."""
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
    pred_map = yolo_seg_labels_to_instance_map(
        gt_txt, image_width=width, image_height=height
    )
    ps = build_unet_prediction_set_from_instance_map(pred_map)
    ps_path = prediction_set_path(tmp_path, "sample")
    save_prediction_set(ps_path, ps)

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
                        "sample_id": "sample",
                        "image": str(image_path),
                        "instance_prediction_set": str(ps_path),
                        "gt_txt": str(gt_txt),
                        "gt_origin": "patch_stem",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    samples = collect_manifest_samples(manifest_path)
    report = evaluate_instance_samples(
        samples,
        model_type="unet",
        variant="PPL",
        unit="patch",
    )
    assert report["samples"][0]["pq"] == 1.0
