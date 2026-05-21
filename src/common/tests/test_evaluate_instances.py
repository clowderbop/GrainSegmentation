"""Tests for common.evaluate_instances."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from common.evaluate_instances import (
    InstanceEvalSample,
    collect_manifest_samples,
    collect_patch_samples,
    evaluate_instance_samples,
)
from common.instance_predictions import instance_map_path, write_instance_map_tiff
from common.yolo_seg_labels import (
    YoloSegGtRow,
    write_yolo_seg_gt_label_file,
)


def _write_blank_image(path: Path, width: int, height: int) -> None:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    tifffile.imwrite(path, rgb, photometric="rgb")


def test_patch_discovery_strips_image_suffix(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    instances_dir = tmp_path / "instances"
    image_dir.mkdir()
    instances_dir.mkdir()
    _write_blank_image(image_dir / "patch001_PPL.tif", 64, 64)
    write_instance_map_tiff(
        instance_map_path(tmp_path, "patch001"),
        np.ones((64, 64), dtype=np.int32),
    )
    samples = collect_patch_samples(
        image_dir=image_dir,
        pred_instances_dir=instances_dir,
        gt_gpkg=tmp_path / "dummy.gpkg",
        image_stem_suffix="_PPL",
    )
    assert len(samples) == 1
    assert samples[0].sample_id == "patch001"


def test_evaluate_two_synthetic_instance_maps(tmp_path: Path) -> None:
    width, height = 48, 48
    image_path = tmp_path / "sample_PPL.tif"
    _write_blank_image(image_path, width, height)

    gt_map = np.zeros((height, width), dtype=np.int32)
    gt_map[4:20, 4:20] = 1

    gt_rows = [
        YoloSegGtRow(
            class_id=0,
            points=np.array(
                [[4.0, 4.0], [20.0, 4.0], [20.0, 20.0], [4.0, 20.0]],
                dtype=np.float32,
            ),
        )
    ]
    gt_txt = tmp_path / "sample.txt"
    pred_path = instance_map_path(tmp_path, "sample")
    write_yolo_seg_gt_label_file(gt_txt, gt_rows, image_width=width, image_height=height)
    write_instance_map_tiff(pred_path, gt_map)

    sample = InstanceEvalSample(
        sample_id="sample",
        image_path=image_path,
        pred_instances=pred_path,
        gt_txt=gt_txt,
    )
    report = evaluate_instance_samples(
        [sample],
        model_type="yolo",
        variant="test",
        unit="patch",
    )
    assert report["samples"][0]["aji"] == 1.0


def test_manifest_mode_pairs_records(tmp_path: Path) -> None:
    width, height = 32, 32
    image_path = tmp_path / "whole.tif"
    _write_blank_image(image_path, width, height)
    pred_path = instance_map_path(tmp_path, "whole")
    write_instance_map_tiff(pred_path, np.zeros((height, width), dtype=np.int32))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "whole",
                        "image": str(image_path),
                        "pred_instances": str(pred_path),
                        "gt_txt": str(pred_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    samples = collect_manifest_samples(manifest)
    assert len(samples) == 1
    assert samples[0].image_path == image_path


def test_manifest_infers_pred_instances_from_dir(tmp_path: Path) -> None:
    width, height = 32, 32
    image_path = tmp_path / "whole.tif"
    _write_blank_image(image_path, width, height)
    instances_dir = tmp_path / "instances"
    instances_dir.mkdir()
    pred_path = instance_map_path(tmp_path, "whole")
    write_instance_map_tiff(pred_path, np.zeros((height, width), dtype=np.int32))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "whole",
                        "image": str(image_path),
                        "gt_gpkg": str(tmp_path / "dummy.gpkg"),
                        "gt_origin": "whole_image",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    samples = collect_manifest_samples(manifest, pred_instances_dir=instances_dir)
    assert len(samples) == 1
    assert samples[0].pred_instances == pred_path
    assert samples[0].image_path == image_path.resolve()
