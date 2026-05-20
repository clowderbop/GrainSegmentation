"""Tests for common.evaluate_instances."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from common.evaluate_instances import (
    InstanceEvalSample,
    collect_patch_samples,
    evaluate_instance_samples,
)
from common.yolo_seg_labels import (
    instance_label_map_to_yolo_seg_pred_label_file,
    write_yolo_seg_gt_label_file,
    YoloSegGtRow,
)


def _write_blank_image(path: Path, width: int, height: int) -> None:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    tifffile.imwrite(path, rgb, photometric="rgb")


def test_patch_discovery_strips_image_suffix(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    image_dir.mkdir()
    labels_dir.mkdir()
    _write_blank_image(image_dir / "patch001_PPL.tif", 64, 64)
    instance_label_map_to_yolo_seg_pred_label_file(
        np.ones((64, 64), dtype=np.int32),
        labels_dir / "patch001.txt",
        default_confidence=1.0,
        min_area_px=1,
    )
    samples = collect_patch_samples(
        image_dir=image_dir,
        pred_labels_dir=labels_dir,
        gt_gpkg=tmp_path / "dummy.gpkg",
        image_stem_suffix="_PPL",
    )
    assert len(samples) == 1
    assert samples[0].sample_id == "patch001"


def test_evaluate_two_synthetic_txt_maps(tmp_path: Path) -> None:
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
    pred_txt = tmp_path / "pred.txt"
    write_yolo_seg_gt_label_file(gt_txt, gt_rows, image_width=width, image_height=height)
    instance_label_map_to_yolo_seg_pred_label_file(
        gt_map, pred_txt, default_confidence=1.0, min_area_px=1
    )

    sample = InstanceEvalSample(
        sample_id="sample",
        image_path=image_path,
        pred_txt=pred_txt,
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
    pred_txt = tmp_path / "labels" / "whole.txt"
    pred_txt.parent.mkdir()
    instance_label_map_to_yolo_seg_pred_label_file(
        np.zeros((height, width), dtype=np.int32),
        pred_txt,
        default_confidence=1.0,
        min_area_px=1,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "sample_id": "whole",
                    "image": str(image_path),
                    "pred_txt": str(pred_txt),
                    "gt_txt": str(pred_txt),
                }
            ]
        ),
        encoding="utf-8",
    )
    from common.evaluate_instances import collect_manifest_samples

    samples = collect_manifest_samples(manifest)
    assert len(samples) == 1
    assert samples[0].image_path == image_path


def test_manifest_samples_wrapper_format(tmp_path: Path) -> None:
    width, height = 32, 32
    image_path = tmp_path / "whole.tif"
    _write_blank_image(image_path, width, height)
    pred_txt = tmp_path / "labels" / "whole.txt"
    pred_txt.parent.mkdir()
    instance_label_map_to_yolo_seg_pred_label_file(
        np.zeros((height, width), dtype=np.int32),
        pred_txt,
        default_confidence=1.0,
        min_area_px=1,
    )
    manifest = tmp_path / "manifest_wrapped.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "whole",
                        "image": str(image_path),
                        "pred_txt": str(pred_txt),
                        "gt_txt": str(pred_txt),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    from common.evaluate_instances import collect_manifest_samples

    samples = collect_manifest_samples(manifest)
    assert len(samples) == 1
    assert samples[0].sample_id == "whole"


def test_manifest_infers_pred_txt_from_labels_dir(tmp_path: Path) -> None:
    width, height = 32, 32
    image_path = tmp_path / "whole.tif"
    _write_blank_image(image_path, width, height)
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    pred_txt = labels_dir / "whole.txt"
    instance_label_map_to_yolo_seg_pred_label_file(
        np.zeros((height, width), dtype=np.int32),
        pred_txt,
        default_confidence=1.0,
        min_area_px=1,
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            [
                {
                    "sample_id": "whole",
                    "test_tiff": str(image_path),
                    "test_gpkg": str(tmp_path / "dummy.gpkg"),
                    "gt_origin": "whole_image",
                }
            ]
        ),
        encoding="utf-8",
    )
    from common.evaluate_instances import collect_manifest_samples

    samples = collect_manifest_samples(manifest, pred_labels_dir=labels_dir)
    assert len(samples) == 1
    assert samples[0].pred_txt == pred_txt
    assert samples[0].image_path == image_path.resolve()
