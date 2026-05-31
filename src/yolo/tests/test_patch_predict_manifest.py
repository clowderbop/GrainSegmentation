"""Patch YOLO predict with staged manifest only (no train dataset YAML)."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from common.prediction_set import load_prediction_set, merge_yolo_proposals_by_score
from common.run_provenance import load_run_provenance
from common.tests.prediction_set_fixtures import (
    assert_yolo_canonical_sets_equal,
    yolo_prediction_set_from_masks,
)
from yolo import predict as predict_module


def _minimal_manifest(tmp_path: Path) -> Path:
    image_path = tmp_path / "patch001.tif"
    image_path.write_bytes(b"\x00" * 64)
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
                        "gt_txt": str(tmp_path / "patch001.txt"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


@patch("yolo.predict.load_image_for_yolo")
@patch("ultralytics.YOLO")
def test_patch_predict_manifest_skips_dataset_yaml(
    mock_yolo_cls: MagicMock,
    mock_load_image: MagicMock,
    tmp_path: Path,
) -> None:
    manifest_path = _minimal_manifest(tmp_path)
    mock_load_image.return_value = np.zeros((16, 16, 3), dtype=np.uint8)

    mock_result = MagicMock()
    mock_result.orig_shape = (16, 16)
    mock_result.masks = None
    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    mock_yolo_cls.return_value = mock_model

    weights = tmp_path / "best.pt"
    weights.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"

    args = Namespace(
        weights=weights,
        output_dir=out_dir,
        variant="PPL",
        imgsz=1024,
        conf=0.25,
        device="cpu",
    )

    with patch.object(predict_module, "_resolve_data_yaml") as mock_resolve_yaml:
        predict_module._run_patch_predict_from_manifest(args, manifest_path)
        mock_resolve_yaml.assert_not_called()

    pred_json = out_dir / "prediction_sets" / "patch001.json"
    assert pred_json.is_file()


@patch("yolo.predict.load_image_for_yolo")
@patch("ultralytics.YOLO")
def test_patch_predict_writes_score_merged_canonical_set(
    mock_yolo_cls: MagicMock,
    mock_load_image: MagicMock,
    tmp_path: Path,
) -> None:
    manifest_path = _minimal_manifest(tmp_path)
    height, width = 16, 16
    mock_load_image.return_value = np.zeros((height, width, 3), dtype=np.uint8)

    plane_low = torch.zeros((height, width), dtype=torch.float32)
    plane_low[4:12, 4:12] = 1.0
    plane_high = plane_low.clone()
    masks = MagicMock()
    masks.__len__.return_value = 2
    masks.data = [plane_low, plane_high]

    mock_result = MagicMock()
    mock_result.masks = masks
    mock_result.boxes.conf = torch.tensor([0.2, 0.9], dtype=torch.float32)
    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    mock_yolo_cls.return_value = mock_model

    weights = tmp_path / "best.pt"
    weights.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"

    args = Namespace(
        weights=weights,
        output_dir=out_dir,
        variant="PPL",
        imgsz=1024,
        conf=0.25,
        device="cpu",
    )
    predict_module._run_patch_predict_from_manifest(args, manifest_path)

    loaded = load_prediction_set(out_dir / "prediction_sets" / "patch001.json")
    proposals = yolo_prediction_set_from_masks(
        masks_hw=np.stack([plane_low.numpy(), plane_high.numpy()]),
        scores=np.array([0.2, 0.9], dtype=np.float32),
        height=height,
        width=width,
    )
    expected = merge_yolo_proposals_by_score(proposals)
    assert_yolo_canonical_sets_equal(loaded, expected)


def test_patch_run_provenance_records_score_merge_at_predict(tmp_path: Path) -> None:
    manifest_path = _minimal_manifest(tmp_path)
    weights = tmp_path / "best.pt"
    weights.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"
    args = Namespace(
        weights=weights,
        output_dir=out_dir,
        variant="PPL",
        imgsz=1024,
        conf=0.25,
        device="cpu",
        manifest=manifest_path,
        data=None,
    )
    with patch.object(predict_module, "_run_patch_predict_from_manifest"):
        predict_module.run_patch_predict(args, data_yaml=None)
    provenance = load_run_provenance(out_dir)
    assert provenance.get("score_merge_at_predict") is True


def test_main_patch_with_manifest_does_not_require_data_yaml(tmp_path: Path) -> None:
    manifest_path = _minimal_manifest(tmp_path)
    weights = tmp_path / "best.pt"
    weights.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"

    with (
        patch.object(predict_module, "_run_patch_predict_from_manifest") as mock_run,
        patch.object(predict_module, "_resolve_data_yaml") as mock_resolve_yaml,
    ):
        predict_module.main(
            [
                "--unit",
                "patch",
                "--weights",
                str(weights),
                "--output-dir",
                str(out_dir),
                "--manifest",
                str(manifest_path),
                "--variant",
                "PPL",
            ]
        )
        mock_run.assert_called_once()
        mock_resolve_yaml.assert_not_called()
