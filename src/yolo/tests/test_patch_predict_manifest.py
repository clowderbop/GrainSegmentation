"""Patch YOLO predict with staged manifest only (no train dataset YAML)."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

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
