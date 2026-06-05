"""YOLO inference profile flows through whole and patch predict."""

from __future__ import annotations

import argparse
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from common.prediction_set import PredictionSet
from common.run_provenance import load_run_provenance
from common.test_inference import load_test_inference_recipe, sahi_overlap_ratio
from yolo import predict as predict_module


def _resolved_defaults(recipe_path: Path | None = None) -> dict[str, object]:
    args = predict_module.resolve_predict_inference_defaults(
        argparse.Namespace(
            imgsz=None,
            conf=None,
            mask_threshold=None,
            slice_height=None,
            slice_width=None,
            overlap_height_ratio=None,
            overlap_width_ratio=None,
        ),
        recipe_path=recipe_path,
    )
    return {
        "conf": args.conf,
        "mask_threshold": args.mask_threshold,
        "slice_height": args.slice_height,
        "slice_width": args.slice_width,
        "overlap_height_ratio": args.overlap_height_ratio,
        "overlap_width_ratio": args.overlap_width_ratio,
    }


def test_predict_cli_defaults_match_test_inference_recipe() -> None:
    recipe = load_test_inference_recipe()
    profile = recipe.yolo.profile
    defaults = _resolved_defaults()
    assert defaults["conf"] == recipe.yolo.conf
    assert defaults["mask_threshold"] == profile.mask_threshold
    assert defaults["slice_height"] == recipe.whole.window
    assert defaults["slice_width"] == recipe.whole.window
    overlap = sahi_overlap_ratio(window=recipe.whole.window, stride=recipe.whole.stride)
    assert defaults["overlap_height_ratio"] == overlap
    assert defaults["overlap_width_ratio"] == overlap


def test_resolve_predict_defaults_read_recipe_at_resolve_time(tmp_path: Path) -> None:
    recipe_path = tmp_path / "test_inference.yaml"
    recipe_path.write_text(
        """whole:
  window: 1024
  stride: 512
patch:
  imgsz: 1024
yolo:
  conf: 0.11
  mask_threshold: 0.22
  postprocess_type: GREEDYNMM
  match_metric: IOS
  match_threshold: 0.33
  patch:
    batch: 16
  val:
    imgsz: 1024
    batch: 16
unet:
  whole:
    patch_size: 1024
    stride: 512
  patch:
    patch_size: 1024
    stride: 1024
    batch_size: 1
""",
        encoding="utf-8",
    )
    load_test_inference_recipe.cache_clear()
    defaults = _resolved_defaults(recipe_path)
    assert defaults["conf"] == pytest.approx(0.11)
    assert defaults["mask_threshold"] == pytest.approx(0.22)
    load_test_inference_recipe.cache_clear()


@patch("yolo.predict.collect_tiled_detector_proposals", return_value=[])
@patch("yolo.predict.load_image_for_yolo")
@patch("yolo.predict._load_whole_predict_pairs")
@patch("sahi.AutoDetectionModel.from_pretrained")
def test_whole_predict_applies_conf_and_mask_threshold_to_detector(
    mock_from_pretrained: MagicMock,
    mock_pairs: MagicMock,
    mock_load_image: MagicMock,
    mock_collect: MagicMock,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "train.tif"
    image_path.write_bytes(b"\x00" * 64)
    mock_pairs.return_value = [(image_path, "train")]
    mock_load_image.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
    weights = tmp_path / "best.pt"
    weights.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"
    mock_from_pretrained.return_value = MagicMock()

    args = Namespace(
        weights=weights,
        output_dir=out_dir,
        variant="PPL",
        imgsz=1024,
        conf=0.4,
        mask_threshold=0.6,
        device="cpu",
        slice_height=1024,
        slice_width=1024,
        overlap_height_ratio=0.5,
        overlap_width_ratio=0.5,
        manifest=None,
        image=image_path,
    )
    predict_module.run_whole_predict(args)

    mock_from_pretrained.assert_called_once()
    assert mock_from_pretrained.call_args.kwargs["confidence_threshold"] == 0.4
    assert mock_from_pretrained.call_args.kwargs["mask_threshold"] == 0.6
    mock_collect.assert_called_once()
    assert mock_collect.call_args.kwargs["mask_threshold"] == 0.6


@patch("yolo.predict.build_yolo_prediction_set_from_ultralytics")
@patch("yolo.predict.load_image_for_yolo")
@patch("ultralytics.YOLO")
def test_patch_predict_passes_conf_and_mask_threshold(
    mock_yolo_cls: MagicMock,
    mock_load_image: MagicMock,
    mock_build: MagicMock,
    tmp_path: Path,
) -> None:
    import json

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
    mock_load_image.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
    mock_result = MagicMock()
    mock_result.orig_shape = (16, 16)
    mock_result.masks = None
    mock_build.return_value = PredictionSet(
        schema_version=1,
        height=16,
        width=16,
        producer="yolo",
        detections=(),
    )
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
        conf=0.4,
        mask_threshold=0.6,
        device="cpu",
    )
    predict_module._run_patch_predict_from_manifest(args, manifest_path)

    assert mock_model.predict.call_args.kwargs["conf"] == 0.4
    assert mock_build.call_args.kwargs["mask_threshold"] == 0.6


@patch("yolo.predict._load_whole_predict_pairs", return_value=[])
@patch("sahi.AutoDetectionModel.from_pretrained")
def test_whole_run_provenance_records_inference_profile(
    mock_from_pretrained: MagicMock,
    mock_pairs: MagicMock,
    tmp_path: Path,
) -> None:
    weights = tmp_path / "best.pt"
    weights.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"
    mock_from_pretrained.return_value = MagicMock()
    args = Namespace(
        weights=weights,
        output_dir=out_dir,
        variant="PPL",
        imgsz=1024,
        conf=0.4,
        mask_threshold=0.6,
        device="cpu",
        slice_height=1024,
        slice_width=1024,
        overlap_height_ratio=0.5,
        overlap_width_ratio=0.5,
        manifest=None,
        image=None,
    )
    predict_module.run_whole_predict(args)
    provenance = load_run_provenance(out_dir)
    assert provenance["conf"] == 0.4
    assert provenance["mask_threshold"] == 0.6
    assert provenance.get("cross_tile_association_at_predict") is True


@patch("yolo.predict.prediction_set_from_tiled_proposal_records")
@patch("yolo.predict.collect_tiled_detector_proposals", return_value=[])
@patch("yolo.predict._load_whole_predict_pairs")
@patch("sahi.AutoDetectionModel.from_pretrained")
def test_whole_predict_collects_tiled_proposals_with_mask_threshold(
    mock_from_pretrained: MagicMock,
    mock_pairs: MagicMock,
    mock_collect: MagicMock,
    mock_pred_set: MagicMock,
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "train.tif"
    image_path.write_bytes(b"\x00" * 64)
    mock_pairs.return_value = [(image_path, "train")]
    mock_from_pretrained.return_value = MagicMock()
    mock_pred_set.return_value = PredictionSet(
        schema_version=1,
        height=8,
        width=8,
        producer="yolo",
        detections=(),
    )
    weights = tmp_path / "best.pt"
    weights.write_text("", encoding="utf-8")
    args = Namespace(
        weights=weights,
        output_dir=tmp_path / "out",
        variant="PPL",
        imgsz=1024,
        conf=0.4,
        mask_threshold=0.6,
        device="cpu",
        slice_height=1024,
        slice_width=1024,
        overlap_height_ratio=0.5,
        overlap_width_ratio=0.5,
        manifest=None,
        image=None,
    )
    with patch("yolo.predict.load_image_for_yolo", return_value=np.zeros((8, 8, 3), dtype=np.uint8)):
        predict_module.run_whole_predict(args)
    assert mock_collect.call_args.kwargs["mask_threshold"] == 0.6


def test_patch_run_provenance_records_inference_profile(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"schema_version":1,"variant":"PPL","unit":"patch","grainseg_root":"'
        + str(tmp_path)
        + '","path_base":"work_root","samples":[]}',
        encoding="utf-8",
    )
    weights = tmp_path / "best.pt"
    weights.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"
    args = Namespace(
        weights=weights,
        output_dir=out_dir,
        variant="PPL",
        imgsz=1024,
        conf=0.4,
        mask_threshold=0.6,
        device="cpu",
        manifest=manifest_path,
        data=None,
    )
    with patch.object(predict_module, "_run_patch_predict_from_manifest"):
        predict_module.run_patch_predict(args, data_yaml=None)
    provenance = load_run_provenance(out_dir)
    assert provenance["conf"] == 0.4
    assert provenance["mask_threshold"] == 0.6
    assert "postprocess_type" not in provenance


@patch("yolo.predict.build_yolo_prediction_set_from_ultralytics")
@patch("yolo.predict.load_image_for_yolo")
@patch("ultralytics.YOLO")
def test_main_patch_resolves_conf_and_mask_from_recipe_when_cli_omits_them(
    mock_yolo_cls: MagicMock,
    mock_load_image: MagicMock,
    mock_build: MagicMock,
    tmp_path: Path,
) -> None:
    import json

    from common.test_inference import inference_recipe_path, load_test_inference_recipe

    recipe_path = tmp_path / "test_inference.yaml"
    recipe_path.write_text(
        """whole:
  window: 1024
  stride: 512
patch:
  imgsz: 1024
yolo:
  conf: 0.11
  mask_threshold: 0.22
  postprocess_type: GREEDYNMM
  match_metric: IOS
  match_threshold: 0.5
  patch:
    batch: 16
  val:
    imgsz: 1024
    batch: 16
unet:
  whole:
    patch_size: 1024
    stride: 512
  patch:
    patch_size: 1024
    stride: 1024
    batch_size: 1
""",
        encoding="utf-8",
    )
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
    mock_load_image.return_value = np.zeros((16, 16, 3), dtype=np.uint8)
    mock_result = MagicMock()
    mock_result.orig_shape = (16, 16)
    mock_result.masks = None
    mock_build.return_value = PredictionSet(
        schema_version=1,
        height=16,
        width=16,
        producer="yolo",
        detections=(),
    )
    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_result]
    mock_yolo_cls.return_value = mock_model

    weights = tmp_path / "best.pt"
    weights.write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"

    load_test_inference_recipe.cache_clear()
    try:
        with patch(
            "common.test_inference.inference_recipe_path", return_value=recipe_path
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
    finally:
        load_test_inference_recipe.cache_clear()

    assert mock_model.predict.call_args.kwargs["conf"] == pytest.approx(0.11)
    assert mock_build.call_args.kwargs["mask_threshold"] == pytest.approx(0.22)
    provenance = load_run_provenance(out_dir)
    assert provenance["conf"] == pytest.approx(0.11)
    assert provenance["mask_threshold"] == pytest.approx(0.22)
