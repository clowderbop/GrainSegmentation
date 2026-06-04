"""Test inference recipe from config/test_inference.yaml (ADR 0003)."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.test_inference import inference_recipe_path, load_test_inference_recipe
from common.variants import repo_root


def test_inference_recipe_path_defaults_to_config_test_inference_yaml() -> None:
    assert inference_recipe_path() == repo_root() / "config" / "test_inference.yaml"


def test_load_test_inference_recipe_yolo_profile_from_yaml(tmp_path: Path) -> None:
    recipe_path = tmp_path / "test_inference.yaml"
    recipe_path.write_text(
        """
whole:
  window: 1024
  stride: 512
patch:
  imgsz: 1024
yolo:
  conf: 0.4
  mask_threshold: 0.6
  postprocess_type: NMM
  match_metric: IOU
  match_threshold: 0.7
  patch:
    batch: 8
  val:
    imgsz: 640
    batch: 4
unet:
  whole:
    patch_size: 1024
    stride: 512
  patch:
    patch_size: 1024
    stride: 1024
    batch_size: 1
""".strip(),
        encoding="utf-8",
    )
    recipe = load_test_inference_recipe(recipe_path)
    assert recipe.yolo.conf == 0.4
    profile = recipe.yolo.profile
    assert profile.mask_threshold == 0.6
    assert profile.postprocess_type == "NMM"
    assert profile.match_metric == "IOU"
    assert profile.match_threshold == 0.7


def test_parse_yolo_profile_candidate_mapping_rejects_empty_postprocess_type() -> None:
    from common.test_inference import parse_yolo_profile_candidate_mapping

    with pytest.raises(ValueError, match="postprocess_type"):
        parse_yolo_profile_candidate_mapping(
            {
                "postprocess_type": "",
                "match_metric": "IOS",
                "match_threshold": 0.5,
                "conf": 0.25,
                "mask_threshold": 0.5,
            },
            context="profile",
        )


def test_emit_shell_exports_yolo_inference_profile(capsys: pytest.CaptureFixture[str]) -> None:
    from common.test_inference import (
        TestInferenceRecipe,
        YoloInferenceProfile,
        YoloInferenceSpec,
        emit_shell_exports,
    )

    base = load_test_inference_recipe()
    recipe = TestInferenceRecipe(
        whole=base.whole,
        patch=base.patch,
        yolo=YoloInferenceSpec(
            conf=0.4,
            profile=YoloInferenceProfile(
                postprocess_type="NMM",
                match_metric="IOU",
                match_threshold=0.7,
                mask_threshold=0.6,
            ),
            patch=base.yolo.patch,
            val=base.yolo.val,
        ),
        unet=base.unet,
    )
    emit_shell_exports(recipe)
    out = capsys.readouterr().out
    assert "export YOLO_CONF=0.4" in out
    assert "export YOLO_MASK_THRESHOLD=0.6" in out
    assert "export YOLO_POSTPROCESS_TYPE=NMM" in out
    assert "export YOLO_MATCH_METRIC=IOU" in out
    assert "export YOLO_MATCH_THRESHOLD=0.7" in out
