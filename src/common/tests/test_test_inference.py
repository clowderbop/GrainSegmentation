"""Test inference recipe from config/test_inference.yaml (ADR 0003)."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.test_inference import (
    load_test_inference_recipe,
    profile_tune_candidate_from_conf,
    profile_tune_fixed_mask_threshold,
)
from common.tests.test_inference_fixtures import write_minimal_test_inference_recipe


def test_load_test_inference_recipe_yolo_profile_from_yaml(tmp_path: Path) -> None:
    """INTENT: load_test_inference_recipe parses YOLO confidence and postprocess profile fields from YAML."""
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


def test_profile_tune_candidate_from_conf_uses_recipe_mask_threshold(
    tmp_path: Path,
) -> None:
    """INTENT: profile_tune_candidate_from_conf pairs conf with the fixed recipe mask threshold only."""
    recipe_path = write_minimal_test_inference_recipe(tmp_path / "test_inference.yaml")
    recipe = load_test_inference_recipe(recipe_path)
    fixed_mask = profile_tune_fixed_mask_threshold(recipe)

    candidate = profile_tune_candidate_from_conf(0.25, recipe=recipe)
    assert candidate.conf == 0.25
    assert candidate.mask_threshold == fixed_mask
    assert not hasattr(candidate, "postprocess_type")
    assert candidate.to_dict() == {
        "conf": 0.25,
        "mask_threshold": fixed_mask,
    }


def test_emit_shell_exports_yolo_inference_profile(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """INTENT: emit_shell_exports prints export statements for YOLO inference profile environment variables."""
    from common.test_inference import (
        TestInferenceRecipe,
        YoloInferenceProfile,
        YoloInferenceSpec,
        emit_shell_exports,
    )

    base = load_test_inference_recipe(
        write_minimal_test_inference_recipe(tmp_path / "test_inference.yaml")
    )
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
