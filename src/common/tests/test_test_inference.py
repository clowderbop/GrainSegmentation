"""Test inference recipe from configs/test_inference.yaml (ADR 0003)."""

from __future__ import annotations

from common.test_inference import inference_recipe_path, load_test_inference_recipe
from common.variants import repo_root


def test_inference_recipe_path_under_repo_root() -> None:
    assert inference_recipe_path() == repo_root() / "configs" / "test_inference.yaml"


def test_load_test_inference_recipe_values() -> None:
    recipe = load_test_inference_recipe()
    assert recipe.whole.window == 1024
    assert recipe.whole.stride == 512
    assert recipe.patch.imgsz == 1024
    assert recipe.yolo.conf == 0.25
    assert recipe.yolo.val.imgsz == 1024
    assert recipe.yolo.val.batch == 16
    assert recipe.yolo.patch.batch == 16
    assert recipe.unet.whole.patch_size == 1024
    assert recipe.unet.whole.stride == 512
    assert recipe.unet.patch.patch_size == 1024
    assert recipe.unet.patch.stride == 1024
    assert recipe.unet.patch.batch_size == 1
