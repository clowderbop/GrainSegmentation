"""Minimal test inference recipe YAML for tmp_path fixtures."""

from __future__ import annotations

from pathlib import Path

MINIMAL_TEST_INFERENCE_YAML = """whole:
  window: 1024
  stride: 512
patch:
  imgsz: 1024
yolo:
  conf: 0.25
  mask_threshold: 0.55
  postprocess_type: GREEDYNMM
  match_metric: IOU
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
"""


def write_minimal_test_inference_recipe(path: Path) -> Path:
    path.write_text(MINIMAL_TEST_INFERENCE_YAML, encoding="utf-8")
    return path
