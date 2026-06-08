"""OpenCV merged instance view for vector ground truth (ADR 0005).

Covers ``evaluate_instances`` / ``ground_truth`` paths. Watershed tune wiring is
tested in ``unet.tests.test_tune_watershed_gpkg_gt``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tifffile

from common.evaluate_instances import InstanceEvalSample, load_gt_instance_map
from common.geometry import load_image_space_polygons
from common.ground_truth import scene_polygons_to_patch_instance_map

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "gpkg_merged_instance_map"
_MICRO_GPKG = _FIXTURES / "micro_labels.gpkg"
_GOLDEN_NPZ = _FIXTURES / "instance_map.npz"
_FIXTURE_HEIGHT = 48
_FIXTURE_WIDTH = 64


def _golden_map() -> np.ndarray:
    with np.load(_GOLDEN_NPZ) as data:
        return np.asarray(data["instance_map"], dtype=np.int32)


def _blank_image(path: Path) -> None:
    tifffile.imwrite(
        path,
        np.zeros((_FIXTURE_HEIGHT, _FIXTURE_WIDTH, 3), dtype=np.uint8),
        photometric="rgb",
    )


def test_load_gt_instance_map_whole_gpkg_matches_golden(tmp_path: Path) -> None:
    """INTENT: load_gt_instance_map rasterizes a whole-section GPKG to the golden instance map."""
    image_path = tmp_path / "train_PPL.tif"
    _blank_image(image_path)
    sample = InstanceEvalSample(
        sample_id="train",
        image_path=image_path,
        instance_prediction_set=image_path,
        gt_gpkg=_MICRO_GPKG,
        gt_origin="whole_image",
    )
    loaded = load_gt_instance_map(
        sample, image_width=_FIXTURE_WIDTH, image_height=_FIXTURE_HEIGHT
    )
    assert np.array_equal(loaded, _golden_map())


def test_scene_polygons_patch_origin_matches_golden_subregion() -> None:
    """INTENT: scene_polygons_to_patch_instance_map matches the golden whole-section crop for patch_stem origin."""
    golden = _golden_map()
    patch_id = "region_0000_y00008_x00016"
    y0, x0 = 8, 16
    patch_h, patch_w = 24, 32
    polygons = load_image_space_polygons(_MICRO_GPKG)
    patch_map = scene_polygons_to_patch_instance_map(
        polygons,
        sample_id=patch_id,
        height=patch_h,
        width=patch_w,
        gt_origin="patch_stem",
    )
    golden_crop = golden[y0 : y0 + patch_h, x0 : x0 + patch_w]
    assert patch_map.shape == golden_crop.shape
    assert np.array_equal(patch_map > 0, golden_crop > 0)
