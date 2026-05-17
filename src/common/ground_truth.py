from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from shapely.affinity import translate

from common.coco_annotations import build_gt_annotations
from common.geometry import load_image_space_polygons
from common.instance_maps import gt_annotations_to_instance_map
from common.patching import sample_origin_xy, sample_origin_xy_or_whole_image

GtOriginMode = Literal["patch_stem", "whole_image"]


def polygons_to_instance_map(
    polygons: list[Any],
    *,
    height: int,
    width: int,
    image_id: int = 1,
) -> np.ndarray:
    gt_anns = build_gt_annotations(
        polygons,
        image_id=image_id,
        height=height,
        width=width,
    )
    return gt_annotations_to_instance_map(gt_anns, height, width)


def scene_polygons_to_patch_instance_map(
    polygons: list[Any],
    *,
    sample_id: str,
    height: int,
    width: int,
    gt_origin: GtOriginMode,
    image_id: int = 1,
) -> np.ndarray:
    """Translate GIS polygons into patch image space, then rasterize to label ids."""
    if gt_origin == "whole_image":
        origin_x, origin_y = sample_origin_xy_or_whole_image(sample_id)
    else:
        origin_x, origin_y = sample_origin_xy(sample_id)
    if origin_x or origin_y:
        polygons = [
            translate(p, xoff=-float(origin_x), yoff=-float(origin_y)) for p in polygons
        ]
    gt_anns = build_gt_annotations(
        polygons,
        image_id=image_id,
        height=height,
        width=width,
    )
    return gt_annotations_to_instance_map(gt_anns, height, width)


def gpkg_to_scene_instance_map(
    gpkg_path: Path,
    *,
    height: int,
    width: int,
    image_id: int = 1,
) -> np.ndarray:
    """Rasterize a GIS layer at scene origin (full-section coordinates)."""
    return polygons_to_instance_map(
        load_image_space_polygons(gpkg_path),
        height=height,
        width=width,
        image_id=image_id,
    )
