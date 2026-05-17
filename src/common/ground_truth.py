from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.geometry import load_image_space_polygons


def polygons_to_instance_map(
    polygons: list[Any],
    *,
    height: int,
    width: int,
    image_id: int = 1,
) -> np.ndarray:
    from yolo.coco_instance_ap import build_gt_annotations
    from yolo.instance_label_maps import gt_annotations_to_instance_map

    gt_anns = build_gt_annotations(
        polygons,
        image_id=image_id,
        height=height,
        width=width,
    )
    return gt_annotations_to_instance_map(gt_anns, height, width)


def gpkg_to_instance_map(
    gpkg_path: Path,
    *,
    height: int,
    width: int,
    image_id: int = 1,
) -> np.ndarray:
    return polygons_to_instance_map(
        load_image_space_polygons(gpkg_path),
        height=height,
        width=width,
        image_id=image_id,
    )
