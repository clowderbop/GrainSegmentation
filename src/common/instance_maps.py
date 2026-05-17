from __future__ import annotations

from typing import Any

import numpy as np
from pycocotools import mask as mask_utils


def segmentation_to_binary_mask(
    segmentation: list | dict, height: int, width: int
) -> np.ndarray:
    if isinstance(segmentation, dict):
        return mask_utils.decode(segmentation).astype(bool)
    rles = mask_utils.frPyObjects(segmentation, height, width)
    if isinstance(rles, list):
        rle = mask_utils.merge(rles) if len(rles) > 1 else rles[0]
    else:
        rle = rles
    return mask_utils.decode(rle).astype(bool)


def gt_annotations_to_instance_map(
    gt_annotations: list[dict[str, Any]], height: int, width: int
) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.int32)
    sorted_anns = sorted(gt_annotations, key=lambda a: int(a["id"]))
    for ann in sorted_anns:
        lid = int(ann["id"])
        seg = ann["segmentation"]
        if seg is None or seg == [] or seg == {}:
            continue
        m = segmentation_to_binary_mask(seg, height, width)
        out[m] = lid
    return out


def _prediction_score(record: dict[str, Any]) -> float:
    raw = record.get("score")
    if raw is None:
        return 0.0
    return float(raw)


def dt_annotations_to_instance_map(
    dt_annotations: list[dict[str, Any]], height: int, width: int
) -> np.ndarray:
    out = np.zeros((height, width), dtype=np.int32)
    indexed = list(enumerate(dt_annotations, start=1))
    indexed.sort(key=lambda pair: _prediction_score(pair[1]))
    for label, ann in indexed:
        seg = ann.get("segmentation")
        if seg is None or seg == [] or seg == {}:
            continue
        m = segmentation_to_binary_mask(seg, height, width)
        out[m] = label
    return out
