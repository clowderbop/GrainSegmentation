from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from common.coco_annotations import build_gt_annotations  # noqa: F401


def _ensure_dt_bbox_area(record: dict[str, Any], *, height: int, width: int) -> None:
    if record.get("bbox") and record.get("area"):
        return
    seg = record.get("segmentation")
    if seg is None or seg == [] or seg == {}:
        return
    if isinstance(seg, dict):
        record["area"] = float(mask_utils.area(seg))
        record["bbox"] = [float(x) for x in mask_utils.toBbox(seg)]
        return
    rles = mask_utils.frPyObjects(seg, height, width)
    rle = (
        mask_utils.merge(rles)
        if isinstance(rles, list) and len(rles) > 1
        else (rles[0] if isinstance(rles, list) else rles)
    )
    record["area"] = float(mask_utils.area(rle))
    record["bbox"] = [float(x) for x in mask_utils.toBbox(rle)]


def object_predictions_to_coco_dt(
    predictions: list[Any],
    *,
    image_id: int,
    height: int,
    width: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pred in predictions:
        coco_p = pred.to_coco_prediction(image_id=image_id)
        record = dict(coco_p.json)
        record["image_id"] = image_id
        cid = int(record.get("category_id", 0))
        record["category_id"] = cid + 1 if cid == 0 else cid
        seg = record.get("segmentation")
        if seg is None or seg == [] or seg == {}:
            continue
        _ensure_dt_bbox_area(record, height=height, width=width)
        out.append(record)
    return out


@dataclass
class InstanceAPSummary:
    ap_50_95: float
    ap_50: float
    ap_75: float
    ap_small: float
    ap_medium: float
    ap_large: float
    ar_1: float
    ar_10: float
    ar_100: float
    raw_stats: np.ndarray | None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "AP": float(self.ap_50_95),
            "AP50": float(self.ap_50),
            "AP75": float(self.ap_75),
            "APs": float(self.ap_small),
            "APm": float(self.ap_medium),
            "APl": float(self.ap_large),
            "AR1": float(self.ar_1),
            "AR10": float(self.ar_10),
            "AR100": float(self.ar_100),
        }
        if self.raw_stats is not None:
            d["coco_stats"] = self.raw_stats.tolist()
        return d


def evaluate_mask_ap(
    *,
    image_id: int,
    file_name: str,
    height: int,
    width: int,
    gt_annotations: list[dict[str, Any]],
    dt_annotations: list[dict[str, Any]],
    category_id: int = 1,
    category_name: str = "grain",
) -> InstanceAPSummary:
    if not gt_annotations:
        return InstanceAPSummary(
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            -1.0,
            None,
        )

    categories = [{"id": category_id, "name": category_name}]
    images = [
        {"id": image_id, "width": width, "height": height, "file_name": file_name}
    ]
    dataset = {
        "images": images,
        "annotations": gt_annotations,
        "categories": categories,
    }
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        coco_gt = COCO()
        coco_gt.dataset = dataset
        coco_gt.createIndex()

    if not dt_annotations:
        return InstanceAPSummary(
            0.0,
            0.0,
            0.0,
            -1.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            None,
        )

    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        coco_dt = coco_gt.loadRes(dt_annotations)
        coco_eval = COCOeval(coco_gt, coco_dt, "segm")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
    return InstanceAPSummary(
        float(stats[0]),
        float(stats[1]),
        float(stats[2]),
        float(stats[3]),
        float(stats[4]),
        float(stats[5]),
        float(stats[6]),
        float(stats[7]),
        float(stats[8]),
        stats,
    )
