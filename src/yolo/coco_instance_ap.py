from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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
