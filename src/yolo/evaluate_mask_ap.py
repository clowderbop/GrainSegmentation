"""Evaluate COCO mask AP from YOLO mask NPZ predictions and GPKG ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common.geometry import load_image_space_polygons
from common.instance_predictions import MASKS_SUBDIR, yolo_mask_npz_path, yolo_mask_npz_to_coco_dt
from common.coco_annotations import build_gt_annotations
from common.manifest_io import (
    load_dataset_manifest,
    resolve_row_path,
)
from common.reporting import json_safe_for_dump
from yolo.coco_instance_ap import evaluate_mask_ap
from yolo.config import variant_choices
from yolo.predict import load_image_for_yolo


def _aggregate_coco_mask_ap_means(
    rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    coco_mean_keys = (
        "AP",
        "AP50",
        "AP75",
        "APs",
        "APm",
        "APl",
        "AR1",
        "AR10",
        "AR100",
    )
    out: dict[str, float | None] = {}
    for key in coco_mean_keys:
        values: list[float] = []
        for row in rows:
            block = row.get("coco_mask_ap")
            if not isinstance(block, dict) or key not in block:
                continue
            v = float(block[key])
            if np.isfinite(v) and v >= 0:
                values.append(v)
        out[f"mean_{key}"] = float(np.mean(values)) if values else None
    return out


def _resolve_manifest_pairs(args: argparse.Namespace) -> list[tuple[Path, Path, str]]:
    if args.manifest is not None:
        doc = load_dataset_manifest(args.manifest)
        pairs: list[tuple[Path, Path, str]] = []
        for index, row in enumerate(doc.samples):
            if row.image is None:
                raise ValueError(f'manifest samples[{index}] requires "image"')
            if row.gt_gpkg is None:
                raise ValueError(f'manifest samples[{index}] requires "gt_gpkg"')
            image_path = resolve_row_path(doc, row.image)
            gpkg_path = resolve_row_path(doc, row.gt_gpkg)
            assert image_path is not None and gpkg_path is not None
            pairs.append((image_path, gpkg_path, row.sample_id))
        return pairs
    if args.image is None or args.gt_gpkg is None:
        raise ValueError("Provide --manifest or both --image and --gt-gpkg")
    image_path = args.image.resolve()
    return [(image_path, args.gt_gpkg.resolve(), image_path.stem)]


def run_evaluate_mask_ap(args: argparse.Namespace) -> dict[str, Any]:
    pred_dir = args.pred_dir.resolve()
    pairs = _resolve_manifest_pairs(args)
    sample_rows: list[dict[str, Any]] = []

    for image_id, (image_path, gpkg_path, sample_id) in enumerate(pairs, start=1):
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not gpkg_path.is_file():
            raise FileNotFoundError(f"GPKG not found: {gpkg_path}")

        image = load_image_for_yolo(image_path)
        height, width = image.shape[:2]
        polygons = load_image_space_polygons(gpkg_path)
        gt_anns = build_gt_annotations(
            polygons,
            image_id=image_id,
            height=height,
            width=width,
        )
        pred_npz = yolo_mask_npz_path(pred_dir, sample_id)
        if not pred_npz.is_file():
            raise FileNotFoundError(f"Prediction masks NPZ not found: {pred_npz}")
        dt_anns = yolo_mask_npz_to_coco_dt(
            pred_npz, image_id=image_id, height=height, width=width
        )
        summary = evaluate_mask_ap(
            image_id=image_id,
            file_name=image_path.name,
            height=height,
            width=width,
            gt_annotations=gt_anns,
            dt_annotations=dt_anns,
        )
        row = {
            "sample_id": sample_id,
            "image": str(image_path),
            "gt_gpkg": str(gpkg_path),
            "pred_npz": str(pred_npz),
            "coco_mask_ap": summary.to_dict(),
        }
        sample_rows.append(row)
        print(
            f"{image_path.name}: AP={summary.ap_50_95:.4f} AP50={summary.ap_50:.4f} "
            f"Pred={len(dt_anns)} GT={len(gt_anns)}"
        )

    report = {
        "schema_version": 1,
        "model_type": "yolo",
        "metric_kind": "coco_mask_ap",
        "variant": args.variant,
        "samples": sample_rows,
        "mean_coco_mask_ap": _aggregate_coco_mask_ap_means(sample_rows),
        "extras": {
            "pred_dir": str(pred_dir),
            "pred_masks_subdir": MASKS_SUBDIR,
            "confidence_from_mask_npz": True,
        },
    }
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="COCO mask AP from prediction mask NPZ archives and GPKG GT.",
    )
    parser.add_argument(
        "--pred-dir",
        required=True,
        type=Path,
        help="YOLO predict output root containing masks/{sample_id}.npz",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--variant", choices=variant_choices(), default=None)
    parser.add_argument("--manifest", default=None, type=Path)
    parser.add_argument("--image", default=None, type=Path)
    parser.add_argument("--gt-gpkg", default=None, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.manifest is None and (args.image is None or args.gt_gpkg is None):
        parser.error("Provide --manifest or both --image and --gt-gpkg")

    report = run_evaluate_mask_ap(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(json_safe_for_dump(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote mask AP metrics to {args.output_json}")


if __name__ == "__main__":
    main()
