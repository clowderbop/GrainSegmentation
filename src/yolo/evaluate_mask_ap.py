"""Evaluate COCO mask AP from YOLO instance prediction sets and GPKG ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common.geometry import load_image_space_polygons
from common.coco_annotations import build_gt_annotations
from common.manifest_io import (
    load_dataset_manifest,
    manifest_path_base_dir,
    require_eval_local_path,
    resolve_row_path,
)
from common.prediction_set import (
    load_prediction_set,
    prediction_set_path,
    yolo_prediction_set_to_coco_dt,
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


def _resolve_manifest_pairs(
    args: argparse.Namespace,
) -> list[tuple[Path, Path, str, Path]]:
    prediction_set_dir = (
        args.prediction_set_dir.resolve()
        if args.prediction_set_dir is not None
        else None
    )
    if args.manifest is not None:
        doc = load_dataset_manifest(args.manifest)
        pairs: list[tuple[Path, Path, str, Path]] = []
        for index, row in enumerate(doc.samples):
            if row.image is None:
                raise ValueError(f'manifest samples[{index}] requires "image"')
            if row.gt_gpkg is None:
                raise ValueError(f'manifest samples[{index}] requires "gt_gpkg"')
            image_path = resolve_row_path(doc, row.image)
            gpkg_path = resolve_row_path(doc, row.gt_gpkg)
            assert image_path is not None and gpkg_path is not None
            if doc.path_base == "work_root":
                work_base = manifest_path_base_dir(doc)
                image_path = require_eval_local_path(image_path, work_base)
                gpkg_path = require_eval_local_path(gpkg_path, work_base)
            if row.instance_prediction_set:
                pred_path = resolve_row_path(doc, row.instance_prediction_set)
            elif prediction_set_dir is not None:
                pred_path = prediction_set_path(prediction_set_dir, row.sample_id)
            else:
                raise ValueError(
                    f'manifest samples[{index}] requires "instance_prediction_set" or '
                    "--prediction-set-dir"
                )
            assert pred_path is not None
            pairs.append((image_path, gpkg_path, row.sample_id, pred_path))
        return pairs
    if args.image is None or args.gt_gpkg is None:
        raise ValueError("Provide --manifest or both --image and --gt-gpkg")
    image_path = args.image.resolve()
    if args.instance_prediction_set is not None:
        pred_path = args.instance_prediction_set.resolve()
    elif prediction_set_dir is not None:
        pred_path = prediction_set_path(prediction_set_dir, image_path.stem)
    else:
        raise ValueError(
            "Single-sample mode requires --instance-prediction-set or --prediction-set-dir"
        )
    return [(image_path, args.gt_gpkg.resolve(), image_path.stem, pred_path)]


def run_evaluate_mask_ap(args: argparse.Namespace) -> dict[str, Any]:
    pairs = _resolve_manifest_pairs(args)
    n_samples = len(pairs)
    print(f"COCO mask AP evaluation: {n_samples} sample(s), variant={args.variant}")
    sample_rows: list[dict[str, Any]] = []

    for image_id, (image_path, gpkg_path, sample_id, pred_path) in enumerate(
        pairs, start=1
    ):
        print(f"Evaluating {sample_id} ({image_id}/{n_samples})...")
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not gpkg_path.is_file():
            raise FileNotFoundError(f"GPKG not found: {gpkg_path}")
        if not pred_path.is_file():
            raise FileNotFoundError(f"Prediction set not found: {pred_path}")

        image = load_image_for_yolo(image_path)
        height, width = image.shape[:2]
        polygons = load_image_space_polygons(gpkg_path)
        gt_anns = build_gt_annotations(
            polygons,
            image_id=image_id,
            height=height,
            width=width,
        )
        prediction_set = load_prediction_set(pred_path)
        dt_anns = yolo_prediction_set_to_coco_dt(
            prediction_set, image_id=image_id, height=height, width=width
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
            "instance_prediction_set": str(pred_path),
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
            "prediction_set_input": True,
        },
    }
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="COCO mask AP from instance prediction sets and GPKG GT.",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--variant", choices=variant_choices(), default=None)
    parser.add_argument("--manifest", default=None, type=Path)
    parser.add_argument("--prediction-set-dir", default=None, type=Path)
    parser.add_argument("--instance-prediction-set", default=None, type=Path)
    parser.add_argument("--image", default=None, type=Path)
    parser.add_argument("--gt-gpkg", default=None, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.manifest is None and (args.image is None or args.gt_gpkg is None):
        parser.error("Provide --manifest or both --image and --gt-gpkg")

    report = run_evaluate_mask_ap(args)
    mean_ap = report.get("mean_coco_mask_ap") or {}
    mean_ap_val = mean_ap.get("mean_AP")
    if mean_ap_val is not None:
        mean_ap50 = mean_ap.get("mean_AP50")
        ap50_str = f"{float(mean_ap50):.4f}" if mean_ap50 is not None else "n/a"
        print(f"mean: AP={float(mean_ap_val):.4f} AP50={ap50_str}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(json_safe_for_dump(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote mask AP metrics to {args.output_json}")


if __name__ == "__main__":
    main()
