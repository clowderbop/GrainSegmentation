"""Evaluate COCO mask AP from YOLO mask NPZ predictions and GPKG ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from common.geometry import load_image_space_polygons
from common.instance_predictions import MASKS_SUBDIR, yolo_mask_npz_path, yolo_mask_npz_to_coco_dt
from common.manifest_io import (
    load_manifest_json,
    manifest_gt_gpkg_field,
    manifest_image_field,
    resolve_manifest_path,
)
from common.reporting import json_safe_for_dump
from yolo.coco_instance_ap import build_gt_annotations, evaluate_mask_ap
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
        manifest_path = args.manifest.resolve()
        manifest_dir = manifest_path.parent
        pairs: list[tuple[Path, Path, str]] = []
        for index, entry in enumerate(load_manifest_json(manifest_path)):
            tiff = manifest_image_field(entry)
            gpkg = manifest_gt_gpkg_field(entry)
            sample_id = str(entry.get("sample_id") or "")
            if not tiff or not gpkg:
                raise ValueError(f"manifest[{index}] needs tiff and gpkg")
            tiff_path = resolve_manifest_path(tiff, manifest_dir)
            gpkg_path = resolve_manifest_path(gpkg, manifest_dir)
            if not sample_id:
                sample_id = tiff_path.stem
            pairs.append((tiff_path, gpkg_path, sample_id))
        return pairs
    if args.test_tiff is None or args.test_gpkg is None:
        raise ValueError("Provide --manifest or both --test-tiff and --test-gpkg")
    tiff = args.test_tiff.resolve()
    return [(tiff, args.test_gpkg.resolve(), tiff.stem)]


def run_evaluate_mask_ap(args: argparse.Namespace) -> dict[str, Any]:
    pred_dir = args.pred_dir.resolve()
    pairs = _resolve_manifest_pairs(args)
    sample_rows: list[dict[str, Any]] = []

    for image_id, (tiff_path, gpkg_path, sample_id) in enumerate(pairs, start=1):
        if not tiff_path.is_file():
            raise FileNotFoundError(f"test TIFF not found: {tiff_path}")
        if not gpkg_path.is_file():
            raise FileNotFoundError(f"test GPKG not found: {gpkg_path}")

        image = load_image_for_yolo(tiff_path)
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
            file_name=tiff_path.name,
            height=height,
            width=width,
            gt_annotations=gt_anns,
            dt_annotations=dt_anns,
        )
        row = {
            "sample_id": sample_id,
            "test_tiff": str(tiff_path),
            "test_gpkg": str(gpkg_path),
            "pred_npz": str(pred_npz),
            "coco_mask_ap": summary.to_dict(),
        }
        sample_rows.append(row)
        print(
            f"{tiff_path.name}: AP={summary.ap_50_95:.4f} AP50={summary.ap_50:.4f} "
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
    parser.add_argument("--test-tiff", default=None, type=Path)
    parser.add_argument("--test-gpkg", default=None, type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.manifest is None and (args.test_tiff is None or args.test_gpkg is None):
        parser.error("Provide --manifest or both --test-tiff and --test-gpkg")

    report = run_evaluate_mask_ap(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(json_safe_for_dump(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote mask AP metrics to {args.output_json}")


if __name__ == "__main__":
    main()
