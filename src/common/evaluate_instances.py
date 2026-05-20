"""Shared instance-segmentation evaluation for YOLO-style prediction artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from common.geometry import load_image_space_polygons
from common.manifest_io import (
    load_manifest_json,
    manifest_gt_gpkg_field,
    manifest_image_field,
    resolve_manifest_path,
)
from common.ground_truth import GtOriginMode, scene_polygons_to_patch_instance_map
from common.metrics import compute_aji, compute_instance_metrics_dict
from common.reporting import build_instance_eval_report, build_sample_row, count_instances
from common.yolo_seg_labels import yolo_seg_labels_to_instance_map

Unit = Literal["patch", "whole"]
ModelType = Literal["unet", "yolo"]


@dataclass(frozen=True)
class InstanceEvalSample:
    sample_id: str
    image_path: Path
    pred_txt: Path
    gt_txt: Path | None = None
    gt_gpkg: Path | None = None
    gt_origin: GtOriginMode | None = None


def image_dimensions(image_path: Path | str) -> tuple[int, int]:
    import tifffile

    with tifffile.TiffFile(image_path) as tif:
        shape = tif.series[0].shape
        axes = tif.series[0].axes
    if len(shape) < 2:
        raise ValueError(f"Cannot determine image dimensions for {image_path}: {shape}")
    if "Y" in axes and "X" in axes:
        height = int(shape[axes.index("Y")])
        width = int(shape[axes.index("X")])
        return height, width
    return int(shape[-2]), int(shape[-1])


def _image_paths_from_dir(image_dir: Path) -> list[Path]:
    suffixes = {".tif", ".tiff"}
    return sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _sample_id_from_image_stem(stem: str, *, image_stem_suffix: str | None) -> str:
    if image_stem_suffix and stem.endswith(image_stem_suffix):
        return stem[: -len(image_stem_suffix)]
    return stem


def collect_patch_samples(
    *,
    image_dir: Path,
    pred_labels_dir: Path,
    gt_labels_dir: Path | None = None,
    gt_gpkg: Path | None = None,
    gt_origin: GtOriginMode = "patch_stem",
    image_stem_suffix: str | None = None,
) -> list[InstanceEvalSample]:
    if gt_labels_dir is None and gt_gpkg is None:
        raise ValueError("Patch evaluation requires either gt_labels_dir or gt_gpkg")
    samples: list[InstanceEvalSample] = []
    for image_path in _image_paths_from_dir(image_dir):
        sample_id = _sample_id_from_image_stem(
            image_path.stem, image_stem_suffix=image_stem_suffix
        )
        pred_txt = pred_labels_dir / f"{sample_id}.txt"
        if not pred_txt.is_file():
            raise FileNotFoundError(f"Missing prediction label file: {pred_txt}")
        gt_txt = gt_labels_dir / f"{sample_id}.txt" if gt_labels_dir is not None else None
        if gt_txt is not None and not gt_txt.is_file():
            raise FileNotFoundError(f"Missing GT label file: {gt_txt}")
        samples.append(
            InstanceEvalSample(
                sample_id=sample_id,
                image_path=image_path,
                pred_txt=pred_txt,
                gt_txt=gt_txt,
                gt_gpkg=gt_gpkg,
                gt_origin=gt_origin if gt_gpkg is not None else None,
            )
        )
    if not samples:
        raise ValueError(f"No TIFF images found in {image_dir}")
    return samples


def collect_manifest_samples(
    manifest: Path,
    *,
    pred_labels_dir: Path | None = None,
    default_gt_gpkg: Path | None = None,
) -> list[InstanceEvalSample]:
    rows: list[dict[str, str]]
    manifest = manifest.resolve()
    manifest_dir = manifest.parent
    if manifest.suffix.lower() == ".json":
        rows = load_manifest_json(manifest)
    else:
        with manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    samples: list[InstanceEvalSample] = []
    for idx, row in enumerate(rows):
        image_raw = manifest_image_field(row)
        if not image_raw:
            raise ValueError(
                f"Manifest row {idx} requires image, test_tiff, or tiff field"
            )
        image_path = resolve_manifest_path(image_raw, manifest_dir)

        sample_id = row.get("sample_id") or image_path.stem

        pred_txt_raw = row.get("pred_txt")
        if pred_txt_raw:
            pred_txt = resolve_manifest_path(pred_txt_raw, manifest_dir)
        elif pred_labels_dir is not None:
            pred_txt = pred_labels_dir / f"{sample_id}.txt"
        else:
            raise ValueError(
                f"Manifest row {idx} requires pred_txt or --pred-labels-dir"
            )

        gt_txt_raw = row.get("gt_txt") or None
        gt_gpkg_raw = manifest_gt_gpkg_field(row)
        if gt_gpkg_raw is None and default_gt_gpkg is not None:
            gt_gpkg_raw = str(default_gt_gpkg)

        gt_txt = (
            resolve_manifest_path(gt_txt_raw, manifest_dir) if gt_txt_raw else None
        )
        gt_gpkg = (
            resolve_manifest_path(gt_gpkg_raw, manifest_dir) if gt_gpkg_raw else None
        )
        if gt_txt is None and gt_gpkg is None:
            raise ValueError(
                f"Manifest row {idx} requires gt_txt, gt_gpkg, or --gt-gpkg"
            )

        gt_origin_raw = row.get("gt_origin") or "whole_image"
        if gt_origin_raw not in ("patch_stem", "whole_image"):
            raise ValueError(f"Invalid gt_origin {gt_origin_raw!r} in manifest row {idx}")
        gt_origin = cast(GtOriginMode, gt_origin_raw)

        samples.append(
            InstanceEvalSample(
                sample_id=sample_id,
                image_path=image_path,
                pred_txt=pred_txt,
                gt_txt=gt_txt,
                gt_gpkg=gt_gpkg,
                gt_origin=gt_origin if gt_gpkg is not None else None,
            )
        )
    if not samples:
        raise ValueError(f"Manifest contains no samples: {manifest}")
    return samples


def collect_single_image_sample(
    *,
    image: Path,
    pred_txt: Path,
    gt_txt: Path | None = None,
    gt_gpkg: Path | None = None,
    gt_origin: GtOriginMode = "whole_image",
    sample_id: str | None = None,
) -> list[InstanceEvalSample]:
    if gt_txt is None and gt_gpkg is None:
        raise ValueError("Single-image evaluation requires gt_txt or gt_gpkg")
    return [
        InstanceEvalSample(
            sample_id=sample_id or image.stem,
            image_path=image,
            pred_txt=pred_txt,
            gt_txt=gt_txt,
            gt_gpkg=gt_gpkg,
            gt_origin=gt_origin if gt_gpkg is not None else None,
        )
    ]


def load_gt_instance_map(
    sample: InstanceEvalSample, *, image_width: int, image_height: int
) -> np.ndarray:
    if sample.gt_txt is not None:
        return yolo_seg_labels_to_instance_map(
            sample.gt_txt, image_width=image_width, image_height=image_height
        )
    if sample.gt_gpkg is not None:
        return scene_polygons_to_patch_instance_map(
            load_image_space_polygons(sample.gt_gpkg),
            sample_id=sample.sample_id,
            height=image_height,
            width=image_width,
            gt_origin=sample.gt_origin or "whole_image",
        )
    raise ValueError(f"No GT source configured for sample {sample.sample_id}")


def load_pred_instance_map(
    sample: InstanceEvalSample, *, image_width: int, image_height: int
) -> np.ndarray:
    return yolo_seg_labels_to_instance_map(
        sample.pred_txt,
        image_width=image_width,
        image_height=image_height,
        has_confidence=True,
    )


def evaluate_instance_samples(
    samples: list[InstanceEvalSample],
    *,
    model_type: ModelType,
    variant: str | None,
    unit: Unit,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_rows: list[dict[str, Any]] = []
    for sample in samples:
        height, width = image_dimensions(sample.image_path)
        gt_map = load_gt_instance_map(sample, image_width=width, image_height=height)
        pred_map = load_pred_instance_map(sample, image_width=width, image_height=height)
        metrics = {
            "aji": float(compute_aji(gt_map, pred_map)),
            **{
                key: float(value)
                for key, value in compute_instance_metrics_dict(gt_map, pred_map).items()
            },
        }
        gt_n = count_instances(gt_map)
        pred_n = count_instances(pred_map)
        sample_rows.append(
            build_sample_row(
                sample.sample_id,
                metrics=metrics,
                gt_instances=gt_n,
                pred_instances=pred_n,
                empty_gt=gt_n == 0,
                extra={
                    "image_path": str(sample.image_path.resolve()),
                    "pred_txt": str(sample.pred_txt.resolve()),
                    **(
                        {"gt_txt": str(sample.gt_txt.resolve())}
                        if sample.gt_txt is not None
                        else {}
                    ),
                    **(
                        {
                            "gt_gpkg": str(sample.gt_gpkg.resolve()),
                            "gt_origin": sample.gt_origin,
                        }
                        if sample.gt_gpkg is not None
                        else {}
                    ),
                },
            )
        )
    return build_instance_eval_report(
        model_type=model_type,
        variant=variant,
        unit=unit,
        samples=sample_rows,
        extras=extras,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-labels-dir", type=Path)
    parser.add_argument("--image-dir", type=Path)
    parser.add_argument("--gt-labels-dir", type=Path)
    parser.add_argument("--gt-gpkg", type=Path)
    parser.add_argument("--gt-txt", type=Path)
    parser.add_argument("--gt-origin", choices=("patch_stem", "whole_image"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--pred-txt", type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--unit", choices=("patch", "whole"), default="patch"
    )
    parser.add_argument("--variant")
    parser.add_argument("--model-type", choices=("unet", "yolo"), required=True)
    parser.add_argument(
        "--image-stem-suffix",
        default=None,
        help="Strip this suffix from image stems to get sample_id (e.g. _PPL).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    default_gt_origin: GtOriginMode = args.gt_origin or (
        "patch_stem" if args.unit == "patch" else "whole_image"
    )
    if args.manifest is not None:
        if args.image_dir is not None:
            print(
                "Warning: --image-dir is ignored when --manifest is set.",
                file=sys.stderr,
            )
        samples = collect_manifest_samples(
            args.manifest,
            pred_labels_dir=args.pred_labels_dir,
            default_gt_gpkg=args.gt_gpkg,
        )
    elif args.image_dir is not None:
        if args.pred_labels_dir is None:
            raise ValueError("--image-dir requires --pred-labels-dir")
        samples = collect_patch_samples(
            image_dir=args.image_dir,
            pred_labels_dir=args.pred_labels_dir,
            gt_labels_dir=args.gt_labels_dir,
            gt_gpkg=args.gt_gpkg,
            gt_origin=default_gt_origin,
            image_stem_suffix=args.image_stem_suffix,
        )
    elif args.image is not None and args.pred_txt is not None:
        samples = collect_single_image_sample(
            image=args.image,
            pred_txt=args.pred_txt,
            gt_txt=args.gt_txt,
            gt_gpkg=args.gt_gpkg,
            gt_origin=default_gt_origin,
            sample_id=args.sample_id,
        )
    else:
        raise ValueError(
            "Provide --image-dir, --manifest, or --image with --pred-txt for pairing"
        )

    report = evaluate_instance_samples(
        samples,
        model_type=args.model_type,
        variant=args.variant,
        unit=args.unit,
        extras={"sample_count": len(samples)},
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
