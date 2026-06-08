"""Shared instance-segmentation evaluation for instance label-map predictions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

from common.geometry import load_image_space_polygons
from common.prediction_set import (
    load_prediction_set,
    prediction_set_path,
    prediction_set_to_merged_instance_view,
)
from common.manifest_io import (
    DatasetManifest,
    load_dataset_manifest,
    manifest_path_base_dir,
    require_eval_local_path,
    resolve_row_path,
)
from common.ground_truth import GtOriginMode, scene_polygons_to_patch_instance_map
from common.instance_metric_bundle import compute_instance_metric_bundle
from common.reporting import (
    build_instance_eval_report,
    build_sample_row,
    json_safe_for_dump,
)
from common.yolo_seg_labels import yolo_seg_labels_to_instance_map

Unit = Literal["patch", "whole"]
ModelType = Literal["unet", "yolo"]


@dataclass(frozen=True)
class InstanceEvalSample:
    sample_id: str
    image_path: Path
    instance_prediction_set: Path
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


def _instance_prediction_set_path(prediction_set_dir: Path, sample_id: str) -> Path:
    return prediction_set_path(prediction_set_dir, sample_id)


def collect_whole_samples_from_manifest(
    manifest: Path | DatasetManifest,
    *,
    prediction_set_dir: Path | None = None,
    default_gt_gpkg: Path | None = None,
) -> list[InstanceEvalSample]:
    """Load eval samples from a manifest (whole or patch units)."""
    return collect_manifest_samples(
        manifest,
        prediction_set_dir=prediction_set_dir,
        default_gt_gpkg=default_gt_gpkg,
    )


def collect_manifest_samples(
    manifest: Path | DatasetManifest,
    *,
    prediction_set_dir: Path | None = None,
    default_gt_gpkg: Path | None = None,
) -> list[InstanceEvalSample]:
    doc = (
        manifest
        if isinstance(manifest, DatasetManifest)
        else load_dataset_manifest(manifest)
    )

    samples: list[InstanceEvalSample] = []
    for idx, row in enumerate(doc.samples):
        if row.image is not None:
            image_path = resolve_row_path(doc, row.image)
        else:
            image_path = resolve_row_path(doc, row.anchor_image_path())
        assert image_path is not None

        if row.instance_prediction_set:
            instance_prediction_set = resolve_row_path(doc, row.instance_prediction_set)
        elif prediction_set_dir is not None:
            instance_prediction_set = _instance_prediction_set_path(
                prediction_set_dir, row.sample_id
            )
        else:
            raise ValueError(
                f'Manifest samples[{idx}] requires "instance_prediction_set" or '
                "--prediction-set-dir"
            )
        assert instance_prediction_set is not None

        gt_txt = resolve_row_path(doc, row.gt_txt)
        gt_gpkg = resolve_row_path(doc, row.gt_gpkg)
        if gt_gpkg is None and default_gt_gpkg is not None:
            gt_gpkg = default_gt_gpkg.resolve()

        if doc.path_base == "work_root":
            work_base = manifest_path_base_dir(doc)
            image_path = require_eval_local_path(image_path, work_base)
            if gt_txt is not None:
                gt_txt = require_eval_local_path(gt_txt, work_base)
            if gt_gpkg is not None:
                gt_gpkg = require_eval_local_path(gt_gpkg, work_base)

        if gt_txt is None and gt_gpkg is None:
            raise ValueError(
                f"Manifest row {idx} requires gt_txt, gt_gpkg, or --gt-gpkg"
            )

        gt_origin_raw = row.gt_origin or "whole_image"
        if gt_origin_raw not in ("patch_stem", "whole_image"):
            raise ValueError(
                f"Invalid gt_origin {gt_origin_raw!r} in manifest row {idx}"
            )
        gt_origin = cast(GtOriginMode, gt_origin_raw)

        samples.append(
            InstanceEvalSample(
                sample_id=row.sample_id,
                image_path=image_path,
                instance_prediction_set=instance_prediction_set,
                gt_txt=gt_txt,
                gt_gpkg=gt_gpkg,
                gt_origin=gt_origin if gt_gpkg is not None else None,
            )
        )
    if not samples:
        raise ValueError("Manifest contains no samples")
    return samples


def collect_single_image_sample(
    *,
    image: Path,
    instance_prediction_set: Path,
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
            instance_prediction_set=instance_prediction_set,
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


def load_pred_instance_map(sample: InstanceEvalSample) -> np.ndarray:
    prediction_set = load_prediction_set(sample.instance_prediction_set)
    return prediction_set_to_merged_instance_view(prediction_set)


def evaluate_instance_samples(
    samples: list[InstanceEvalSample],
    *,
    model_type: ModelType,
    variant: str | None,
    unit: Unit,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_rows: list[dict[str, Any]] = []
    n_samples = len(samples)
    for idx, sample in enumerate(samples):
        print(f"Evaluating {sample.sample_id} ({idx + 1}/{n_samples})...")
        height, width = image_dimensions(sample.image_path)
        gt_map = load_gt_instance_map(sample, image_width=width, image_height=height)
        pred_map = load_pred_instance_map(sample)
        if pred_map.shape != (height, width):
            raise ValueError(
                f"Prediction map shape {pred_map.shape} does not match image "
                f"({height}, {width}) for {sample.sample_id}"
            )
        metrics = compute_instance_metric_bundle(gt_map, pred_map)
        gt_n = int(metrics["gt_instance_count"])
        pred_n = int(metrics["pred_instance_count"])
        print(
            f"{sample.sample_id}: pq={metrics['pq']:.4f} "
            f"dq={metrics['dq']:.4f} sq={metrics['sq']:.4f} "
            f"gt={gt_n} pred={pred_n}"
        )
        sample_rows.append(
            build_sample_row(
                sample.sample_id,
                metrics=metrics,
                empty_gt=gt_n == 0,
                extra={
                    "image_path": str(sample.image_path.resolve()),
                    "instance_prediction_set_path": str(
                        sample.instance_prediction_set.resolve()
                    ),
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
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Dataset manifest listing samples (required unless --image is set).",
    )
    parser.add_argument(
        "--prediction-set-dir",
        type=Path,
        help=(
            "Run output directory containing prediction_sets/ when manifest rows "
            "omit instance_prediction_set."
        ),
    )
    parser.add_argument("--gt-gpkg", type=Path)
    parser.add_argument("--gt-txt", type=Path)
    parser.add_argument("--gt-origin", choices=("patch_stem", "whole_image"))
    parser.add_argument("--image", type=Path)
    parser.add_argument("--instance-prediction-set", type=Path)
    parser.add_argument("--sample-id")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--unit", choices=("patch", "whole"), default="patch")
    parser.add_argument("--variant")
    parser.add_argument("--model-type", choices=("unet", "yolo"), required=True)
    return parser.parse_args()


def _resolve_eval_samples(args: argparse.Namespace) -> list[InstanceEvalSample]:
    default_gt_origin: GtOriginMode = args.gt_origin or (
        "patch_stem" if args.unit == "patch" else "whole_image"
    )

    if args.manifest is not None:
        return collect_whole_samples_from_manifest(
            args.manifest,
            prediction_set_dir=args.prediction_set_dir,
            default_gt_gpkg=args.gt_gpkg,
        )
    if args.image is not None and args.instance_prediction_set is not None:
        return collect_single_image_sample(
            image=args.image,
            instance_prediction_set=args.instance_prediction_set,
            gt_txt=args.gt_txt,
            gt_gpkg=args.gt_gpkg,
            gt_origin=default_gt_origin,
            sample_id=args.sample_id,
        )
    raise ValueError("Provide --manifest or --image with --instance-prediction-set")


def main() -> None:
    args = _parse_args()
    samples = _resolve_eval_samples(args)
    print(
        f"Instance evaluation: {len(samples)} sample(s), "
        f"unit={args.unit}, model_type={args.model_type}"
    )

    report = evaluate_instance_samples(
        samples,
        model_type=args.model_type,
        variant=args.variant,
        unit=args.unit,
        extras={"sample_count": len(samples)},
    )
    mean = report.get("mean")
    if mean is not None:
        print(f"mean: pq={mean['pq']:.4f} aji_plus={mean['aji_plus']:.4f}")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(json_safe_for_dump(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote instance metrics to {args.output_json}")


if __name__ == "__main__":
    main()
