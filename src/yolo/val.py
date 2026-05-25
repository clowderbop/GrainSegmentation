"""Run Ultralytics validation and write metrics JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common.reporting import json_safe_for_dump
from yolo.config import variant_choices
from yolo.pipeline import resolve_variant_paths
from yolo.train import _parse_device


def _optional_metric_attr(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    try:
        return getattr(obj, name)
    except Exception:
        return None


def _metric_section(obj: Any) -> dict[str, Any]:
    keys = (
        "map",
        "map50",
        "map75",
        "maps",
        "mp",
        "mr",
        "p",
        "r",
        "f1",
        "ap_class_index",
        "image_metrics",
    )
    out: dict[str, Any] = {}
    for key in keys:
        value = _optional_metric_attr(obj, key)
        if value is not None:
            out[key] = json_safe_for_dump(value)
    return out


def collect_val_metrics(metrics: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for section in ("box", "seg", "mask", "pose", "obb"):
        values = _metric_section(_optional_metric_attr(metrics, section))
        if values:
            payload[section] = values
    for key in ("speed", "results_dict", "fitness"):
        value = _optional_metric_attr(metrics, key)
        if value is not None:
            payload[key] = json_safe_for_dump(value)
    return payload


def write_val_metrics_json(
    metrics: Any, *, project: Path | None, name: str
) -> Path | None:
    if project is None:
        return None
    out_path = project.resolve() / name / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(collect_val_metrics(metrics), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Wrote val metrics JSON to {out_path}")
    return out_path


def _resolve_data_yaml(args: argparse.Namespace) -> Path:
    if args.data is not None:
        path = args.data.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {path}")
        return path
    if args.variant is None:
        raise ValueError("Provide --variant or --data")
    resolved = resolve_variant_paths(variant_name=args.variant)
    if not resolved.data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {resolved.data_yaml}")
    return resolved.data_yaml


def run_val(args: argparse.Namespace, data_yaml: Path) -> Any:
    from ultralytics import YOLO

    device = _parse_device(args.device)
    model = YOLO(str(Path(args.weights).resolve()))
    val_kwargs: dict[str, Any] = dict(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        split="test",
        plots=args.plots,
        half=args.half,
    )
    if args.save_json:
        val_kwargs["save_json"] = True
    val_kwargs["name"] = args.name
    if args.project is not None:
        val_kwargs["project"] = str(args.project.resolve())
    return model.val(**val_kwargs)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ultralytics YOLO validation.")
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--variant", choices=variant_choices(), default=None)
    parser.add_argument("--data", default=None, type=Path)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--name", default="test")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--half", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--save-json", action=argparse.BooleanOptionalAction, default=False
    )
    return parser


def _print_val_metric_summary(metrics: Any) -> None:
    for section in ("box", "seg", "mask"):
        block = _optional_metric_attr(metrics, section)
        if block is None:
            continue
        map_all = _optional_metric_attr(block, "map")
        map50 = _optional_metric_attr(block, "map50")
        map75 = _optional_metric_attr(block, "map75")
        if map_all is None and map50 is None and map75 is None:
            continue
        parts: list[str] = []
        if map_all is not None:
            parts.append(f"mAP50-95={float(map_all):.4f}")
        if map50 is not None:
            parts.append(f"mAP50={float(map50):.4f}")
        if map75 is not None:
            parts.append(f"mAP75={float(map75):.4f}")
        print(f"{section}: " + " ".join(parts))


def main(argv: list[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    data_yaml = _resolve_data_yaml(args)
    print(
        f"YOLO validation: weights={Path(args.weights).resolve()}, "
        f"data={data_yaml}, split=test"
    )
    metrics = run_val(args, data_yaml)
    _print_val_metric_summary(metrics)
    write_val_metrics_json(metrics, project=args.project, name=args.name)


if __name__ == "__main__":
    main()
