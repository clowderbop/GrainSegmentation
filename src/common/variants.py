"""Canonical microscopy variant registry loaded from config/variants.yaml."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_REGISTRY_RELATIVE = Path("config") / "variants.yaml"
_VARIANT_ORDER = ("PPL+AllPPX", "PPL+PPXblend", "PPLPPXblend", "PPL")
_THESIS_VARIANT_ORDER = ("PPL", "PPL+AllPPX", "PPL+PPXblend", "PPLPPXblend")
_EXPECTED_DISPLAY_NAMES = {
    "PPL": "PPL",
    "PPL+AllPPX": "FullStack",
    "PPL+PPXblend": "PPL+XPLComp",
    "PPLPPXblend": "FullComp",
}


@dataclass(frozen=True)
class UnetSpec:
    num_inputs: int
    input_suffixes: tuple[str, ...]
    channels_per_input: int


@dataclass(frozen=True)
class YoloSpec:
    input_channels: int
    dataset_subdir: str
    yaml_name: str


@dataclass(frozen=True)
class PathTemplates:
    """Relative paths under grainseg_root (from registry YAML)."""

    grainseg_root_env: str
    train_dir: str
    test_dir: str
    train_labels_raster: str
    train_labels_gpkg: str
    test_labels_raster: str
    test_labels_gpkg: str
    train_mosaic_stacked: str
    test_mosaic_stacked: str
    train_channel_template: str
    test_channel_template: str
    train_patches_dir: str
    test_patches_dir: str
    yolo_dataset_root: str


@dataclass(frozen=True)
class ResolvedPaths:
    """Absolute paths resolved against a grainseg root."""

    grainseg_root: Path
    templates: PathTemplates
    train_dir: Path
    test_dir: Path
    train_labels_raster: Path
    train_labels_gpkg: Path
    test_labels_raster: Path
    test_labels_gpkg: Path
    train_mosaic_stacked: Path
    test_mosaic_stacked: Path
    train_patches_dir: Path
    test_patches_dir: Path
    yolo_dataset_root: Path

    def train_channel_path(self, suffix: str) -> Path:
        rel = self.templates.train_channel_template.format(suffix=suffix)
        return self.grainseg_root / rel

    def test_channel_path(self, suffix: str) -> Path:
        rel = self.templates.test_channel_template.format(suffix=suffix)
        return self.grainseg_root / rel


@dataclass(frozen=True)
class VariantSlugs:
    job: str
    slurm_job: str
    model_file: str


@dataclass(frozen=True)
class VariantSpec:
    name: str
    display_name: str
    unet: UnetSpec
    yolo: YoloSpec
    paths: PathTemplates
    slugs: VariantSlugs

    def resolve_paths(self, grainseg_root: str | Path) -> ResolvedPaths:
        root = Path(grainseg_root).resolve()

        def _join(rel: str) -> Path:
            return root / rel

        return ResolvedPaths(
            grainseg_root=root,
            templates=self.paths,
            train_dir=_join(self.paths.train_dir),
            test_dir=_join(self.paths.test_dir),
            train_labels_raster=_join(self.paths.train_labels_raster),
            train_labels_gpkg=_join(self.paths.train_labels_gpkg),
            test_labels_raster=_join(self.paths.test_labels_raster),
            test_labels_gpkg=_join(self.paths.test_labels_gpkg),
            train_mosaic_stacked=_join(self.paths.train_mosaic_stacked),
            test_mosaic_stacked=_join(self.paths.test_mosaic_stacked),
            train_patches_dir=_join(self.paths.train_patches_dir),
            test_patches_dir=_join(self.paths.test_patches_dir),
            yolo_dataset_root=_join(self.paths.yolo_dataset_root),
        )

    def train_channel_path(self, grainseg_root: str | Path, suffix: str) -> Path:
        return self.resolve_paths(grainseg_root).train_channel_path(suffix)

    def test_channel_path(self, grainseg_root: str | Path, suffix: str) -> Path:
        return self.resolve_paths(grainseg_root).test_channel_path(suffix)


@dataclass(frozen=True)
class VariantRegistry:
    schema_version: int
    variants: dict[str, VariantSpec]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def registry_path() -> Path:
    return repo_root() / _REGISTRY_RELATIVE


def default_grainseg_root(scratch_root: str | Path | None = None) -> Path:
    if scratch_root is not None:
        return Path(scratch_root).resolve() / "GrainSeg"
    scratch = os.environ.get("SCRATCH")
    if scratch:
        return Path(scratch).resolve() / "GrainSeg"
    user = os.environ.get("USER", "user")
    return Path(f"/scratch/{user}/GrainSeg")


@lru_cache(maxsize=1)
def load_registry() -> VariantRegistry:
    path = registry_path()
    if not path.is_file():
        raise FileNotFoundError(f"Variant registry not found: {path}")
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Registry {path} must be a YAML mapping")
    return _parse_registry(raw)


def all_variant_names() -> tuple[str, ...]:
    """Variant names in registry YAML order."""
    reg = load_registry()
    return tuple(reg.variants)


def variants_in_thesis_order() -> tuple[str, ...]:
    """Registry keys in thesis figure/table order (PPL → FullStack → …)."""
    reg = load_registry()
    missing = [name for name in _THESIS_VARIANT_ORDER if name not in reg.variants]
    if missing:
        raise ValueError(f"Thesis-order variants missing from registry: {missing}")
    return _THESIS_VARIANT_ORDER


def variant_display_names_in_thesis_order() -> tuple[str, ...]:
    reg = load_registry()
    return tuple(reg.variants[name].display_name for name in variants_in_thesis_order())


def get_variant(name: str) -> VariantSpec:
    reg = load_registry()
    try:
        return reg.variants[name]
    except KeyError as exc:
        valid = ", ".join(sorted(reg.variants))
        raise ValueError(
            f"Unknown microscopy variant {name!r}. Expected one of: {valid}"
        ) from exc


def unet_finetuned_eval_run_name(variant: str) -> str:
    """Run subdirectory name under a whole-section U-Net eval output dir."""
    model_file = get_variant(variant).slugs.model_file
    return f"run_{Path(model_file).stem}"


def variant_input_image_count(variant_key: str) -> int:
    """Microscopy image count for an input configuration (thesis complexity axis)."""
    return get_variant(variant_key).unet.num_inputs


def validate(registry: VariantRegistry | None = None) -> None:
    reg = registry or load_registry()
    if reg.schema_version != 1:
        raise ValueError(f"Unsupported schema_version: {reg.schema_version}")

    expected_yolo_channels = {
        "PPL": 3,
        "PPLPPXblend": 3,
        "PPL+PPXblend": 6,
        "PPL+AllPPX": 21,
    }
    if set(reg.variants) != set(expected_yolo_channels):
        raise ValueError(
            f"Registry variants {sorted(reg.variants)} != expected "
            f"{sorted(expected_yolo_channels)}"
        )

    if set(_EXPECTED_DISPLAY_NAMES) != set(reg.variants):
        raise ValueError(
            f"Registry variants {sorted(reg.variants)} != expected "
            f"{sorted(_EXPECTED_DISPLAY_NAMES)}"
        )
    seen_display: set[str] = set()
    for name, spec in reg.variants.items():
        expected_display = _EXPECTED_DISPLAY_NAMES.get(name)
        if expected_display is None:
            raise ValueError(f"Unexpected variant {name!r}")
        if spec.display_name != expected_display:
            raise ValueError(
                f"{name}: display_name {spec.display_name!r} != {expected_display!r}"
            )
        if spec.display_name in seen_display:
            raise ValueError(f"Duplicate display_name: {spec.display_name!r}")
        seen_display.add(spec.display_name)
        if spec.unet.channels_per_input != 3:
            raise ValueError(f"{name}: channels_per_input must be 3")
        if len(spec.unet.input_suffixes) != spec.unet.num_inputs:
            raise ValueError(f"{name}: input_suffixes length != num_inputs")
        if spec.yolo.input_channels != expected_yolo_channels[name]:
            raise ValueError(
                f"{name}: yolo.input_channels {spec.yolo.input_channels} != "
                f"{expected_yolo_channels[name]}"
            )
        for suffix in spec.unet.input_suffixes:
            if not suffix.startswith("_"):
                raise ValueError(f"{name}: suffix {suffix!r} must start with '_'")


def _parse_registry(raw: dict[str, Any]) -> VariantRegistry:
    schema_version = int(raw["schema_version"])
    variants_raw = raw.get("variants")
    if not isinstance(variants_raw, dict):
        raise ValueError('Registry requires top-level "variants" mapping')

    variants: dict[str, VariantSpec] = {}
    for name, entry in variants_raw.items():
        if not isinstance(entry, dict):
            raise ValueError(f"Variant {name!r} must be a mapping")
        variants[name] = _parse_variant(name, entry)

    return VariantRegistry(schema_version=schema_version, variants=variants)


def _parse_variant(name: str, entry: dict[str, Any]) -> VariantSpec:
    unet_raw = entry["unet"]
    yolo_raw = entry["yolo"]
    paths_raw = entry["paths"]
    slugs_raw = entry["slugs"]

    unet = UnetSpec(
        num_inputs=int(unet_raw["num_inputs"]),
        input_suffixes=tuple(str(s) for s in unet_raw["input_suffixes"]),
        channels_per_input=int(unet_raw["channels_per_input"]),
    )
    yolo = YoloSpec(
        input_channels=int(yolo_raw["input_channels"]),
        dataset_subdir=str(yolo_raw["dataset_subdir"]),
        yaml_name=str(yolo_raw["yaml_name"]),
    )
    paths = PathTemplates(
        grainseg_root_env=str(paths_raw.get("grainseg_root_env", "SCRATCH")),
        train_dir=str(paths_raw["train_dir"]),
        test_dir=str(paths_raw["test_dir"]),
        train_labels_raster=str(paths_raw["train_labels_raster"]),
        train_labels_gpkg=str(paths_raw["train_labels_gpkg"]),
        test_labels_raster=str(paths_raw["test_labels_raster"]),
        test_labels_gpkg=str(paths_raw["test_labels_gpkg"]),
        train_mosaic_stacked=str(paths_raw["train_mosaic_stacked"]),
        test_mosaic_stacked=str(paths_raw["test_mosaic_stacked"]),
        train_channel_template=str(paths_raw["train_channel_template"]),
        test_channel_template=str(paths_raw["test_channel_template"]),
        train_patches_dir=str(paths_raw["train_patches_dir"]),
        test_patches_dir=str(paths_raw["test_patches_dir"]),
        yolo_dataset_root=str(paths_raw["yolo_dataset_root"]),
    )
    slugs = VariantSlugs(
        job=str(slugs_raw["job"]),
        slurm_job=str(slugs_raw["slurm_job"]),
        model_file=str(slugs_raw["model_file"]),
    )
    display_name = str(entry["display_name"])
    return VariantSpec(
        name=name,
        display_name=display_name,
        unet=unet,
        yolo=yolo,
        paths=paths,
        slugs=slugs,
    )


def _shell_quote(value: str) -> str:
    if re.fullmatch(r"[\w.+@-]+", value):
        return value
    return "'" + value.replace("'", "'\"'\"'") + "'"


def _format_env_exports(spec: VariantSpec, *, grainseg_root: Path | None = None) -> str:
    resolved = spec.resolve_paths(grainseg_root) if grainseg_root else None
    suffixes_bash = " ".join(_shell_quote(s) for s in spec.unet.input_suffixes)
    lines = [
        f"export VARIANT_NAME={_shell_quote(spec.name)}",
        f"export NUM_INPUTS={spec.unet.num_inputs}",
        f"export IMAGE_SUFFIXES_CSV={_shell_quote(','.join(spec.unet.input_suffixes))}",
        f"IMAGE_SUFFIXES=({suffixes_bash})",
        f"export DEFAULT_MODEL_BASENAME={_shell_quote(spec.slugs.model_file)}",
        f"export DATASET_SUBDIR={_shell_quote(spec.yolo.dataset_subdir)}",
        f"export YAML_NAME={_shell_quote(spec.yolo.yaml_name)}",
        f"export YOLO_INPUT_CHANNELS={spec.yolo.input_channels}",
        f"export WATERSHED_JOB_SLUG={_shell_quote(spec.slugs.job)}",
        f"export SLURM_JOB_SLUG={_shell_quote(spec.slugs.slurm_job)}",
    ]
    if resolved is not None:
        lines.extend(
            [
                f"export GRAINSEG_ROOT={_shell_quote(str(resolved.grainseg_root))}",
                f"export TRAIN_MOSAIC_STACKED={_shell_quote(str(resolved.train_mosaic_stacked))}",
                f"export TEST_MOSAIC_STACKED={_shell_quote(str(resolved.test_mosaic_stacked))}",
                f"export YOLO_TEST_TIFF={_shell_quote(str(resolved.test_mosaic_stacked))}",
                f"export TRAIN_LABELS_RASTER={_shell_quote(str(resolved.train_labels_raster))}",
                f"export TRAIN_LABELS_GPKG={_shell_quote(str(resolved.train_labels_gpkg))}",
            ]
        )
    return "\n".join(lines)


def _variant_to_json(
    spec: VariantSpec, *, grainseg_root: Path | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": spec.name,
        "display_name": spec.display_name,
        "unet": {
            "num_inputs": spec.unet.num_inputs,
            "input_suffixes": list(spec.unet.input_suffixes),
            "channels_per_input": spec.unet.channels_per_input,
        },
        "yolo": {
            "input_channels": spec.yolo.input_channels,
            "dataset_subdir": spec.yolo.dataset_subdir,
            "yaml_name": spec.yolo.yaml_name,
        },
        "paths": {
            field.name: getattr(spec.paths, field.name)
            for field in PathTemplates.__dataclass_fields__.values()
        },
        "slugs": {
            "job": spec.slugs.job,
            "slurm_job": spec.slugs.slurm_job,
            "model_file": spec.slugs.model_file,
        },
    }
    if grainseg_root is not None:
        resolved = spec.resolve_paths(grainseg_root)
        payload["resolved_paths"] = {
            "grainseg_root": str(resolved.grainseg_root),
            "train_mosaic_stacked": str(resolved.train_mosaic_stacked),
            "test_mosaic_stacked": str(resolved.test_mosaic_stacked),
            "yolo_dataset_root": str(resolved.yolo_dataset_root),
        }
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m common.variants")
    parser.add_argument(
        "--grainseg-root",
        type=Path,
        default=None,
        help="Scratch GrainSeg root (default: $SCRATCH/GrainSeg)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    env_p = sub.add_parser("env", help="Shell-exportable variables for one variant")
    env_p.add_argument("--variant", required=True)

    json_p = sub.add_parser("print-json", help="Print variant spec as JSON")
    json_p.add_argument("--variant", required=True)

    sub.add_parser("all-names", help="Print variant names (space-separated)")

    meta_p = sub.add_parser("unet-metadata-tsv", help="num_inputs<TAB>suffix_csv")
    meta_p.add_argument("--variant", required=True)

    ws_p = sub.add_parser("watershed-subdir", help="Watershed tune subdirectory slug")
    ws_p.add_argument("--variant", required=True)

    yolo_p = sub.add_parser("yolo-test-tiff", help="Absolute path to test stacked TIFF")
    yolo_p.add_argument("--variant", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    grainseg = args.grainseg_root
    if grainseg is None and args.command in {
        "env",
        "print-json",
        "yolo-test-tiff",
    }:
        grainseg = default_grainseg_root()

    try:
        if args.command == "all-names":
            print(" ".join(all_variant_names()))
            return 0
        if args.command == "env":
            print(
                _format_env_exports(get_variant(args.variant), grainseg_root=grainseg)
            )
            return 0
        if args.command == "print-json":
            print(
                json.dumps(
                    _variant_to_json(get_variant(args.variant), grainseg_root=grainseg),
                    indent=2,
                )
            )
            return 0
        if args.command == "unet-metadata-tsv":
            spec = get_variant(args.variant)
            print(f"{spec.unet.num_inputs}\t{','.join(spec.unet.input_suffixes)}")
            return 0
        if args.command == "watershed-subdir":
            print(get_variant(args.variant).slugs.job)
            return 0
        if args.command == "yolo-test-tiff":
            resolved = get_variant(args.variant).resolve_paths(grainseg)
            print(resolved.test_mosaic_stacked)
            return 0
    except (ValueError, FileNotFoundError) as exc:
        print(exc, file=sys.stderr)
        return 1

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
