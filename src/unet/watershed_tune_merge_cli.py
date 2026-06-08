"""CLI: merge shard watershed tune grid CSVs into canonical tune artifacts."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

from common.arg_errors import raise_cli_argument_error
from common.manifest_io import collect_manifest_tune_samples, load_dataset_manifest
from unet.extraction_tune_scoring import sanitize_watershed_tune_csv_sample_id
from unet.watershed_tune_grid import load_watershed_tune_grid
from unet.watershed_tune_merge import (
    merge_watershed_tune_shard_csvs,
    write_watershed_tune_merge_artifacts,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-csv",
        action="append",
        type=Path,
        default=[],
        help="Explicit shard grid CSV path (repeatable)",
    )
    parser.add_argument(
        "--shard-csv-glob",
        default=None,
        help="Glob pattern resolving shard grid CSV paths",
    )
    parser.add_argument(
        "--grid-config",
        type=Path,
        default=None,
        help="Watershed tune grid YAML for partition validation",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Dataset manifest supplying tune sample_ids for best JSON",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=None,
        help="Explicit tune sample_ids when manifest is not supplied",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _validate_merge_args(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    if not args.shard_csv and not args.shard_csv_glob:
        raise_cli_argument_error(
            "at least one shard CSV is required via --shard-csv or --shard-csv-glob",
            parser=parser,
        )
    if args.sample_ids is None and args.manifest is None:
        raise_cli_argument_error(
            "sample_ids require --manifest or --sample-ids",
            parser=parser,
        )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    _validate_merge_args(args, parser)
    return args


def _resolve_shard_csv_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.shard_csv)
    if args.shard_csv_glob:
        paths.extend(Path(path) for path in sorted(glob.glob(args.shard_csv_glob)))
    return sorted(paths)


def _resolve_sample_ids(args: argparse.Namespace) -> list[str]:
    if args.sample_ids is not None:
        return list(args.sample_ids)
    doc = load_dataset_manifest(args.manifest)
    samples = collect_manifest_tune_samples(doc)
    return [sample["id"] for sample in samples]


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    shard_csv_paths = _resolve_shard_csv_paths(args)
    sample_ids = _resolve_sample_ids(args)
    grid = (
        load_watershed_tune_grid(args.grid_config).grid
        if args.grid_config is not None
        else None
    )
    sanitize = sanitize_watershed_tune_csv_sample_id
    result = merge_watershed_tune_shard_csvs(
        shard_csv_paths,
        grid=grid,
        sample_ids=sample_ids,
        sanitize_sample_id=sanitize,
    )
    write_watershed_tune_merge_artifacts(
        result,
        output_csv=args.output_csv,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
