"""Copy manifest-referenced files to a work directory and rewrite the manifest."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from common.manifest_io import (
    DatasetManifest,
    ManifestSampleRow,
    build_eval_manifest,
    load_dataset_manifest,
    resolve_manifest_path,
    write_dataset_manifest,
)


def _copy_asset(
    source: Path,
    dest: Path,
    *,
    exist_ok: bool = True,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == dest.resolve():
        return
    shutil.copy2(source, dest, follow_symlinks=True)


def stage_manifest(
    manifest: Path | DatasetManifest,
    work_root: str | Path,
    *,
    sample_ids: set[str] | None = None,
    copy_masks: bool = True,
    flatten_images: bool = True,
) -> DatasetManifest:
    """Copy referenced files into ``work_root`` and return an updated manifest."""
    doc = manifest if isinstance(manifest, DatasetManifest) else load_dataset_manifest(manifest)
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    source_base = Path(doc.grainseg_root).resolve()
    if doc.path_base == "work_root":
        source_base = work_root
    elif doc.path_base != "grainseg_root":
        raise ValueError(f"Unsupported path_base for staging: {doc.path_base!r}")

    staged_rows: list[ManifestSampleRow] = []
    rows_to_stage = [
        row
        for row in doc.samples
        if sample_ids is None or row.sample_id in sample_ids
    ]
    n_rows = len(rows_to_stage)
    print(
        f"Staging {n_rows} manifest sample(s) to {work_root} "
        f"(variant={doc.variant}, unit={doc.unit})..."
    )
    for idx, row in enumerate(rows_to_stage):
        print(f"Staging sample {row.sample_id} ({idx + 1}/{n_rows})...")

        new_image: str | None = None
        new_images: tuple[str, ...] | None = None

        if row.image is not None:
            src = resolve_manifest_path(row.image, source_base)
            dest = work_root / Path(row.image).name
            print(f"  copy image: {src.name} -> {dest}")
            _copy_asset(src, dest)
            new_image = dest.name if flatten_images else str(dest.relative_to(work_root))

        if row.images is not None:
            copied: list[str] = []
            for rel in row.images:
                src = resolve_manifest_path(rel, source_base)
                dest = work_root / Path(rel).name
                print(f"  copy channel: {src.name} -> {dest}")
                _copy_asset(src, dest)
                copied.append(dest.name if flatten_images else str(dest.relative_to(work_root)))
            new_images = tuple(copied)

        new_mask: str | None = None
        if copy_masks and row.mask is not None:
            src = resolve_manifest_path(row.mask, source_base)
            dest = work_root / Path(row.mask).name
            print(f"  copy mask: {src.name} -> {dest}")
            _copy_asset(src, dest)
            new_mask = dest.name

        staged_rows.append(
            ManifestSampleRow(
                sample_id=row.sample_id,
                image=new_image,
                images=new_images,
                mask=new_mask,
                gt_gpkg=row.gt_gpkg,
                gt_origin=row.gt_origin,
                gt_txt=row.gt_txt,
                pred_instances=row.pred_instances,
                semantic=row.semantic,
            )
        )

    if not staged_rows:
        raise ValueError("No manifest samples staged (check sample_ids filter)")

    return DatasetManifest(
        schema_version=doc.schema_version,
        variant=doc.variant,
        unit=doc.unit,
        grainseg_root=str(work_root),
        path_base="work_root",
        samples=tuple(staged_rows),
        source_path=work_root / "manifest.json",
    )


def stage_manifest_to_file(
    manifest_path: Path,
    work_root: Path,
    *,
    sample_ids: set[str] | None = None,
    output_name: str = "manifest.json",
) -> Path:
    staged = stage_manifest(manifest_path, work_root, sample_ids=sample_ids)
    out_path = work_root / output_name
    write_dataset_manifest(out_path, staged)
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Copy assets and write work_root/manifest.json")
    run_p.add_argument("manifest", type=Path)
    run_p.add_argument("work_root", type=Path)
    run_p.add_argument("--sample-id", action="append", default=None)

    eval_p = sub.add_parser(
        "write-eval",
        help="Write single-image eval manifest with pred_instances paths",
    )
    eval_p.add_argument("--source", type=Path, required=True)
    eval_p.add_argument("--pred-instances-dir", type=Path, required=True)
    eval_p.add_argument("--output", type=Path, required=True)
    eval_p.add_argument("--work-root", type=Path, default=None)
    eval_p.add_argument("--gt-gpkg", type=Path, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            sample_ids = set(args.sample_id) if args.sample_id else None
            manifest_path = args.manifest.resolve()
            work_root = args.work_root.resolve()
            print(f"Staging manifest {manifest_path} -> {work_root}")
            out = stage_manifest_to_file(
                manifest_path,
                work_root,
                sample_ids=sample_ids,
            )
            print(f"Wrote staged manifest to {out}")
            return 0
        if args.command == "write-eval":
            source = load_dataset_manifest(args.source)
            work_root = args.work_root or args.output.parent
            print(
                f"Building eval manifest: {len(source.samples)} sample(s), "
                f"pred_instances_dir={args.pred_instances_dir.resolve()}"
            )
            eval_doc = build_eval_manifest(
                source,
                pred_instances_dir=args.pred_instances_dir,
                manifest_parent=args.output.parent,
                gt_gpkg=args.gt_gpkg,
            )
            write_dataset_manifest(args.output, eval_doc)
            print(f"Wrote eval manifest to {args.output}")
            return 0
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(exc, file=sys.stderr)
        return 1

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
