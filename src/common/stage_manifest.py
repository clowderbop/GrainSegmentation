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
    manifest_path_base_dir,
    require_eval_local_path,
    resolve_manifest_path,
    resolve_row_path,
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


def _is_under(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    root = root.resolve()
    return resolved == root or root in resolved.parents


def _materialize_eval_file(source: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _copy_asset(source, dest)
    return dest.resolve()


def _allowed_materialize_roots(
    source: DatasetManifest, manifest_parent: Path
) -> list[Path]:
    roots = [manifest_parent.resolve()]
    if source.path_base == "work_root":
        roots.append(manifest_path_base_dir(source))
    return roots


def _assert_materializable_gt(
    path: Path,
    source: DatasetManifest,
    allowed_roots: list[Path],
    manifest_parent: Path,
) -> None:
    resolved = path.resolve()
    if _is_under(resolved, manifest_parent):
        return
    for root in allowed_roots:
        if _is_under(resolved, root):
            return
    if source.path_base == "grainseg_root":
        grainseg = Path(source.grainseg_root).resolve()
        if _is_under(resolved, grainseg):
            raise ValueError(
                f"Ground-truth path must be under staged work directory "
                f"({manifest_parent}): {resolved}"
            )
    require_eval_local_path(resolved, *allowed_roots)


def materialize_eval_row_assets(
    source: DatasetManifest,
    row: ManifestSampleRow,
    manifest_parent: Path,
    *,
    anchor_suffix: str = "_PPL",
    gt_gpkg_override: Path | None = None,
) -> tuple[Path, Path | None, Path | None]:
    """Copy anchor image and ground truth into ``manifest_parent`` when needed."""
    manifest_parent = manifest_parent.resolve()
    allowed_roots = _allowed_materialize_roots(source, manifest_parent)

    anchor = row.anchor_image_path(suffix=anchor_suffix)
    image_resolved = resolve_row_path(source, anchor)
    assert image_resolved is not None
    image_dest = manifest_parent / image_resolved.name
    if not _is_under(image_resolved, manifest_parent):
        image_resolved = _materialize_eval_file(image_resolved, image_dest)

    gt_resolved: Path | None = None
    if gt_gpkg_override is not None:
        override = Path(gt_gpkg_override).resolve()
        _assert_materializable_gt(override, source, allowed_roots, manifest_parent)
        rel = row.gt_gpkg if row.gt_gpkg is not None else override.name
        gt_dest = manifest_parent / rel
        gt_resolved = (
            override
            if _is_under(override, manifest_parent)
            else _materialize_eval_file(override, gt_dest)
        )
    elif row.gt_gpkg is not None:
        gt_resolved = resolve_row_path(source, row.gt_gpkg)
        assert gt_resolved is not None
        gt_dest = manifest_parent / row.gt_gpkg
        if not _is_under(gt_resolved, manifest_parent):
            gt_resolved = _materialize_eval_file(gt_resolved, gt_dest)

    gt_txt_resolved: Path | None = None
    if row.gt_txt is not None:
        gt_txt_resolved = resolve_row_path(source, row.gt_txt)
        assert gt_txt_resolved is not None
        gt_txt_dest = manifest_parent / row.gt_txt
        if not _is_under(gt_txt_resolved, manifest_parent):
            gt_txt_resolved = _materialize_eval_file(gt_txt_resolved, gt_txt_dest)

    return image_resolved, gt_resolved, gt_txt_resolved


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
    copied_relative: set[str] = set()
    rows_to_stage = [
        row
        for row in doc.samples
        if sample_ids is None or row.sample_id in sample_ids
    ]

    def _copy_relative_asset(rel: str, *, label: str) -> None:
        if rel in copied_relative:
            return
        src = resolve_manifest_path(rel, source_base)
        dest = work_root / rel
        print(f"  copy {label}: {src.name} -> {dest.relative_to(work_root)}")
        _copy_asset(src, dest)
        copied_relative.add(rel)

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

        if row.gt_txt is not None:
            _copy_relative_asset(row.gt_txt, label="gt_txt")
        if row.gt_gpkg is not None:
            _copy_relative_asset(row.gt_gpkg, label="gt_gpkg")

        staged_rows.append(
            ManifestSampleRow(
                sample_id=row.sample_id,
                image=new_image,
                images=new_images,
                mask=new_mask,
                gt_gpkg=row.gt_gpkg,
                gt_origin=row.gt_origin,
                gt_txt=row.gt_txt,
                instance_prediction_set=row.instance_prediction_set,
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
        help="Write eval manifest with instance_prediction_set paths",
    )
    eval_p.add_argument("--source", type=Path, required=True)
    eval_p.add_argument("--prediction-set-dir", type=Path, required=True)
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
                f"prediction_set_dir={args.prediction_set_dir.resolve()}"
            )
            eval_doc = build_eval_manifest(
                source,
                prediction_set_dir=args.prediction_set_dir,
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
