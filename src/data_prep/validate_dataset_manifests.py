#!/usr/bin/env python3
"""Validate whole and patch manifests on an existing GrainSeg scratch tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common.manifest_io import default_patch_manifest_path, default_whole_manifest_path, load_dataset_manifest
from common.variants import all_variant_names, default_grainseg_root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grainseg-root", type=Path, default=None)
    parser.add_argument("--variant", action="append", default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    grainseg = (args.grainseg_root or default_grainseg_root()).resolve()
    variants = tuple(args.variant) if args.variant else all_variant_names()
    checked = 0
    missing: list[str] = []

    for split in ("train", "test"):
        for variant in variants:
            whole_path = default_whole_manifest_path(grainseg, split, variant)
            if whole_path.is_file():
                load_dataset_manifest(whole_path)
                print(f"OK {whole_path.relative_to(grainseg)}")
                checked += 1
            else:
                missing.append(str(whole_path.relative_to(grainseg)))

            patch_path = default_patch_manifest_path(grainseg, split, variant)
            if patch_path.is_file():
                load_dataset_manifest(patch_path)
                print(f"OK {patch_path.relative_to(grainseg)}")
                checked += 1
            else:
                missing.append(str(patch_path.relative_to(grainseg)))

    if checked == 0:
        print("No manifests found to validate.", file=sys.stderr)
        return 1
    if missing:
        print("Missing (not validated):", file=sys.stderr)
        for rel in missing:
            print(f"  {rel}", file=sys.stderr)
    print(f"Validated {checked} manifest(s) under {grainseg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
