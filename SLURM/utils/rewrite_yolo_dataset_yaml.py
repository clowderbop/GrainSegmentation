"""Rewrite a YOLO dataset YAML path (and optional test split) after staging to TMPDIR."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    yaml_path = Path(sys.argv[1])
    dataset_root = Path(sys.argv[2])
    rewrite_test = len(sys.argv) > 3 and sys.argv[3] == "test"

    text = yaml_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("path:"):
            lines[index] = f"path: {dataset_root}"
            break
    else:
        raise SystemExit(f"Dataset YAML missing path entry: {yaml_path}")

    if rewrite_test:
        for index, line in enumerate(lines):
            if line.strip() == "test:":
                lines[index] = "test: images/test"
                break

    trailing_newline = "\n" if text.endswith("\n") else ""
    yaml_path.write_text("\n".join(lines) + trailing_newline, encoding="utf-8")


if __name__ == "__main__":
    main()
