"""Run provenance sidecar next to a prediction set directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

RUN_PROVENANCE_FILENAME = "run_provenance.json"


def run_provenance_path(output_root: Path | str) -> Path:
    return Path(output_root) / RUN_PROVENANCE_FILENAME


def write_run_provenance(output_root: Path | str, payload: dict[str, Any]) -> Path:
    path = run_provenance_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_run_provenance(output_root: Path | str) -> dict[str, Any]:
    path = run_provenance_path(output_root)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Run provenance {path} must be a JSON object")
    return data


__all__ = [
    "RUN_PROVENANCE_FILENAME",
    "load_run_provenance",
    "run_provenance_path",
    "write_run_provenance",
]
