"""Run provenance sidecar I/O."""

from __future__ import annotations

from pathlib import Path

from common.run_provenance import load_run_provenance, write_run_provenance


def test_run_provenance_round_trip(tmp_path: Path) -> None:
    """INTENT: write_run_provenance and load_run_provenance preserve inference run metadata."""
    payload = {
        "producer": "yolo",
        "conf": 0.25,
        "slice_height": 1024,
        "slice_width": 1024,
        "overlap_height_ratio": 0.5,
        "overlap_width_ratio": 0.5,
    }
    write_run_provenance(tmp_path, payload)
    loaded = load_run_provenance(tmp_path)
    assert loaded == payload
    assert loaded["producer"] == "yolo"
    assert isinstance(loaded["conf"], float)
    assert isinstance(loaded["slice_height"], int)
