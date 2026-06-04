"""Tests for profile selection durable cache path contract."""

from __future__ import annotations

from pathlib import Path

from common.profile_tune_paths import profile_tune_cache_root


def test_profile_tune_cache_root_defaults_to_dot_cache(tmp_path: Path) -> None:
    output_dir = tmp_path / "runs" / "yolo_inference_profile_tune" / "20260101_120000"
    assert profile_tune_cache_root(output_dir) == output_dir / ".cache"
