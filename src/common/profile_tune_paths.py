"""Profile selection durable scratch cache paths (ADR 0005)."""

from __future__ import annotations

from pathlib import Path

PROFILE_TUNE_CACHE_DIR_NAME = ".cache"


def profile_tune_cache_root(output_dir: Path) -> Path:
    """Durable cache root under a profile selection run directory."""
    return output_dir / PROFILE_TUNE_CACHE_DIR_NAME
