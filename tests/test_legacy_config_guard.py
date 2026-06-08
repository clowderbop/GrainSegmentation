"""Guard against reintroducing retired configs/ default recipe paths."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

RETIRED_DEFAULT_PATHS = (
    "configs/test_inference.yaml",
    "configs/yolo_inference_profile_tune.yaml",
)

VERSIONED_CONFIG_YAML = (
    "config/variants.yaml",
    "config/test_inference.yaml",
    "config/yolo_inference_profile_tune.yaml",
    "config/watershed_tune_grid.yaml",
)

TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".sh",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".json",
}

SKIP_DIR_NAMES = {
    ".git",
    ".scratch",
    "__pycache__",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
}


def find_retired_default_path_hits(text: str) -> list[str]:
    """Return retired repo-default paths referenced in *text*."""
    return [path for path in RETIRED_DEFAULT_PATHS if path in text]


def iter_repo_text_files() -> list[Path]:
    """Return git-tracked text files, excluding historical .scratch/ issue bodies."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO_ROOT,
        capture_output=True,
        check=True,
    )
    files: list[Path] = []
    for rel in result.stdout.decode("utf-8").split("\0"):
        if not rel:
            continue
        path = Path(rel)
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        files.append(REPO_ROOT / path)
    return sorted(files)


def test_guard_detects_retired_default_path_reference() -> None:
    hits = find_retired_default_path_hits(
        "Load defaults from configs/test_inference.yaml before promotion."
    )
    assert hits == ["configs/test_inference.yaml"]


def test_guard_ignores_custom_grid_config_override_paths() -> None:
    """Custom GRID_CONFIG paths outside the repo are out of scope for this guard."""
    text = "\n".join(
        (
            'GRID_CONFIG="${GRID_CONFIG:-$REPO_ROOT/config/yolo_inference_profile_tune.yaml}"',
            "GRID_CONFIG=/scratch/user/custom_profile_grid.yaml",
            "Use config/yolo_inference_profile_tune.yaml as the committed default.",
        )
    )
    assert find_retired_default_path_hits(text) == []


@pytest.mark.parametrize("rel_path", VERSIONED_CONFIG_YAML)
def test_versioned_project_yaml_exists(rel_path: str) -> None:
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"missing versioned project YAML: {rel_path}"


def test_retired_config_defaults_not_tracked_in_git() -> None:
    result = subprocess.run(
        ["git", "ls-files", *RETIRED_DEFAULT_PATHS],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked = [line for line in result.stdout.splitlines() if line.strip()]
    assert tracked == []


def test_no_committed_references_to_retired_config_defaults() -> None:
    offenders: list[str] = []
    for path in iter_repo_text_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel == Path(__file__).relative_to(REPO_ROOT).as_posix():
            continue
        hits = find_retired_default_path_hits(path.read_text(encoding="utf-8"))
        if hits:
            offenders.append(f"{rel}: {', '.join(hits)}")
    assert offenders == []
