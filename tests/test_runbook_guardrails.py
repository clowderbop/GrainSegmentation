"""Light guardrails: cluster ops docs live under docs/runbooks/."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_RUNBOOKS = (
    "docs/runbooks/README.md",
    "docs/runbooks/preprocessing.md",
    "docs/runbooks/yolo.md",
    "docs/runbooks/unet.md",
    "docs/runbooks/analysis.md",
)

SUBMIT_SCRIPTS = sorted((REPO_ROOT / "SLURM").rglob("submit_*.sh"))


@pytest.mark.parametrize("rel_path", EXPECTED_RUNBOOKS)
def test_expected_runbook_exists(rel_path: str) -> None:
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"missing runbook: {rel_path}"


@pytest.mark.parametrize(
    "submit_script",
    SUBMIT_SCRIPTS,
    ids=lambda p: p.relative_to(REPO_ROOT).as_posix(),
)
def test_submit_script_references_runbooks(submit_script: Path) -> None:
    text = submit_script.read_text(encoding="utf-8")
    assert "docs/runbooks/" in text, (
        f"{submit_script.relative_to(REPO_ROOT)} must mention docs/runbooks/ "
        "(usage, --help, or header comment)"
    )
