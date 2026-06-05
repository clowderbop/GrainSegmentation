"""Shared assertions for YOLO whole-section phase progress logs."""

from __future__ import annotations

import re

DONE_TIMING_PATTERN = re.compile(r"\d+\.\d+s")


def assert_done_timing_lines(text: str, *, min_count: int = 1) -> None:
    matches = DONE_TIMING_PATTERN.findall(text)
    assert len(matches) >= min_count, (
        f"expected at least {min_count} done timing(s) matching \\d+\\.\\ds, got {matches!r} in:\n{text}"
    )


def assert_substrings_in_order(text: str, *labels: str) -> None:
    positions = [text.index(label) for label in labels]
    assert positions == sorted(positions), (
        f"expected labels in order {labels!r}, positions were {positions!r} in:\n{text}"
    )
