"""Shared YAML scalar and list validators."""

from __future__ import annotations

from typing import Any


def require_mapping(raw: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{context} must be a YAML mapping")
    return raw


def require_int(raw: Any, *, context: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ValueError(f"{context} must be an integer, got {raw!r}")
    return raw


def require_float(raw: Any, *, context: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{context} must be a number, got {raw!r}")
    return float(raw)


def require_str(raw: Any, *, context: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{context} must be a non-empty string, got {raw!r}")
    return raw


def require_str_list(raw: Any, *, context: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{context} must be a non-empty list of strings")
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item:
            raise ValueError(f"{context} entries must be non-empty strings")
        out.append(item)
    return tuple(out)


def require_float_list(raw: Any, *, context: str) -> tuple[float, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{context} must be a non-empty list of numbers")
    return tuple(float(v) for v in raw)
