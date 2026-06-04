"""Thesis-order helpers for input configuration display names."""

from __future__ import annotations

from collections.abc import Iterable

from common.variants import variant_display_names_in_thesis_order


def thesis_ordered_display_names(names: Iterable[str]) -> list[str]:
    """Order display names in thesis registry order, then alphabetically for extras."""
    name_set = set(names)
    thesis_order = list(variant_display_names_in_thesis_order())
    present = [name for name in thesis_order if name in name_set]
    extra = sorted(name_set - set(present))
    return present + extra
