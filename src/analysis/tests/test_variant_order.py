"""Thesis-order display name helpers."""

from analysis.variant_order import thesis_ordered_display_names


def test_thesis_ordered_display_names_puts_registry_first() -> None:
    assert thesis_ordered_display_names(["FullStack", "PPL", "Extra"]) == [
        "PPL",
        "FullStack",
        "Extra",
    ]
