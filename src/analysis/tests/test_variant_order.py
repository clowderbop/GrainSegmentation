"""Thesis-order display name helpers."""

from analysis.variant_order import thesis_ordered_display_names


def test_thesis_ordered_display_names_puts_registry_first() -> None:
    """INTENT: thesis display name ordering places registry-known names before extras."""
    assert thesis_ordered_display_names(["FullStack", "PPL", "Extra"]) == [
        "PPL",
        "FullStack",
        "Extra",
    ]
