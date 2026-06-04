"""Thesis-facing producer labels shared by derived tables and figures."""

from __future__ import annotations

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "yolo": "YOLO",
    "unet": "U-Net",
}
MODEL_AXIS_LABEL = "Model"
MODEL_LEGEND_ORDER: tuple[str, ...] = ("YOLO", "U-Net")


def model_display_name(producer: str) -> str:
    """Thesis-facing label for a producer family (data rows keep `producer`)."""
    return MODEL_DISPLAY_NAMES.get(producer, producer)
