"""Figure-facing labels for producer families."""

from analysis.figures import MODEL_AXIS_LABEL, model_display_name


def test_model_display_name_maps_producers() -> None:
    assert model_display_name("yolo") == "YOLO"
    assert model_display_name("unet") == "U-Net"


def test_model_axis_label() -> None:
    assert MODEL_AXIS_LABEL == "Model"
