from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


TIFF_SUFFIXES = {".tif", ".tiff"}


def as_uint8_image(array: np.ndarray, *, clip: bool = True) -> np.ndarray:
    if clip:
        array = np.clip(array, 0, 255)
    return array.astype(np.uint8, copy=False)


def to_channel_first_uint8(
    image: np.ndarray,
    *,
    exact_channels: int | None = None,
    channel_multiple: int | None = None,
    description: str = "image",
    prefer_channel_last: bool = False,
) -> np.ndarray:
    if image.ndim != 3:
        raise ValueError(f"Expected {description} with 3 dimensions, got shape {image.shape}.")

    image_uint8 = as_uint8_image(image)

    def _valid(channel_count: int) -> bool:
        if exact_channels is not None:
            return channel_count == exact_channels
        if channel_multiple is not None:
            return channel_count > 0 and channel_count % channel_multiple == 0
        return True

    axis_order = (-1, 0) if prefer_channel_last else (0, -1)
    for axis in axis_order:
        if _valid(image_uint8.shape[axis]):
            if axis == 0:
                return image_uint8
            return np.transpose(image_uint8, (2, 0, 1))

    expected = (
        f"{exact_channels} channels"
        if exact_channels is not None
        else f"a channel count divisible by {channel_multiple}"
        if channel_multiple is not None
        else "channels"
    )
    raise ValueError(
        f"Expected {description} with {expected} in either channel-first or "
        f"channel-last order, got shape {image.shape}."
    )


def load_tiff_channel_first(path: str | Path) -> np.ndarray:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        image = tif.asarray()
        axes = tif.series[0].axes

    if image.ndim != 3:
        return image
    if axes == "YXS":
        return np.transpose(image, (2, 0, 1))
    if axes in {"SYX", "CYX"}:
        return image
    if axes == "QYX":
        first, middle, last = image.shape
        if first < middle and last >= middle:
            return image
        if last < middle and first >= middle:
            return np.transpose(image, (2, 0, 1))
        raise ValueError(
            f"TIFF layout cannot be inferred safely from axes={axes!r} and shape={image.shape}"
        )
    raise ValueError(f"Unsupported 3D TIFF axes {axes!r} for shape {image.shape}")


def load_tiff_rgb_hwc_float(path: str | Path) -> np.ndarray:
    arr = load_tiff_channel_first(path)
    if arr.ndim != 3:
        raise ValueError(
            f"Expected RGB TIFF with 3 dimensions, got shape {arr.shape} for {path}"
        )
    if arr.shape[0] != 3:
        raise ValueError(
            f"Expected 3 channels (RGB) in channel-first order after load, "
            f"got C={arr.shape[0]} for {path}"
        )
    hwc = np.transpose(arr, (1, 2, 0))
    return hwc.astype(np.float32) / 255.0


def load_tiff_single_channel_mask(path: str | Path) -> np.ndarray:
    import tifffile

    mask_path = Path(path)
    if mask_path.suffix.lower() not in TIFF_SUFFIXES:
        raise ValueError(
            f"Mask must be a TIFF (.tif / .tiff), got suffix {mask_path.suffix!r} "
            f"for {mask_path}"
        )

    with tifffile.TiffFile(mask_path) as tif:
        arr = tif.series[0].asarray()

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            raise ValueError(f"Mask TIFF must be single-channel: {mask_path}")
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D: {mask_path}")
    return arr


def load_single_channel_mask(path: str | Path, *, allow_tiff: bool = True) -> np.ndarray:
    mask_path = Path(path)
    if allow_tiff and mask_path.suffix.lower() in TIFF_SUFFIXES:
        return load_tiff_single_channel_mask(mask_path)
    else:
        with Image.open(mask_path) as img:
            if img.mode not in ("L", "I", "I;16", "F"):
                img = img.convert("L")
            arr = np.asarray(img)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[2] == 1:
            arr = arr[:, :, 0]
        else:
            raise ValueError(f"Mask must be single-channel: {mask_path}")
    if arr.ndim != 2:
        raise ValueError(f"Mask must be 2D: {mask_path}")
    return arr


def validate_semantic_labels(
    mask: np.ndarray,
    mask_path: str | Path,
    *,
    min_label: int = 0,
    max_label: int = 2,
    allow_float: bool = False,
) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D: {mask_path}")
    if allow_float and np.issubdtype(mask.dtype, np.floating):
        if not np.all(np.isfinite(mask)):
            raise ValueError(f"Mask must be finite: {mask_path}")
        rounded = np.rint(mask)
        if not np.allclose(mask, rounded, rtol=0.0, atol=1e-4):
            raise ValueError(
                f"Mask must have integer labels in [{min_label}, {max_label}]: {mask_path}"
            )
        mask_int = rounded.astype(np.int32)
    else:
        mask_int = mask.astype(np.int32)
        if not np.all(mask == mask_int):
            raise ValueError(
                f"Mask values must be integers in [{min_label}, {max_label}] for {mask_path}"
            )
    if np.any((mask_int < min_label) | (mask_int > max_label)):
        raise ValueError(f"Mask values must be in [{min_label}, {max_label}] for {mask_path}")
    return mask_int


def validate_image_mask_sample(
    images: list[np.ndarray], mask: np.ndarray, mask_path: str | Path
) -> None:
    if not images:
        raise ValueError("Sample must contain at least one input image.")

    expected_shape = images[0].shape
    if len(expected_shape) != 3:
        raise ValueError("All input images must have shape (H, W, C).")
    for img in images[1:]:
        if img.shape != expected_shape:
            raise ValueError("All input images must share the same shape.")

    if mask.ndim != 2:
        raise ValueError(f"Raster mask must be 2D: {mask_path}")

    image_shape = expected_shape[:2]
    if mask.shape != image_shape:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image shape {image_shape} "
            f"for {mask_path}"
        )
