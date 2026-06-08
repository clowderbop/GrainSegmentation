import numpy as np
import tensorflow as tf

from common.patching import compute_starts


def predict_full_image(
    model: tf.keras.Model,
    inputs: tuple[np.ndarray, ...],
    patch_size: int = 1024,
    stride: int = 512,
    batch_size: int = 4,
):
    if not inputs:
        raise ValueError("inputs must contain at least one image array")
    if patch_size <= 0 or stride <= 0 or batch_size <= 0:
        raise ValueError("patch_size, stride, and batch_size must be > 0")
    if stride > patch_size:
        raise ValueError("stride must be <= patch_size")

    base_shape = inputs[0].shape
    if len(base_shape) != 3:
        raise ValueError("Each input image must have shape (H, W, C)")

    h, w = base_shape[:2]
    for img in inputs[1:]:
        if img.shape != base_shape:
            raise ValueError("All input images must share the same shape")

    y_starts = compute_starts(h, patch_size, stride)
    x_starts = compute_starts(w, patch_size, stride)

    padded_h = max(h, patch_size)
    padded_w = max(w, patch_size)

    padded_inputs = []
    for img in inputs:
        pad_img = np.pad(
            img, ((0, padded_h - h), (0, padded_w - w), (0, 0)), mode="constant"
        )
        padded_inputs.append(pad_img)

    num_classes = model.output_shape[-1]

    prediction_map = np.zeros((padded_h, padded_w, num_classes), dtype=np.float32)
    weight_map = np.zeros((padded_h, padded_w, 1), dtype=np.float32)

    y_window = np.hanning(patch_size)
    x_window = np.hanning(patch_size)
    window = np.outer(y_window, x_window)[..., np.newaxis]
    window = np.maximum(window, 1e-4)

    patches = []
    coords = []

    for y in y_starts:
        for x in x_starts:
            patch_tuple = [
                img[y : y + patch_size, x : x + patch_size, :] for img in padded_inputs
            ]
            patches.append(patch_tuple)
            coords.append((y, x))

    num_inputs = len(inputs)
    num_patches = len(patches)
    num_batches = (num_patches + batch_size - 1) // batch_size
    print(
        f"Sliding-window inference: {num_patches} patches in {num_batches} batches "
        f"(image {h}x{w}, patch_size={patch_size}, stride={stride}, batch_size={batch_size})"
    )
    next_report_pct = 0.0

    for batch_idx, i in enumerate(range(0, num_patches, batch_size)):
        batch_patches = patches[i : i + batch_size]
        batch_coords = coords[i : i + batch_size]

        model_inputs = []
        for inp_idx in range(num_inputs):
            inp_batch = np.stack([p[inp_idx] for p in batch_patches], axis=0)
            model_inputs.append(inp_batch)

        if num_inputs == 1:
            preds = model.predict(model_inputs[0], verbose=0)
        else:
            preds = model.predict(model_inputs, verbose=0)

        for b_idx, (y, x) in enumerate(batch_coords):
            prediction_map[y : y + patch_size, x : x + patch_size] += (
                preds[b_idx] * window
            )
            weight_map[y : y + patch_size, x : x + patch_size] += window

        progress = (batch_idx + 1) / num_batches
        if progress >= next_report_pct or batch_idx == num_batches - 1:
            print(
                f"  inference batches {batch_idx + 1}/{num_batches} "
                f"({100.0 * progress:.0f}%)"
            )
            while next_report_pct <= progress:
                next_report_pct += 0.1

    prediction_map /= weight_map

    prediction_map = prediction_map[:h, :w, :]

    pred_classes = np.argmax(prediction_map, axis=-1)

    return pred_classes, prediction_map
