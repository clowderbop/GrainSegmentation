import math
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf

from common.patching import build_coverage_bin_ids, compute_starts_tf, region_bounds
from common.samples import (
    list_samples,  # noqa: F401
    load_rgb_image,
    load_raster_mask,
    validate_loaded_sample,
    validate_mask_labels,
)

MIN_VALIDATION_COVERAGE = 0.10


def _tf_augment(
    inputs: Tuple[tf.Tensor, ...], label: tf.Tensor
) -> Tuple[Tuple[tf.Tensor, ...], tf.Tensor]:
    num_inputs = len(inputs)

    concat = tf.concat(list(inputs) + [label], axis=-1)

    seed_lr = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)
    concat = tf.cond(
        seed_lr < 0.5, lambda: tf.image.flip_left_right(concat), lambda: concat
    )

    seed_ud = tf.random.uniform(shape=[], minval=0.0, maxval=1.0)
    concat = tf.cond(
        seed_ud < 0.5, lambda: tf.image.flip_up_down(concat), lambda: concat
    )


    k = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    concat = tf.image.rot90(concat, k=k)


    split_sizes = [3] * num_inputs + [3]
    splits = tf.split(concat, split_sizes, axis=-1)

    aug_inputs = list(splits[:-1])
    aug_label = splits[-1]


    final_inputs = []
    for img in aug_inputs:
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)


        img = tf.where(img < 0.0, tf.zeros_like(img), img)
        img = tf.where(img > 1.0, tf.ones_like(img), img)
        final_inputs.append(img)

    return tuple(final_inputs), aug_label


def _validate_sample_inputs(samples: List[Dict[str, Any]], num_inputs: int) -> None:
    for sample in samples:
        if len(sample["images"]) != num_inputs:
            raise ValueError("Mismatch between num_inputs and loaded images.")


def _build_region_samples_and_coverages(
    samples: List[Dict[str, Any]], tile_size: int
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    if tile_size <= 0:
        raise ValueError("tile_size must be > 0")

    region_samples: List[Dict[str, Any]] = []
    coverages: List[float] = []

    for sample in samples:
        mask = load_raster_mask(sample["mask"])
        height, width = mask.shape
        regions = region_bounds(height, width, tile_size)
        if not regions:
            continue
        for y0, y1, x0, x1 in regions:
            cov = float(np.mean(mask[y0:y1, x0:x1] > 0))
            coverages.append(cov)

            region_sample = dict(sample)
            region_sample["region"] = (y0, y1, x0, x1)
            region_samples.append(region_sample)

    coverages_arr = np.array(coverages, dtype=np.float32)
    return region_samples, coverages_arr


def create_spatial_holdout_split(
    samples: List[Dict[str, Any]],
    tile_size: int,
    validation_fraction: float,
    random_state: int,
    coverage_bins: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

    if validation_fraction <= 0 or validation_fraction >= 1:
        raise ValueError("validation_fraction must be in (0, 1).")

    region_samples, coverages_arr = _build_region_samples_and_coverages(
        samples, tile_size
    )
    eligible_mask = coverages_arr >= MIN_VALIDATION_COVERAGE
    train_only_mask = ~eligible_mask
    eligible_indices = np.flatnonzero(eligible_mask)
    train_only_indices = np.flatnonzero(train_only_mask)
    total = eligible_indices.shape[0]

    if total < 2:
        raise ValueError(
            "Not enough validation-eligible regions "
            f"({total}) to create a train/validation split."
        )

    val_count = math.ceil(total * validation_fraction)
    val_count = max(1, min(val_count, total - 1))
    group_count = max(2, math.ceil(1 / validation_fraction))
    bin_ids = build_coverage_bin_ids(
        coverages_arr[eligible_indices],
        coverage_bins=coverage_bins,
        group_count=group_count,
    )

    counts = np.bincount(bin_ids)
    can_stratify = (
        counts.min() >= 2
        and np.count_nonzero(counts) <= val_count
        and np.count_nonzero(counts) <= (total - val_count)
    )
    if can_stratify:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=val_count, random_state=random_state
        )
    else:
        splitter = ShuffleSplit(
            n_splits=1, test_size=val_count, random_state=random_state
        )

    train_idx, val_idx = next(splitter.split(np.zeros(total), bin_ids))
    selected_train_indices = eligible_indices[train_idx]
    selected_val_indices = eligible_indices[val_idx]
    train_samples = [region_samples[i] for i in selected_train_indices]
    train_samples.extend(region_samples[i] for i in train_only_indices)
    val_samples = [region_samples[i] for i in selected_val_indices]
    return train_samples, val_samples


def build_dataset(
    samples: List[Dict[str, Any]],
    patch_size: int,
    stride: int,
    batch_size: int,
    augment: bool,
    num_inputs: int,
    cache_file: str | None = None,
) -> tf.data.Dataset:
    if not samples:
        raise ValueError("samples list must not be empty")
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be > 0")
    _validate_sample_inputs(samples, num_inputs)

    images_list = [s["images"] for s in samples]
    mask_list = [s["mask"] for s in samples]
    region_list = [s.get("region", (0, 0, 0, 0)) for s in samples]
    has_region_list = [("region" in s) for s in samples]

    ds = tf.data.Dataset.from_tensor_slices(
        (images_list, mask_list, region_list, has_region_list)
    )

    def _load_sample_py(
        image_paths_tensor, mask_path_tensor, region_tensor, has_region_tensor
    ):
        image_paths = [p.decode("utf-8") for p in image_paths_tensor.numpy()]
        mask_path = mask_path_tensor.numpy().decode("utf-8")
        region = region_tensor.numpy()
        has_region = has_region_tensor.numpy()

        if len(image_paths) != num_inputs:
            raise ValueError("Mismatch between num_inputs and loaded images.")

        images = [load_rgb_image(p) for p in image_paths]
        mask = load_raster_mask(mask_path)
        validate_loaded_sample(images, mask, mask_path)

        if has_region:
            y0, y1, x0, x1 = region
            images = [img[y0:y1, x0:x1, :] for img in images]
            mask = mask[y0:y1, x0:x1]

        return tuple(images) + (validate_mask_labels(mask, mask_path),)

    def _py_wrapper(image_paths, mask_path, region, has_region):
        res = tf.py_function(
            func=_load_sample_py,
            inp=[image_paths, mask_path, region, has_region],
            Tout=[tf.float32] * num_inputs + [tf.int32],
        )
        for i in range(num_inputs):
            res[i].set_shape([None, None, 3])
        res[-1].set_shape([None, None])
        return tuple(res[:num_inputs]), res[-1]

    ds = ds.map(_py_wrapper, num_parallel_calls=tf.data.AUTOTUNE)

    def pad_fn(images_tuple, mask):
        shape = tf.shape(mask)
        height, width = shape[0], shape[1]

        pad_h = tf.maximum(0, patch_size - height)
        pad_w = tf.maximum(0, patch_size - width)

        paddings_img = [[0, pad_h], [0, pad_w], [0, 0]]
        paddings_mask = [[0, pad_h], [0, pad_w]]

        padded_images = tuple(
            [
                tf.pad(img, paddings_img, mode="CONSTANT", constant_values=0.0)
                for img in images_tuple
            ]
        )
        padded_mask = tf.pad(mask, paddings_mask, mode="CONSTANT", constant_values=0)

        return padded_images, padded_mask

    ds = ds.map(pad_fn, num_parallel_calls=tf.data.AUTOTUNE)

    def extract_patches_fn(images_tuple, mask):
        shape = tf.shape(mask)
        height = shape[0]
        width = shape[1]

        y_starts = compute_starts_tf(
            height,
            tf.constant(patch_size, dtype=tf.int32),
            tf.constant(stride, dtype=tf.int32),
        )
        x_starts = compute_starts_tf(
            width,
            tf.constant(patch_size, dtype=tf.int32),
            tf.constant(stride, dtype=tf.int32),
        )

        Y, X = tf.meshgrid(y_starts, x_starts, indexing="ij")
        coords = tf.stack([tf.reshape(Y, [-1]), tf.reshape(X, [-1])], axis=1)

        coord_ds = tf.data.Dataset.from_tensor_slices(coords)

        def crop_patch(coord):
            y = coord[0]
            x = coord[1]

            patch_images = tuple(
                [
                    tf.image.crop_to_bounding_box(img, y, x, patch_size, patch_size)
                    for img in images_tuple
                ]
            )
            patch_mask = tf.image.crop_to_bounding_box(
                tf.expand_dims(mask, -1), y, x, patch_size, patch_size
            )
            patch_mask = tf.squeeze(patch_mask, axis=-1)

            patch_label = tf.one_hot(patch_mask, depth=3, dtype=tf.float32)

            return patch_images, patch_label

        return coord_ds.map(crop_patch)

    ds = ds.flat_map(extract_patches_fn)

    if cache_file:
        ds = ds.cache(cache_file)
    else:
        ds = ds.cache()

    if augment:
        ds = ds.map(
            _tf_augment, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False
        )
        ds = ds.shuffle(buffer_size=100)

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
