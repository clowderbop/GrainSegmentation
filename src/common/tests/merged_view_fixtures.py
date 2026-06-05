"""Shared merged-instance-view maps for common metrics tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def blank_map(height: int, width: int) -> np.ndarray:
    return np.zeros((height, width), dtype=np.int32)


def paint_box(
    instance_map: np.ndarray, label: int, y0: int, x0: int, y1: int, x1: int
) -> None:
    instance_map[y0:y1, x0:x1] = label


def bundle_fixture_perfect_single() -> tuple[np.ndarray, np.ndarray]:
    gt = blank_map(32, 32)
    pred = blank_map(32, 32)
    paint_box(gt, 1, 4, 4, 20, 20)
    paint_box(pred, 1, 4, 4, 20, 20)
    return gt, pred


def bundle_fixture_both_empty() -> tuple[np.ndarray, np.ndarray]:
    return blank_map(16, 16), blank_map(16, 16)


def bundle_fixture_empty_pred() -> tuple[np.ndarray, np.ndarray]:
    gt = blank_map(24, 24)
    pred = blank_map(24, 24)
    paint_box(gt, 1, 2, 2, 14, 14)
    paint_box(gt, 2, 16, 16, 22, 22)
    return gt, pred


def bundle_fixture_empty_gt() -> tuple[np.ndarray, np.ndarray]:
    gt = blank_map(20, 20)
    pred = blank_map(20, 20)
    paint_box(pred, 1, 4, 4, 16, 16)
    return gt, pred


def bundle_fixture_missed_grain() -> tuple[np.ndarray, np.ndarray]:
    gt = blank_map(32, 32)
    pred = blank_map(32, 32)
    paint_box(gt, 1, 4, 4, 14, 14)
    paint_box(gt, 2, 18, 18, 28, 28)
    paint_box(pred, 1, 4, 4, 14, 14)
    return gt, pred


def bundle_fixture_duplicate_preds() -> tuple[np.ndarray, np.ndarray]:
    gt = blank_map(32, 32)
    pred = blank_map(32, 32)
    paint_box(gt, 1, 6, 6, 22, 22)
    paint_box(pred, 1, 6, 6, 22, 22)
    paint_box(pred, 2, 8, 8, 20, 20)
    return gt, pred


def bundle_fixture_split_merge() -> tuple[np.ndarray, np.ndarray]:
    gt = blank_map(48, 48)
    pred = blank_map(48, 48)
    paint_box(gt, 1, 10, 10, 20, 20)
    paint_box(gt, 2, 28, 28, 38, 38)
    paint_box(pred, 1, 10, 10, 20, 20)
    paint_box(pred, 1, 20, 20, 28, 28)
    paint_box(pred, 1, 28, 28, 32, 32)
    return gt, pred


def bundle_fixture_poor_mask() -> tuple[np.ndarray, np.ndarray]:
    gt = blank_map(32, 32)
    pred = blank_map(32, 32)
    paint_box(gt, 1, 8, 8, 24, 24)
    paint_box(pred, 1, 8, 8, 18, 18)
    return gt, pred


def bundle_fixture_pq_decomposition() -> tuple[np.ndarray, np.ndarray]:
    gt = blank_map(48, 48)
    pred = blank_map(48, 48)
    paint_box(gt, 1, 4, 4, 18, 18)
    paint_box(gt, 2, 28, 28, 44, 44)
    paint_box(pred, 1, 4, 4, 18, 18)
    paint_box(pred, 2, 28, 28, 44, 44)
    paint_box(pred, 3, 4, 28, 18, 44)
    return gt, pred


def bundle_fixture_aji_plus_duplicates() -> tuple[np.ndarray, np.ndarray]:
    gt = blank_map(24, 24)
    pred = blank_map(24, 24)
    paint_box(gt, 1, 4, 4, 16, 16)
    paint_box(pred, 1, 4, 4, 16, 16)
    paint_box(pred, 2, 6, 6, 12, 12)
    return gt, pred


BUNDLE_FIXTURE_BUILDERS: dict[str, Callable[[], tuple[np.ndarray, np.ndarray]]] = {
    "perfect_single": bundle_fixture_perfect_single,
    "both_empty": bundle_fixture_both_empty,
    "empty_pred": bundle_fixture_empty_pred,
    "empty_gt": bundle_fixture_empty_gt,
    "missed_grain": bundle_fixture_missed_grain,
    "duplicate_preds": bundle_fixture_duplicate_preds,
    "split_merge": bundle_fixture_split_merge,
    "poor_mask": bundle_fixture_poor_mask,
    "pq_decomposition": bundle_fixture_pq_decomposition,
    "aji_plus_duplicates": bundle_fixture_aji_plus_duplicates,
}


def get_bundle_fixture(name: str) -> tuple[np.ndarray, np.ndarray]:
    return BUNDLE_FIXTURE_BUILDERS[name]()
