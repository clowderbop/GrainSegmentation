"""Shared helpers for relabeling and filtering connected-component label maps."""

from __future__ import annotations

import numpy as np


def relabel_sequential(labeled: np.ndarray) -> np.ndarray:
    mask = labeled != 0
    if not mask.any():
        return np.zeros_like(labeled)
    _, inv = np.unique(labeled[mask], return_inverse=True)
    out = np.zeros_like(labeled)
    out[mask] = inv.astype(np.int32) + 1
    return out


def drop_small_components(labeled: np.ndarray, min_area_px: int) -> np.ndarray:
    if min_area_px <= 0 or not np.any(labeled):
        return labeled
    ids, counts = np.unique(labeled, return_counts=True)
    drop_ids = ids[(ids != 0) & (counts < min_area_px)]
    if drop_ids.size == 0:
        return relabel_sequential(labeled)
    out = labeled.copy()
    out[np.isin(labeled, drop_ids)] = 0
    return relabel_sequential(out)
