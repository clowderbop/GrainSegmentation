"""Shared helpers for relabeling and filtering connected-component label maps."""

from __future__ import annotations

import numpy as np


def relabel_sequential(labeled: np.ndarray) -> np.ndarray:
    ids = sorted(x for x in np.unique(labeled) if x != 0)
    if not ids:
        return np.zeros_like(labeled)
    out = np.zeros_like(labeled)
    for new_id, old_id in enumerate(ids, start=1):
        out[labeled == old_id] = new_id
    return out


def drop_small_components(labeled: np.ndarray, min_area_px: int) -> np.ndarray:
    if min_area_px <= 0:
        return labeled
    out = labeled.copy()
    max_id = int(labeled.max())
    for lid in range(1, max_id + 1):
        if (labeled == lid).sum() < min_area_px:
            out[labeled == lid] = 0
    return relabel_sequential(out)
