"""Connected-components instance extraction from semantic label maps."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import generate_binary_structure, label

Connectivity = Literal[1, 2]


def _structure_for_connectivity(ndim: int, connectivity: Connectivity) -> np.ndarray:
    if connectivity not in (1, 2):
        raise ValueError(f"connectivity must be 1 or 2, got {connectivity}")
    return generate_binary_structure(ndim, connectivity)


def _relabel_sequential(labeled: np.ndarray) -> np.ndarray:
    ids = sorted(x for x in np.unique(labeled) if x != 0)
    if not ids:
        return np.zeros_like(labeled)
    out = np.zeros_like(labeled)
    for new_id, old_id in enumerate(ids, start=1):
        out[labeled == old_id] = new_id
    return out


def _drop_small_components(labeled: np.ndarray, min_area_px: int) -> np.ndarray:
    if min_area_px <= 0:
        return labeled
    out = labeled.copy()
    max_id = int(labeled.max())
    for lid in range(1, max_id + 1):
        if (labeled == lid).sum() < min_area_px:
            out[labeled == lid] = 0
    return _relabel_sequential(out)


def semantic_to_instance_label_map(
    semantic: np.ndarray,
    *,
    interior_class: int = 1,
    connectivity: Connectivity = 1,
    min_area_px: int = 0,
) -> np.ndarray:
    if semantic.ndim != 2:
        raise ValueError(f"semantic must be 2D, got shape {semantic.shape}")
    interior = semantic == interior_class
    structure = _structure_for_connectivity(semantic.ndim, connectivity)
    labeled, _ = label(interior, structure=structure)
    if min_area_px > 0:
        return _drop_small_components(labeled, min_area_px)
    return labeled
