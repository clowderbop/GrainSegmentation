"""Connected-components instance extraction from semantic label maps."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import generate_binary_structure, label

from common.labeled_components import drop_small_components

Connectivity = Literal[1, 2]


def _structure_for_connectivity(ndim: int, connectivity: Connectivity) -> np.ndarray:
    if connectivity not in (1, 2):
        raise ValueError(f"connectivity must be 1 or 2, got {connectivity}")
    return generate_binary_structure(ndim, connectivity)


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
        return drop_small_components(labeled, min_area_px)
    return labeled
