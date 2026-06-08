"""In-memory tune-path cache for watershed semantic prep and base label maps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from unet.extraction_tune_scoring import (
    WatershedParamSet,
    mean_train_pq_for_watershed_params,
)
from unet.instance_masks import (
    WatershedSemanticPrep,
    build_watershed_semantic_prep,
    watershed_area_filter,
    watershed_base_extraction,
)
from unet.watershed_tune_grid import WatershedTuneGrid, iter_watershed_tune_param_sets


@dataclass(frozen=True)
class WatershedBaseExtractionKey:
    min_distance: int
    exclude_border: bool
    boundary_dilate_iter: int
    watershed_connectivity: int
    ridge_level: float | None


def iter_unique_watershed_base_extraction_keys(
    grid: WatershedTuneGrid,
) -> list[WatershedBaseExtractionKey]:
    seen: set[WatershedBaseExtractionKey] = set()
    keys: list[WatershedBaseExtractionKey] = []
    for params in iter_watershed_tune_param_sets(grid):
        key = base_extraction_key_from_params(params)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def watershed_base_extraction_key_count(grid: WatershedTuneGrid) -> int:
    return len(iter_unique_watershed_base_extraction_keys(grid))


def base_extraction_key_from_params(params: WatershedParamSet) -> WatershedBaseExtractionKey:
    return WatershedBaseExtractionKey(
        min_distance=params.min_distance,
        exclude_border=params.exclude_border,
        boundary_dilate_iter=params.boundary_dilate_iter,
        watershed_connectivity=params.watershed_connectivity,
        ridge_level=params.ridge_level,
    )


class WatershedTuneSampleCache:
    """Per-sample prep plus on-demand base label maps.

    Retains ``WatershedSemanticPrep`` for the job lifetime and computes each
    unique base-extraction key on first use. Peak memory is one prep plus at
    most one int32 label map per unique grid base key touched during the job
    (24 for the default grid). On the train whole section (~520M pixels), 24
    int32 maps is on the order of 50 GB — well within the 256G tune SLURM
    allocation; larger grids or rasters should profile before relying on full
    retention.
    """

    def __init__(self, pred_semantic: np.ndarray) -> None:
        self._semantic = pred_semantic
        self._prep = build_watershed_semantic_prep(pred_semantic)
        self._base_maps: dict[WatershedBaseExtractionKey, np.ndarray] = {}

    @property
    def semantic(self) -> np.ndarray:
        return self._semantic

    @property
    def prep(self) -> WatershedSemanticPrep:
        return self._prep

    def base_label_map(self, key: WatershedBaseExtractionKey) -> np.ndarray:
        cached = self._base_maps.get(key)
        if cached is not None:
            return cached
        base = watershed_base_extraction(
            self._prep,
            min_distance=key.min_distance,
            exclude_border=key.exclude_border,
            boundary_dilate_iter=key.boundary_dilate_iter,
            watershed_connectivity=key.watershed_connectivity,
            ridge_level=key.ridge_level,
        )
        self._base_maps[key] = base
        return base


def build_watershed_tune_sample_caches(
    pred_semantic_per_sample: Sequence[np.ndarray],
) -> list[WatershedTuneSampleCache]:
    return [WatershedTuneSampleCache(semantic) for semantic in pred_semantic_per_sample]


def instance_map_from_tune_cache(
    cache: WatershedTuneSampleCache,
    params: WatershedParamSet,
) -> np.ndarray:
    base = cache.base_label_map(base_extraction_key_from_params(params))
    return watershed_area_filter(base, params.min_area_px)


def mean_train_pq_for_watershed_params_cached(
    true_instances_per_sample: Sequence[np.ndarray],
    sample_caches: Sequence[WatershedTuneSampleCache],
    params: WatershedParamSet,
    *,
    sample_ids: Sequence[str] | None = None,
    log: bool = False,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    if len(true_instances_per_sample) != len(sample_caches):
        raise ValueError("true instances and sample caches must have the same length")
    return mean_train_pq_for_watershed_params(
        true_instances_per_sample,
        [cache.semantic for cache in sample_caches],
        params,
        get_pred_instances=lambda idx, p: instance_map_from_tune_cache(
            sample_caches[idx], p
        ),
        sample_ids=sample_ids,
        log=log,
    )
