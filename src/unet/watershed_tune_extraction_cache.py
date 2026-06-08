"""In-memory tune-path cache for watershed semantic prep and base label maps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from common.instance_overlap import GtOverlapPrep, gt_overlap_prep
from unet.extraction_tune_scoring import (
    WatershedParamSet,
    watershed_tune_sample_prefix,
    mean_train_pq_for_watershed_params,
)
from unet.instance_masks import (
    WatershedSemanticPrep,
    build_watershed_semantic_prep,
    watershed_area_filter,
    watershed_base_extraction,
)
from unet.watershed_tune_grid import WatershedTuneGrid, iter_watershed_tune_param_sets


def log_extraction_cache_lookup(
    *,
    hit: bool,
    sample_id: str | None = None,
    prefix: str = "",
) -> None:
    status = "hit" if hit else "miss"
    sid = f" ({sample_id})" if sample_id else ""
    print(f"{prefix}    extraction cache: {status}{sid}", flush=True)


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
        base, _ = self.lookup_base_label_map(key)
        return base

    def lookup_base_label_map(
        self, key: WatershedBaseExtractionKey
    ) -> tuple[np.ndarray, bool]:
        cached = self._base_maps.get(key)
        if cached is not None:
            return cached, True
        base = watershed_base_extraction(
            self._prep,
            min_distance=key.min_distance,
            exclude_border=key.exclude_border,
            boundary_dilate_iter=key.boundary_dilate_iter,
            watershed_connectivity=key.watershed_connectivity,
            ridge_level=key.ridge_level,
        )
        self._base_maps[key] = base
        return base, False


def build_watershed_tune_sample_caches(
    pred_semantic_per_sample: Sequence[np.ndarray],
) -> list[WatershedTuneSampleCache]:
    return [WatershedTuneSampleCache(semantic) for semantic in pred_semantic_per_sample]


def build_gt_overlap_preps(
    true_instances_per_sample: Sequence[np.ndarray],
) -> list[GtOverlapPrep]:
    """Build per-sample GT overlap metadata once per tune job.

    Mirrors ``build_watershed_tune_sample_caches``: tune ``main()`` calls this
    once and threads the result through every grid combo. Saves repeated GT
    ``instance_ids`` + ``bincount`` bookkeeping; pred-side overlap extraction
    and the O(pixels) co-occurrence scan still run per combo.

    **Profiling rationale (issue 04):** On the train whole section, metrics
    time is dominated by pred-side work (tens of thousands of instances, full
    raster scan per combo). GT bookkeeping is a small fraction of
    ``instance_overlap_stats``, so this is a modest incremental win — not
    comparable to the 24× watershed extraction cache. Shipped because cost is
    trivial, parity-tested, and avoids 71 redundant GT ``bincount`` passes per
    sample on a 72-combo grid with no semantic risk.
    """
    return [gt_overlap_prep(true_instances) for true_instances in true_instances_per_sample]


def instance_map_from_tune_cache(
    cache: WatershedTuneSampleCache,
    params: WatershedParamSet,
    *,
    log_extraction_cache: bool = False,
    sample_id: str | None = None,
    log_prefix: str = "",
) -> np.ndarray:
    key = base_extraction_key_from_params(params)
    base, hit = cache.lookup_base_label_map(key)
    if log_extraction_cache:
        log_extraction_cache_lookup(
            hit=hit, sample_id=sample_id, prefix=log_prefix
        )
    return watershed_area_filter(base, params.min_area_px)


def mean_train_pq_for_watershed_params_cached(
    true_instances_per_sample: Sequence[np.ndarray],
    sample_caches: Sequence[WatershedTuneSampleCache],
    params: WatershedParamSet,
    *,
    gt_overlap_preps: Sequence[GtOverlapPrep] | None = None,
    sample_ids: Sequence[str] | None = None,
    log: bool = False,
    log_extraction_cache: bool = False,
) -> tuple[dict[str, float | int], list[dict[str, float | int]]]:
    if len(true_instances_per_sample) != len(sample_caches):
        raise ValueError("true instances and sample caches must have the same length")
    n_samples = len(true_instances_per_sample)

    def get_pred_instances(idx: int, p: WatershedParamSet) -> np.ndarray:
        prefix = watershed_tune_sample_prefix(
            idx, n_samples, sample_ids, log=log_extraction_cache
        )
        sid = sample_ids[idx] if sample_ids is not None else None
        return instance_map_from_tune_cache(
            sample_caches[idx],
            p,
            log_extraction_cache=log_extraction_cache,
            sample_id=sid,
            log_prefix=prefix,
        )

    return mean_train_pq_for_watershed_params(
        true_instances_per_sample,
        None,
        params,
        get_pred_instances=get_pred_instances,
        gt_overlap_preps=gt_overlap_preps,
        sample_ids=sample_ids,
        log=log,
    )
