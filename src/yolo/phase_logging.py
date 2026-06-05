"""Shared phase progress logging for YOLO whole-section hot paths."""

from __future__ import annotations

PHASE_ENRICH_PROPOSALS = "Enriching proposals"
PHASE_BUILD_CANDIDATE_PAIRS = "Building candidate pairs"
PHASE_MERGE_PREDICTIONS = "Merging predictions"
PHASE_CROSS_TILE_ASSOCIATION = "Cross-tile association"
PHASE_RASTERIZE_MERGED_VIEW = "Rasterizing merged instance view"
PHASE_PREDICTING_TILES = "Predicting tiles"
PHASE_LOADING_PROPOSALS = "Loading tiled detector proposals"
PHASE_EVALUATING_TRAIN_PQ = "Evaluating train PQ"
NESTED_CROSS_TILE_ASSOCIATION = "cross-tile association"
NESTED_METRICS = "metrics"


def log_phase_start(phase: str, *, prefix: str = "") -> None:
    print(f"{prefix}{phase} …", flush=True)


def log_phase_done(
    phase: str, elapsed_s: float, *, prefix: str = "", detail: str = ""
) -> None:
    line = f"{prefix}{phase} done {elapsed_s:.1f}s"
    if detail:
        line += f" — {detail}"
    print(line, flush=True)


def log_nested_phase_start(phase: str, *, prefix: str = "  ") -> None:
    print(f"{prefix}running {phase} …", flush=True)


def log_nested_phase_done(phase: str, elapsed_s: float, *, indent: str = "    ") -> None:
    print(f"{indent}{phase} {elapsed_s:.1f}s", flush=True)
