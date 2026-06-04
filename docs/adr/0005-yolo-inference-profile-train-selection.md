# YOLO inference profile selection architecture

Status: accepted

## Context / Problem

YOLO whole-section inference uses sliding windows plus per-slice Ultralytics detection. The shared test recipe fixes window geometry, but YOLO still needs five train-selected knobs before held-out test: `postprocess_type`, `match_metric`, `match_threshold`, `conf`, and `mask_threshold`.

## Decision

We select one shared **YOLO inference profile** on the whole train section. A factorial grid from `config/yolo_inference_profile_tune.yaml` is scored by mean whole-section train PQ across all registry variants. Pre-merge proposal counts, patch metrics, and other diagnostics are audit information only. The winner is promoted into `config/test_inference.yaml` and committed before held-out test.

The profile is shared across variants. Per-variant profiles are rejected because they would confound **variant test ranking** with inference settings. Re-run profile selection when train labels or YOLO weights change materially, not after every single-variant training job.

Profile tuning uses two durable scratch caches under `OUTPUT_DIR/.cache/` (internal to tune runs):

- **Tiled detector proposals:** detector outputs cached per `(variant, conf, mask_threshold)` as compact records with score, whole-image bbox, crop-local COCO RLE, crop offset, and image shape under `.cache/{variant}/tiled_proposals/c{conf}_t{mask}/`. The one-cache-per-detector-key contract is unchanged. The cache is not the canonical prediction artifact and is not used by held-out `yolo.predict`.
- **Profile selection ground truth cache:** one train **merged instance view** under `.cache/gt_cache/train/`, rasterized from `train_labels.gpkg` with the canonical OpenCV polygon painter. It is shared across variants because label geometry is shared.

**Profile selection** rows, `results.csv`, and `winner.json` live under `OUTPUT_DIR/grid/` on scratch so array resume and parallel candidates do not depend on node-local disks.

In-flight tune runs that used the legacy `_work/` cache root are incompatible and are not resumed; start a new run id after the layout change.

Detector SLURM array tasks are bundled by registry **input configuration** (variant): each task stages only that variant’s train whole stacked TIFF to `$TMPDIR`, loads YOLO once, writes all `conf × mask_threshold` proposal caches for that variant back to scratch `.cache/`, and does not persist manifest staging trees under the tune run directory.

Candidate array tasks copy only the subtrees required for that grid point (shared GT cache plus the four variant proposal trees for its `conf` and `mask_threshold`) from `.cache/` into job-unique `$TMPDIR`, then run **profile selection scoring** against the local work root. Scoring loads tiled proposals and the GT cache, applies SAHI slice-merge, paints **score merge** into a merged instance view, and computes train PQ plus the full diagnostic bundle. It does not write an **instance prediction set** for every grid point. Held-out test still runs full prediction and persists canonical prediction sets.

Each candidate writes a **profile selection result row** with knob values, per-variant PQ, mean PQ, diagnostics, and input fingerprints. Finalization merges rows into `grid/results.csv` and writes `grid/winner.json`. Valid detector caches may be reused within a compatible run directory; stale rows or incompatible cache schema versions must not resume.

## Rejected Alternatives

Per-variant profiles; coordinate search instead of a factorial grid; optimizing overlapping detector proposals; AJI as the selection objective; Bayesian search for v1; scratch-only winners without git promotion; full-section proposal masks in caches; per-variant GT caches; using semantic preprocessing TIFFs as GT. These were rejected for fairness, reproducibility, memory, wrong target semantics, or avoidable complexity.

## Consequences

Profile selection depends on score-merged YOLO predictions as the deployed system and on PQ as the selection objective. The runbook owns SLURM orchestration, resource defaults, and recovery steps.

## Links

- Canonical prediction output: [ADR 0001](0001-instance-prediction-set.md)
- Evaluation policy: [ADR 0003](0003-test-evaluation-policy.md)
- YOLO runbook: [`docs/runbooks/yolo.md`](../runbooks/yolo.md#profile-selection)
- PQ policy: [`docs/metrics.md`](../metrics.md#pq-centered-rerun-policy)
- Metric policy: [`docs/metrics.md`](../metrics.md)
- Glossary: [`CONTEXT.md`](../../CONTEXT.md)
- Profile grid: [`config/yolo_inference_profile_tune.yaml`](../../config/yolo_inference_profile_tune.yaml)

