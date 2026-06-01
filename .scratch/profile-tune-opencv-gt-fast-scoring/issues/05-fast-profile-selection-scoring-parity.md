Status: ready-for-agent
Category: enhancement
Labels: ready-for-agent, enhancement
Depends-on: .scratch/profile-tune-opencv-gt-fast-scoring/issues/02-profile-selection-gt-cache-train-layout.md, .scratch/profile-tune-opencv-gt-fast-scoring/issues/04-tiled-detector-proposals-schema-v2.md
Blocks: .scratch/profile-tune-opencv-gt-fast-scoring/issues/06-slurm-profile-tune-resources-docs.md

# Fast profile selection scoring + legacy parity gate

## What to build

Implement ADR 0007 **profile selection scoring**: load v2 **tiled detector proposals**, adapt records for SAHI `merge_sliced_object_predictions` (slice-merge knobs from the grid candidate), then **score merge** by painting crop masks into one in-memory **merged instance view** (ascending **score**) via a shared **common** helper—no transient **instance prediction set** per grid point. Compute train **AJI** against the ADR 0006 GT cache.

Refactor the candidate task so **profile selection ground truth cache** is loaded **once per candidate** (before the variant loop) and passed into per-variant scoring—do not reload the same train **merged instance view** (~2 GB decompressed) four times per task. Log **phase timings** at fine granularity (load GT, load proposals, slice-merge, score merge, AJI; per-variant totals) so cluster logs show where walltime goes.

Add an automated parity gate: v2 scoring **AJI** must match legacy v1 scoring **AJI** on fixed fixtures with identical GT (tight float tolerance); remove the legacy v1 scoring path after the gate passes. Update **profile selection result row** fingerprints to include proposal `schema_version` 2. Optional **instance prediction set** materialization remains only for winner/audit, not every grid point.

## Acceptance criteria

- [ ] **Profile selection scoring** uses v2 caches and direct paint **score merge** (no per-grid prediction-set RLE round-trip)
- [ ] SAHI slice-merge still used via adapter (not reimplemented)
- [ ] Parity test: v2 AJI == legacy v1 AJI on fixture(s) with fixed GT
- [ ] Legacy v1 scoring path removed after parity gate
- [ ] **Profile selection result row** fingerprints include proposal schema v2
- [ ] Fixture test: v2 cache + GT cache → `compute_train_aji` returns finite AJI in seconds, not hours
- [ ] One GT load per candidate task; logs show a single GT load line then four variant scores
- [ ] Scoring logs report per-phase timings (merge, score merge, AJI) with ≥0.1s resolution

## Blocked by

- [.scratch/profile-tune-opencv-gt-fast-scoring/issues/02-profile-selection-gt-cache-train-layout.md](02-profile-selection-gt-cache-train-layout.md)
- [.scratch/profile-tune-opencv-gt-fast-scoring/issues/04-tiled-detector-proposals-schema-v2.md](04-tiled-detector-proposals-schema-v2.md)
