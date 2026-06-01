Status: done
Category: enhancement
Labels: ready-for-agent, enhancement
Depends-on: .scratch/profile-tune-opencv-gt-fast-scoring/issues/01-opencv-gpkg-painter-golden-fixture.md
Blocks: .scratch/profile-tune-opencv-gt-fast-scoring/issues/05-fast-profile-selection-scoring-parity.md

# Profile selection ground truth cache (`gt_cache/train/`)

## What to build

End-to-end **profile selection ground truth cache** per ADR 0006: one canonical train **merged instance view** per tune run under `_work/gt_cache/train/` (`instance_map.npz` + `fingerprint.json`), fingerprinted on `train_labels.gpkg` SHA-256, **sample id**, width, and height—not on input variant.

Refactor the GT-cache CLI and SLURM job to rasterize via the OpenCV painter from slice 01, copy the GPKG to `$TMPDIR` before painting, and sync **common** only (no YOLO/torch on the GT job). Log **phase timings** on the GT-cache job (copy GPKG, rasterize, write cache) so train-scale runs are diagnosable from logs. Update **profile selection** consumers to load the shared cache path and emit updated GT fingerprints on **profile selection result row** records (variant removed from GT fingerprint; cache schema version bumped). Verify locally on the micro fixture that the cache writes and candidate code can load GT successfully.

## Acceptance criteria

- [x] Cache layout is `_work/gt_cache/train/` (not per-variant trees)
- [x] Fingerprint excludes input variant; includes gpkg hash, sample id, width, height
- [x] GT-cache SLURM job uses OpenCV painter and `$TMPDIR` GPKG copy
- [x] Candidate / scoring loads shared GT cache with validation
- [x] **Profile selection result row** GT fingerprint shape updated; stale rows would not resume
- [x] Fixture-scale CLI test: build cache → load in scoring path
- [x] GT-cache logs include per-phase timings (copy, rasterize, write)

## Blocked by

- [.scratch/profile-tune-opencv-gt-fast-scoring/issues/01-opencv-gpkg-painter-golden-fixture.md](01-opencv-gpkg-painter-golden-fixture.md)
