Status: done
Category: enhancement
Labels: ready-for-agent, enhancement
Blocks: .scratch/profile-tune-opencv-gt-fast-scoring/issues/02-profile-selection-gt-cache-train-layout.md, .scratch/profile-tune-opencv-gt-fast-scoring/issues/03-gpkg-gt-evaluate-instances-unet-watershed.md

# OpenCV GPKG → merged instance view (common) + golden fixture

## What to build

Add the canonical **common** path that paints a GeoPackage into a **merged instance view** (per-pixel instance ids): exterior rings only via OpenCV `fillPoly` after `np.rint` vertex discretization, ascending instance id paint order (later grains overwrite overlaps), clip to image frame. Add `opencv-python-headless` to **common** dependencies.

Lock behavior with a committed micro-GPKG under test fixtures, a golden compressed **instance label map**, and a small regen script for intentional fixture updates. Tests assert the golden map matches painter output. Do not wire production callers in this slice.

## Acceptance criteria

- [x] OpenCV GPKG → **merged instance view** helper lives in **common** and matches ADR 0006 topology rules
- [x] `opencv-python-headless` is a **common** dependency (not an optional extra)
- [x] Committed micro-GPKG fixture, golden `instance_map.npz`, and regen script exist
- [x] Tests fail on painter drift from the golden map
- [x] No production caller switched yet (evaluate_instances / tune GT cache follow in later slices)

## Blocked by

None — can start immediately.
