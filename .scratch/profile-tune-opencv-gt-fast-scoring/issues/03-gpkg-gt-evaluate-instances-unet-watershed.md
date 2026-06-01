Status: ready-for-agent
Category: enhancement
Labels: ready-for-agent, enhancement
Depends-on: .scratch/profile-tune-opencv-gt-fast-scoring/issues/01-opencv-gpkg-painter-golden-fixture.md

# GPKG ground truth via OpenCV in evaluate_instances and U-Net watershed

## What to build

Wire all production GPKG → **merged instance view** callers to the OpenCV painter from slice 01: **evaluate_instances** (whole/patch when `gt_gpkg` is used) and U-Net **watershed** tuning on GPKG. Remove the production code path that rasterizes vector annotations through `gt_annotations_to_instance_map` / per-annotation **pycocotools** decode for GPKG ground truth. Keep **pycocotools** only where predictions (RLE masks), not GPKG labels, require decode.

## Acceptance criteria

- [ ] **evaluate_instances** uses OpenCV painter for GPKG **merged instance view**
- [ ] U-Net watershed GPKG tuning uses the same canonical painter
- [ ] No production GPKG GT path calls `gt_annotations_to_instance_map` on vector annotations
- [ ] Existing / updated tests pass for eval and watershed GT loading
- [ ] Prediction-mask decode still uses **pycocotools** where appropriate

## Blocked by

- [.scratch/profile-tune-opencv-gt-fast-scoring/issues/01-opencv-gpkg-painter-golden-fixture.md](01-opencv-gpkg-painter-golden-fixture.md)
