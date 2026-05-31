Status: ready-for-agent
Category: enhancement
Labels: ready-for-agent, enhancement
Depends-on: 01-yolo-patch-prediction-set-and-instance-eval
Blocks: —

# YOLO whole: SAHI predict, mask AP, and prediction overlay

## Parent

[PRD: Instance prediction set](../PRD.md) · [ADR-0001](../../../docs/adr/0001-instance-prediction-set.md)

## What to build

Extend the instance prediction set pipeline to **whole-section** YOLO SAHI evaluation with low peak RAM.

Whole **YOLO predict** must stream **detector proposals** to RLE JSON (same schema as patch slice 01) without materializing a full mask stack. Write **run provenance** once per run (score threshold, slice size, overlap) beside the prediction set directory.

**Mask AP** reads the same **`instance_prediction_set`** path as instance metrics; requires `producer: yolo` and detection **score**; runs on **whole-section** test samples with GPKG ground truth in default SLURM flow (not patch).

**Prediction overlay** export loads the prediction set and paints grains on the microscopy RGB image in **score order** (no persisted instance label map).

Update whole-section SAHI SLURM and YOLO pipeline docs. Remove whole-path writes/reads of legacy mask NPZ, canonical instance TIFF, and mask-AP input from NPZ in code touched by this slice.

## Acceptance criteria

- [ ] Whole-section SAHI predict completes for at least one variant on a 10k×10k-class mosaic without OOM from dense `(N,H,W)` masks (cluster or documented memory check)
- [x] `prediction_sets/{sample_id}.json` written for whole test sample; run provenance sidecar present with SAHI/YOLO parameters
- [x] `evaluate_mask_ap` builds COCO detections from prediction set RLE + scores; no NPZ loader in mask AP path
- [x] Mask AP SLURM step uses eval manifest `instance_prediction_set` paths
- [x] Prediction overlay export uses score-order painting; no requirement for `instances/*_instances.tif`
- [x] `run_sahi_test_eval.sh` runs predict → write-eval → overlay → instance metrics → mask AP on prediction sets
- [x] YOLO pipeline doc updated for prediction sets, provenance, and removed NPZ/TIFF artifacts
- [x] Tests: mask AP adapter from prediction set matches prior behavior for a fixed RLE+score fixture (no full SAHI GPU test required)

## Blocked by

- [01-yolo-patch-prediction-set-and-instance-eval](01-yolo-patch-prediction-set-and-instance-eval.md)

## Comments
