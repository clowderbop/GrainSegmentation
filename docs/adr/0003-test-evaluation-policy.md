# Test evaluation policy

Status: accepted

Held-out test reporting uses one headline and a fixed inference geometry so input variants and model families are comparable. **Variant test ranking** and **model test comparison** both use whole-section sliding-window inference, scored by instance **AJI** with **F1@IoU50** alongside (see `CONTEXT.md`). YOLO reaches instances via SAHI detector proposals and **score merge**; U-Net via semantic prediction, sliding-window stitch, and per-checkpoint **U-Net extraction profile** (watershed tune JSON — not shared across models).

**Supporting test metrics** (not headlines): patch-level instance AJI/F1 with **patch metric aggregate** — unweighted mean over grain-bearing patches only (empty-GT tiles excluded and counted) plus grain-weighted mean; Ultralytics segmentation mAP on every default patch test job; YOLO whole-section **Mask AP** on SAHI runs (COCO on proposals, not **merged instance view**). **Mask AP** requires GT and predictions to share **grain class** `0`; pre-fix runs that reported AP=0 are invalid and all SAHI test jobs should be re-run after the class-id fix.

**Test inference recipe** lives in `configs/test_inference.yaml` at the repo root. It governs window size, stride/overlap, patch crop size, batching, and YOLO **score** threshold and val settings for all variants and both producers. SLURM and Python eval entrypoints read it instead of duplicating constants. Per-variant tuning of `conf` or window geometry is out of scope for official ranking tables.

**Considered options:** Patch mean AJI or Ultralytics mAP as headline (rejected — training-crop or detector-native, not deployment unit); per-variant SAHI/conf tuning (rejected — confounds modality with inference settings); optional Ultralytics val behind a flag (rejected — supporting bundle always includes val on patch jobs); single watershed profile in the shared recipe (rejected — U-Net checkpoints retain per-model tune JSON).

**Consequences:** Implement grain-class alignment in COCO GT builders; add patch aggregates to `instance_metrics.json` reporting; wire four YOLO SLURM scripts and U-Net whole/patch test scripts to the shared YAML; remove `RUN_ULTRALYTICS_VAL` gating; re-submit SAHI `test_yolo_*` jobs after Mask AP fix.
