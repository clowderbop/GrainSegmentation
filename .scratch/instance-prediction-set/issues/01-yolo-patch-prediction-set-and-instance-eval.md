Status: ready-for-agent
Category: enhancement
Labels: ready-for-agent, enhancement
Depends-on: —
Blocks: 02-yolo-whole-sahi-mask-ap-overlay, 03-unet-extract-prediction-set

# YOLO patch: instance prediction set and instance metrics

## Parent

[PRD: Instance prediction set](../PRD.md) · [ADR-0001](../../../docs/adr/0001-instance-prediction-set.md)

## What to build

Deliver the shared **instance prediction set** contract and prove it on the **patch** YOLO path end-to-end.

Introduce schema v1 JSON under a **prediction set directory** (`prediction_sets/{sample_id}.json`): `producer`, `height`, `width`, `detections[]` with COCO RLE, `category_id: 0`, and **confidence** on each entry for YOLO **detector proposals**. Implement load/save/validate and **merged instance view** (confidence merge for YOLO) in the shared package with unit tests.

YOLO **patch** predict must stream proposals to RLE (no dense `(N,H,W)` stack, no canonical instance label-map TIFF, no mask NPZ). When a staged **manifest** is provided, predict must not fail on missing train dataset YAML.

**Eval manifests** must use field **`instance_prediction_set`** (not `pred_instances`). `write-eval` populates paths into the prediction set directory. **`evaluate_instances`** loads prediction sets, builds merged instance view, and runs AJI/F1 unchanged in meaning.

Update patch test SLURM and **docs/manifests** for the new field and layout. Remove patch-path writes/reads of legacy TIFF/NPZ and `pred_instances` in code touched by this slice.

```json
{
  "schema_version": 1,
  "height": 1024,
  "width": 1024,
  "producer": "yolo",
  "detections": [
    {
      "segmentation": { "size": [1024, 1024], "counts": "..." },
      "score": 0.87,
      "category_id": 0
    }
  ]
}
```

## Acceptance criteria

- [ ] Shared module loads/saves schema v1; rejects `unet` detections with `score` and `yolo` detections without `score`
- [ ] Merged instance view for two overlapping YOLO proposals matches prior confidence-painting behavior (unit test)
- [ ] Patch YOLO predict writes only `prediction_sets/{sample_id}.json` (no `instances/*_instances.tif`, no `masks/*.npz`)
- [ ] Patch predict succeeds with `--manifest` and `--variant` when train YOLO YAML is absent on scratch
- [ ] `stage_manifest write-eval` sets `instance_prediction_set` per sample
- [ ] `evaluate_instances` on a staged patch eval manifest produces metrics JSON without reading label-map TIFF
- [ ] `run_patch_test_eval.sh` (or equivalent) runs predict → write-eval → evaluate_instances using prediction sets
- [ ] Manifest documentation describes `instance_prediction_set`; `pred_instances` removed from docs as current contract
- [ ] Tests cover prediction set I/O, merge parity, and write-eval path shape (no GPU required)

## Blocked by

None — can start immediately.

## Comments
