Status: ready-for-agent
Category: enhancement
Labels: ready-for-agent, enhancement
Depends-on: 01-yolo-patch-prediction-set-and-instance-eval
Blocks: —

# U-Net: extracted grains as instance prediction set

## Parent

[PRD: Instance prediction set](../PRD.md) · [ADR-0001](../../../docs/adr/0001-instance-prediction-set.md)

## What to build

Wire U-Net instance extraction into the same **instance prediction set** contract as YOLO.

After watershed or connected-components extraction from **semantic prediction**, emit **extracted grains** as schema v1 JSON: COCO RLE per region, `category_id: 0`, `producer: unet`, **no confidence field**. Store under **prediction set directory**. Record **run provenance** once per run (instance method, watershed parameters, etc.) in a sidecar next to prediction sets—not legacy metadata under an `instances/` folder.

**Semantic prediction** TIFF output stays unchanged for semantic metrics.

**Eval manifests** for U-Net whole (and patch if applicable) use **`instance_prediction_set`**. **`evaluate_instances`** loads U-Net prediction sets (disjoint grains; no confidence merge) and computes AJI/F1.

Update U-Net whole/patch eval SLURM and pipeline docs. Remove U-Net canonical instance label-map TIFF writes and eval reads of TIFF in code touched by this slice.

## Acceptance criteria

- [ ] `extract_instances` writes `prediction_sets/{sample_id}.json` per sample, not canonical `*_instances.tif`
- [ ] U-Net prediction set entries have no `score`; validation enforces producer rules
- [ ] Run provenance sidecar written with extraction method and watershed/CC parameters
- [ ] Semantic prediction TIFF and `evaluate_semantic` behavior unchanged
- [ ] U-Net whole-section eval manifest includes `instance_prediction_set`; `evaluate_instances` succeeds on a fixture or staged sample
- [ ] U-Net pipeline doc describes prediction sets + semantic prediction; legacy instances folder layout documented as removed
- [ ] Unit test: label map from extraction fixture converts to prediction set and back to equivalent merged instance view

## Blocked by

- [01-yolo-patch-prediction-set-and-instance-eval](01-yolo-patch-prediction-set-and-instance-eval.md)

## Comments
