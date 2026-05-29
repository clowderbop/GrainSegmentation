Status: ready-for-agent
Category: enhancement
Labels: ready-for-agent, enhancement

# PRD: Instance prediction set (canonical instance output)

Implements [ADR-0001](../../docs/adr/0001-instance-prediction-set.md). Domain vocabulary: [CONTEXT.md](../../CONTEXT.md).

## Problem Statement

Whole-section YOLO evaluation on 10k×10k test mosaics runs out of memory (400–500 GB requested) because inference materializes a dense `(N, H, W)` float32 mask stack for every detector proposal before writing artifacts. Separately, YOLO and U-Net persist incompatible instance outputs (instance label-map TIFF vs mask NPZ), while only some evaluation steps share a code path. Patch YOLO evaluation also fails when a training dataset YAML is required even though a staged patch manifest is provided. Researchers cannot reliably run or compare whole-image and patch instance evaluation on the cluster without excessive RAM or redundant formats.

## Solution

Introduce the **instance prediction set** as the single canonical per-sample instance output for both model families: JSON (`schema_version: 1`) with COCO RLE per grain entry, stored under a **prediction set directory** (`prediction_sets/{sample_id}.json`). YOLO writes **detector proposals** (overlapping, with **confidence**); U-Net writes **extracted grains** (non-overlapping, no confidence) after instance extraction from **semantic prediction** (unchanged, U-Net-only). Eval manifests reference **`instance_prediction_set`** instead of legacy instance TIFF paths. **Instance metrics** build a transient **merged instance view** via **confidence merge** for YOLO; **mask AP** reads proposals directly (whole-section YOLO test only). **Prediction overlays** paint proposals on the RGB image in confidence order without storing a label map. **Run provenance** (thresholds, slice settings, watershed params) lives in one sidecar per run output directory. Legacy label-map TIFF and dense mask NPZ are not written or read (clean break).

## User Stories

1. As a researcher running whole-section YOLO SAHI eval, I want inference to stream detector proposals into RLE JSON without building a full mask stack, so that jobs fit in reasonable cluster memory.
2. As a researcher running whole-section YOLO SAHI eval, I want mask AP and instance metrics to read the same prediction artifact, so that I do not duplicate storage or drift between metrics.
3. As a researcher running patch YOLO eval, I want predict to use only the staged manifest, so that missing train YOLO YAMLs do not block test evaluation.
4. As a researcher running patch YOLO eval, I want patch outputs in the same instance prediction set format as whole-section outputs, so that tooling is uniform across sample units.
5. As a researcher running U-Net whole-section eval, I want extracted grains written as an instance prediction set, so that instance metrics match the YOLO evaluation contract.
6. As a researcher running U-Net eval, I want semantic prediction TIFFs kept for pixel-wise semantic metrics, so that semantic and instance evaluation remain distinct.
7. As a researcher comparing U-Net and YOLO, I want both families to produce instance prediction sets with a declared **producer**, so that consumers know whether confidence merge applies.
8. As a researcher reviewing eval manifests, I want a manifest field `instance_prediction_set` pointing at each sample’s JSON file, so that staging and eval discovery are explicit.
9. As a researcher computing AJI and F1, I want YOLO overlapping proposals merged with the same confidence ordering as today’s label-map pipeline, so that instance metric numbers remain comparable to prior runs after re-inference.
10. As a researcher computing mask AP, I want COCO detections built from stored RLE proposals and confidence without loading dense masks, so that mask AP remains correct with low RAM.
11. As a researcher computing mask AP, I want mask AP limited to whole-section YOLO test samples in default SLURM flows, so that GT (GPKG) and pipeline scope stay aligned.
12. As a researcher, I do not want mask AP on U-Net extracted grains, so that we do not imply detector-style confidence where none exists.
13. As a researcher viewing prediction overlays, I want overlapping YOLO proposals drawn in confidence order on the microscopy image, so that overlays match the merged instance view visually without persisting a label map.
14. As a researcher viewing U-Net overlays, I want non-overlapping extracted grains drawn on the image, so that overlays reflect watershed/CC output.
15. As a researcher auditing a run, I want run provenance recorded once per output directory, so that I can see confidence thresholds, SAHI slice overlap, or watershed settings without opening every sample file.
16. As an AFK agent implementing eval, I want a small, testable module for loading and validating instance prediction sets, so that schema and producer rules are enforced in one place.
17. As an AFK agent implementing YOLO predict, I want to encode one full-resolution mask to RLE and append to the set before processing the next proposal, so that peak RAM scales with one mask plane plus the SAHI working list, not N planes.
18. As an AFK agent implementing U-Net extract, I want each connected region encoded to RLE with `category_id` 0 and no confidence field, so that the file matches the U-Net producer contract.
19. As a researcher re-running failed SLURM jobs from an older eval layout, I accept that legacy TIFF/NPZ outputs are not read, so that the codebase maintains one path (re-run predict/eval).
20. As a researcher using `stage_manifest write-eval`, I want eval manifests populated with `instance_prediction_set` paths under the prediction set directory, so that `evaluate_instances` and `evaluate_mask_ap` share one pointer per sample.
21. As a researcher validating manifests, I want manifest validation to accept the new field and reject obsolete `pred_instances` in new manifests, so that schema drift is caught early.
22. As a researcher running patch instance eval only, I want patch jobs to skip mask AP, so that cluster time is not spent on metrics we have not defined for patch vector GT.
23. As a researcher, I want every detection entry to use grain class identifier 0, so that single-class segmentation stays explicit until multi-class work exists.
24. As a maintainer, I want unit tests on RLE round-trip and confidence-merge parity, so that refactors do not silently change metrics.
25. As a maintainer, I want documentation for YOLO and U-Net SLURM pipelines updated to describe prediction sets and removed artifacts, so that cluster users follow the new layout.

## Implementation Decisions

### Deep modules (build or extend)

| Module | Responsibility | Interface (conceptual) |
|--------|----------------|-------------------------|
| **Instance prediction set I/O** | Schema v1 load/save/validate; path helpers for prediction set directory; RLE encode/decode for one mask; list↔COCO detection dicts | `load_prediction_set(path) → PredictionSet`; `save_prediction_set(path, data)`; `validate_prediction_set(data)` enforces producer/score/category rules |
| **Merged instance view** | Confidence merge for YOLO; pass-through ordering for U-Net; output 2D int32 label map | `prediction_set_to_merged_instance_view(set) → ndarray` |
| **Run provenance I/O** | Read/write sidecar JSON next to prediction set directory | `write_run_provenance(dir, dict)` / `load_run_provenance(dir)` |
| **YOLO predict (whole + patch)** | After SAHI/NMM or Ultralytics patch infer: stream each proposal → RLE → append; write provenance; no TIFF/NPZ | Uses prediction set I/O; patch path must not require train YAML when manifest provided |
| **U-Net extract instances** | After watershed/CC: emit extracted grains as prediction set per sample; move extract metadata into run provenance sidecar | Replaces persisting instance label-map TIFF as canonical output |
| **Manifest layer** | Sample row field `instance_prediction_set`; `write-eval` resolves paths into prediction set directory; remove/rename CLI flags from `pred_instances` | Breaking manifest contract (clean break) |
| **Instance evaluation** | Load prediction set → merged instance view → existing AJI/F1 metrics | `evaluate_instances` no longer reads instance label-map TIFF for predictions |
| **Mask AP evaluation** | Load same path; require `producer == yolo`; build COCO dt from RLE list; whole-section + GPKG only in default SLURM | Replaces NPZ loader |
| **Prediction overlay export** | Load prediction set; paint on RGB in confidence order (YOLO) or arbitrary order (U-Net disjoint) | No full label map allocation required |

Shallow orchestration (modify, do not over-abstract): SLURM bash scripts for YOLO SAHI/patch eval and U-Net whole/patch eval; pipeline markdown; `docs/manifests.md`.

### Schema (instance prediction set v1)

Top-level JSON:

- `schema_version`: `1` (required)
- `height`, `width`: image size (required)
- `producer`: `"yolo"` | `"unet"` (required)
- `detections`: array (required)

Per detection:

- `segmentation`: COCO RLE dict `{ "size": [H, W], "counts": "<string>" }` (required)
- `category_id`: `0` (required, grain class)
- `score`: float (required if `producer` is `yolo`; must be absent if `producer` is `unet`)

Not stored in per-sample files: sample unit (`patch` / `whole` remains on manifests only).

### Producer semantics

- **yolo**: entries are **detector proposals**; overlaps allowed; **confidence merge** at instance-metric and overlay consumers.
- **unet**: entries are **extracted grains**; disjoint; no confidence field.

### Run provenance sidecar

One file per run output root (e.g. alongside prediction set directory): records YOLO conf threshold, slice size, overlap ratios, and/or U-Net instance extraction method and watershed parameters. Replaces per-run `.extract_meta.json` under legacy instances folder.

### Clean break

- Remove write paths for canonical instance label-map TIFF and mask NPZ from YOLO predict.
- Remove read paths for those formats in instance eval, mask AP, and SAHI visualization.
- Internal use of transient **instance label map** in memory remains allowed for metrics only.
- Do not migrate historical scratch eval trees.

### Architectural constraints

- Keep `common` free of imports from `yolo` or `unet` packages (existing project rule).
- Prediction set I/O and merged instance view live in `common`.
- Family-specific predict/extract call into shared I/O.

### Related fix (patch YOLO predict)

When `unit` is patch and a manifest is supplied, predict must not require resolving train dataset YAML solely because `variant` is set. Aligns with manifest-driven patch eval already used in SLURM.

## Testing Decisions

**What makes a good test here:** Assert on external behavior—saved JSON shape, round-trip RLE geometry, merged label map equality against a golden raster, COCO detection list equivalence—not private SAHI loop internals.

**Modules to test (recommended):**

| Module | Tests |
|--------|--------|
| **Instance prediction set I/O** | Save/load round-trip; reject `unet` entries with `score`; reject `yolo` entries without `score`; invalid `schema_version`; path helper returns expected relative layout |
| **Merged instance view** | YOLO: two overlapping proposals, higher confidence wins pixels (match prior `instance_map_from_masks` behavior); U-Net: disjoint regions map 1:1 to labels |
| **Mask AP input adapter** | Prediction set with one RLE produces equivalent COCO dt to legacy NPZ adapter for the same binary mask and score |
| **Manifest write-eval** | Staged manifest gains `instance_prediction_set` paths pointing at `prediction_sets/{sample_id}.json` |
| **YOLO patch predict entry** | With manifest only, predict does not raise missing dataset YAML (smoke or mocked predict) |

**Prior art:** `common/tests/test_instance_predictions.py` (label map round-trip, `yolo_mask_npz_to_coco_dt`); `common/tests/test_evaluate_instances.py`; `common/tests/test_manifest_io.py`.

**Not required in v1:** Full SAHI integration test on 10k images; SLURM job execution; GPU inference.

## Out of Scope

- Reading legacy instance label-map TIFF or mask NPZ artifacts
- Automatic migration of existing eval directories on scratch
- gzip-compressed prediction set files
- Multi-class `category_id` beyond grain class `0`
- Mask AP on patch samples or U-Net extracted grains in default pipelines
- Replacing semantic prediction TIFF for U-Net
- Changing SAHI, Ultralytics, or watershed algorithms themselves
- Reducing SAHI pre-merge peak RAM (only eliminating final dense stack is in scope)
- CI execution of SLURM/GPU jobs

## Phased issues

| # | Issue | Blocked by |
|---|--------|------------|
| 01 | [YOLO patch: prediction set + instance eval](issues/01-yolo-patch-prediction-set-and-instance-eval.md) | — |
| 02 | [YOLO whole: SAHI + mask AP + overlay](issues/02-yolo-whole-sahi-mask-ap-overlay.md) | 01 |
| 03 | [U-Net: extract → prediction set](issues/03-unet-extract-prediction-set.md) | 01 |

Issues 02 and 03 can run in parallel after 01. Legacy artifact removal is folded into each slice (no separate cleanup issue).

## Further Notes

- Re-run whole-section and patch YOLO/U-Net eval after implementation; prior failed job outputs are not compatible.
- Update `docs/manifests.md` and YOLO/U-Net pipeline docs when manifest field renames land.
- Memory table for communication: 10k×10k merged instance view ≈ 400 MB int32 in eval is acceptable; target predict peak ≪ legacy 400 GB dense stack.
- Implementation can land as one tracer-bullet PR or thin vertical slices: (1) common I/O + tests, (2) YOLO predict + mask AP + SAHI eval SLURM, (3) U-Net extract + instance eval, (4) manifest/docs cleanup.

## Comments
