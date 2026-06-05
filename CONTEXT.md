# Grain segmentation

Research codebase comparing U-Net (semantic segmentation + instance extraction) and YOLO (direct instance segmentation) on sandstone thin-section microscopy, across multi-modal input variants.

## Language

### Instance outputs & geometry

**Instance prediction set**:
The canonical per-sample model output: a list of non-overlapping grain instances, each with encoded mask geometry. YOLO entries also carry **score** (winning proposal after **cross-tile association** on whole sections, or **score merge** on patch predict); U-Net entries do not. On-disk layout: ADR 0001.
_Avoid:_ handover file, predictions JSON, NPZ (format names, not the concept)

**Instance label map**:
A single raster image where each pixel carries one instance id (or background); used internally when metrics need a painted map. Not persisted as the canonical prediction artifact.
_Avoid:_ instance map, instances TIFF (implementation paths)

**Grain instance**:
One segmented grain region in an image, represented as a mask or polygon regardless of which model produced it.

**Grain class**:
The single object category for instance segmentation in this project. All entries in an **instance prediction set** use this class until multi-class segmentation is introduced.
_Avoid:_ category, label id (ambiguous with instance ids)

**Merged instance view**:
A single raster where each pixel has one instance id (or background). For YOLO, rasterizing the canonical **instance prediction set** after **cross-tile association** (whole) or **score merge** (patch); for U-Net, rasterizing **extracted grains**. Vector ground truth is painted into this form for metrics and for **profile selection ground truth cache**. Built transiently when a label map is required.
_Avoid:_ pred map, prediction raster (too vague)

**Prediction overlay**:
A visualization that blends predicted grain regions onto the microscopy image from the canonical **instance prediction set** (non-overlapping grains for both **producer** families).
_Avoid:_ SAHI visualization (pipeline step name)

**Prediction set directory**:
The per-run folder holding one **instance prediction set** per **sample id** (ADR 0001).
_Avoid:_ instances/ (legacy label-map folder name)

### YOLO detection & tiling

**Detector proposal**:
A detected grain mask and **score** from the YOLO instance segmentation model before **score merge**. Proposals may overlap; the default predict pipeline does not persist them as the canonical **instance prediction set**.
_Avoid:_ detection, prediction row (too generic), canonical YOLO output (use **instance prediction set**)

**Tiled detector proposals**:
The full set of **detector proposals** from every **sliding window** slice, in whole-image coordinates, with source tile bounds metadata, before **cross-tile association** into non-overlapping grains. Overlapping; not the canonical **instance prediction set**. **Profile selection** persists and reuses them per detector key (ADR 0005–0007).
_Avoid:_ pre-merge cache, SAHI pickle (implementation paths)

**Cross-tile association**:
YOLO whole-section post-processing that fuses **tiled detector proposals** into non-overlapping grains using mask overlap, tile-centrality, and border-partial rules (not SAHI slice-merge or score-paint). Shared by **profile selection scoring** and held-out whole predict. Produces the canonical **instance prediction set** on whole sections.
_Avoid:_ slice-merge, score merge (whole-section path), NMS (legacy)

**Score**:
A detector-assigned value for how likely a YOLO **detector proposal** is a true grain instance. Required on YOLO grains in the canonical **instance prediction set**; must be absent on U-Net **extracted grains**.
_Avoid:_ confidence, probability (unless explicitly calibrated)

**Score merge**:
YOLO patch post-processing that resolves overlapping **detector proposals** into non-overlapping grains by painting higher-**score** masks over lower-score ones. Runs at patch predict time only; each surviving grain keeps the **score** of its winning proposal.
_Avoid:_ NMS, confidence merge (legacy name), using score merge for whole-section output (use **cross-tile association**)

**Slice-boundary duplicate**:
Two or more YOLO grains in the canonical **instance prediction set** that correspond to one ground-truth grain, often when adjacent **sliding window** tiles each detect part of the same grain and neither **cross-tile association** nor legacy score-paint could fuse them. Target failure mode for **cross-tile association** improvements.
_Avoid:_ split grain, tile artifact (informal only)

**Mask threshold**:
Detector mask binarization cutoff at inference (whole **sliding window** and patch predict). Not re-applied when building the **instance prediction set** from an already-binarized mask.
_Avoid:_ second binarization at encode time, using **mask threshold** for minimum **score**

### U-Net outputs & extraction

**Semantic prediction**:
A U-Net per-pixel class label raster for one microscopy sample. Used for pixel-wise semantic metrics, not for cross-model instance comparison with YOLO.
_Avoid:_ semantic map, class TIFF (implementation)

**Extracted grain**:
One entry in a U-Net instance prediction set: a grain region produced by instance extraction (connected components or watershed) on a semantic prediction. Extracted grains do not overlap each other.
_Avoid:_ watershed instance, CC label (method names)

**U-Net extraction profile**:
Per-model watershed (or other) settings used after semantic prediction to produce **extracted grains** at test time. Selected on train by **whole-section PQ** and recorded in tune JSON per checkpoint, not from the **test inference recipe**.
_Avoid:_ global watershed in test recipe, extraction recipe (ambiguous with instance prediction set)

### Models & input configurations

**Producer**:
Which model family wrote an **instance prediction set**: YOLO (non-overlapping grains with **score**) or U-Net (non-overlapping **extracted grains** without **score**). Thesis figures use **model display label**, not this token.
_Avoid:_ model_type, source (too vague)

**Model display label**:
Thesis-facing name for a **producer** on post-eval figures and legends: **YOLO** and **U-Net**. Axis titles read **Model**; ticks and legends use these labels, not machine tokens.
_Avoid:_ Producer on figure axes, using Model for **input configuration** (that is **variant display name**)

**Input configuration**:
A multi-modal microscopy input setup for training and evaluation (which channels or composites are fed to the model). Identified by a registry variant key in `config/variants.yaml`; thesis prose and figures use **variant display name**, not the key.
_Avoid:_ modality (ambiguous), All-Stack / All-Comp (legacy thesis names), repeating per-variant definitions here when the registry is authoritative

**Variant display name**:
Thesis-facing label for an **input configuration** on tables and figures. One `display_name` per registry variant; join key remains the variant key (e.g. *FullStack* for `PPL+AllPPX`).
_Avoid:_ registry key on plot axes, PPL+AllPPX / PPL+PPXblend on figures, separate reporting config for labels only

**Input image count**:
The number of microscopy images in an **input configuration**, used as the thesis-facing complexity axis for cost/benefit reporting. Values come from the variant registry input suffixes: PPL and FullComp use one image, PPL+XPLComp uses two images, and FullStack uses seven images.
_Avoid:_ YOLO tensor channel count, RGB channel count, using "input count" without saying what is counted

### Manifests & run layout

**Sample id**:
The stable key for one manifest row and its **instance prediction set**. For whole-section samples it is the data split (`train`, `test`), not the mosaic or channel stem. For patch samples it is the patch stem (e.g. `region_0001_y00000_x00000`). ADR 0001.
_Avoid:_ TIFF filename, channel suffix, split directory name (paths, not ids)

**Sample unit**:
Whether a manifest row refers to a whole microscopy section or a single patch crop. Carried on dataset manifests (`patch` or `whole`), not duplicated inside instance prediction set files.
_Avoid:_ scale, tile (ambiguous)

**Eval manifest**:
A dataset manifest augmented after inference with paths to predictions and ground truth, used to drive evaluation. All inputs must resolve under the manifest’s work directory so jobs do not depend on ephemeral or external paths (ADR 0002).
_Avoid:_ eval_manifest.json as a domain term (filename only)

**Staging**:
Mirroring data between persistent storage and a node-local work directory around a cluster job: pull what the job needs at start, compute against the local tree, push durable outputs at end. Staged inputs are not read from persistent storage during the job. Outputs that other tasks must see before the job or that should persist despite a job failure or timeout may be written to persistent storage directly instead of write-back (e.g. resume or parallel work).
_Avoid:_ partial staging (only some inputs listed for the job), pulling more than the job reads, staging the entire persistent tree when a subset suffices

**Run provenance**:
Inference and post-processing parameters for a model run (e.g. **test inference recipe**, **YOLO inference profile**, **U-Net extraction profile**, **score merge**), stored once per run output directory alongside **prediction set directory** (ADR 0001).
_Avoid:_ metadata, .extract_meta (legacy filename)

### Test evaluation policy

**Individual grain instance recovery**:
The primary scientific evaluation target: recovering each annotated grain as a separate predicted instance on the whole section, balancing object detection and mask quality rather than only grain-area coverage.
_Avoid:_ treating area coverage alone as sufficient, optimizing profiles that improve overlap while producing many duplicate or split grain instances

**Whole-section PQ**:
The headline held-out test metric for **individual grain instance recovery**: Panoptic Quality on the whole-section **merged instance view**, combining detection quality and segmentation quality for one-to-one matched grains using the standard IoU > 0.5 match convention.
_Avoid:_ area-only overlap headlines, patch-level PQ as the primary rank

**PQ diagnostics**:
Required companion metrics for PQ on held-out **eval** and **post-eval reporting**: detection quality (**DQ**), segmentation quality (**SQ**), IoU50 true-positive / false-positive / false-negative match counts (**TP**/**FP**/**FN**), precision/recall/F1 at IoU50 and IoU75, mean precision/recall/F1 over IoU50:95 (`mP_iou50_95`, `mR_iou50_95`, `mF1_iou50_95`), predicted instance count, ground-truth instance count, predicted/ground-truth instance ratio, and **AJI+**. Thresholded matching diagnostics use the same strict IoU > threshold convention as PQ. **PQ**, **DQ**, **SQ**, IoU50 **TP**/**FP**/**FN**, and IoU50 precision/recall/F1 use the same greedy match and strict IoU > 0.5 definition as tune-path scoring; tune paths do not recompute IoU75, mP/mR/mF1 0.5:0.95, or **AJI+**.
_Avoid:_ reporting PQ without explaining whether failures come from missed/duplicate grains, poor mask boundaries, or object-count inflation; treating tune-path artifacts as if they contained the full eval diagnostic set

**MergedViewPqResult**:
The tune-path diagnostic record from `compute_merged_view_pq` on a **merged instance view**: **PQ**, **DQ**, **SQ**, IoU50 precision/recall/F1, TP/FP/FN, instance counts, matched-IoU spread (`min`/`max`/`median`), and overlap forensics (co-occurring pairs, near-miss counts, unmatched-prediction IoU summary). Persisted by **profile selection scoring**, U-Net watershed tune, and CC-vs-watershed train selection. Selection objectives use **`pq` / `mean_pq` only**; other fields are audit diagnostics. Not the **instance metric bundle** (no IoU75, mP/mR/mF1 0.5:0.95, **AJI+**).
_Avoid:_ calling this the full **PQ diagnostics** set, recomputing the eval bundle inside tune hot paths

**Instance metric bundle**:
The standard metric set from `compute_instance_metric_bundle` on held-out **eval** whenever producer artifacts support it: **PQ** on the evaluated sample unit plus full **PQ diagnostics** (`pq`, `dq`, `sq`, `tp`, `fp`, `fn`, IoU50/IoU75 and mP/mR/mF1 0.5:0.95 precision/recall/F1, instance counts, predicted/ground-truth ratio, **AJI+**). **Whole-section PQ** is the headline for ranking and reporting; patch evaluations compute patch-level bundle fields as supporting diagnostics. Train-side tune paths (**profile selection scoring**, watershed tune) score via **`MergedViewPqResult`** instead — same **PQ** and IoU50 match-count definition, narrower persisted fields (no IoU75, mP/mR/mF1 0.5:0.95, **AJI+**). AP/mAP metrics are outside this bundle.
_Avoid:_ maintaining separate PQ definitions for tune vs eval; implying YOLO profile selection or watershed tune wrote the full bundle

**AJI+**:
Supporting microscopy-style instance overlap metric using unique instance pairing. Useful as an overlap diagnostic but not the headline for **individual grain instance recovery**.
_Avoid:_ treating AJI+ as sufficient evidence that individual grains were recovered one-for-one

**Variant test ranking**:
The primary ordering of **input configuration**s on held-out test: whole-section sliding-window inference, ranked by **whole-section PQ** with **PQ diagnostics** reported alongside. **Supporting test metrics** are not the headline rank.
_Avoid:_ AJI as the headline, test mAP, patch mean metrics as headline (training-crop or detector-native, not deployment unit)

**Model test comparison**:
Comparing **producer** families (YOLO vs U-Net) on the same input variant and test mosaic: same headline as **variant test ranking** — **whole-section PQ** and **PQ diagnostics** under the shared **test inference recipe**; both use non-overlapping grain lists (U-Net via extraction after semantic prediction, YOLO via **cross-tile association** on whole sections).
_Avoid:_ comparing models on AP/mAP, YOLO-only metrics as cross-model evidence

**Supporting test metrics**:
Diagnostics bundled with eval jobs but not used for **variant test ranking**: patch-level **instance metric bundle** for diagnosing whether **sliding window** inference changes performance, and patch AP/mAP from Ultralytics val where available.
_Avoid:_ whole-section Mask AP, AP/mAP outside Ultralytics patch val, patch metrics as headline rank, using patch metrics to guide locked grid designs or profile selection

**Patch metric aggregate**:
Summary of patch-level instance metrics across the test split: (1) **unweighted mean** over patches that have at least one grain in ground truth, with empty-GT tiles excluded and counted separately; (2) **grain-weighted mean** over the same grain-bearing patches, weighting each patch by its GT instance count.
_Avoid:_ mean over all patches (includes empty tiles), simple average (ambiguous which rule)

**Patch AP/mAP**:
YOLO-only detector diagnostics computed by Ultralytics val on patch data. AP/mAP metrics are optional YOLO patch diagnostic reporting only: they are not part of the **instance metric bundle**, not shown in main result figures, not computed on whole-section **instance prediction set**s, and not used for **variant test ranking** or **model test comparison**.
_Avoid:_ whole-section Mask AP, AP/mAP as cross-model evidence, AP/mAP beside headline metrics in main figures

**Test inference recipe**:
The single shared held-out test configuration for every **input configuration** and both **producer** families: whole-section **sliding window** geometry, patch crop size and batching, the frozen **YOLO inference profile**, and supporting YOLO val settings. Enables fair **variant test ranking** and **model test comparison**. U-Net instance extraction stays per **U-Net extraction profile**. Recorded in **run provenance** (ADR 0003).
_Avoid:_ per-variant inference settings on test, duplicating recipe constants in job scripts

### YOLO profile selection

**YOLO inference profile**:
The train-selected detector minimum **score** (`conf`) plus the fixed **mask threshold** from the **test inference recipe** (not a grid axis). **Profile selection** searches only `conf` on the grid in `config/yolo_inference_profile_tune.yaml` (~7 candidates); SAHI slice-merge postprocess axes are not tuned on the whole-section path. One profile shared across all input variants on held-out test. Chosen via **profile selection**, then **profile promotion** into the recipe. Recorded in **run provenance**. Re-run **profile selection** when train labels or YOLO weights change materially.
_Avoid:_ per-variant profiles (confounds **variant test ranking**), tuning on overlapping **detector proposals** as the primary objective, multi-axis SAHI merge grids on whole-section output

**Profile selection**:
Train-side search for the shared **YOLO inference profile** on the whole **train** section, maximizing mean **whole-section PQ** averaged across input variants. Winner feeds **profile promotion**; audit trail on scratch records per-variant **`MergedViewPqResult`** diagnostics for each candidate (ADR 0005).
_Avoid:_ tuning on overlapping **detector proposals** as the headline objective, scoring with the full **instance metric bundle**

**Profile selection scoring**:
Computing train **whole-section PQ** via `compute_merged_view_pq` (`compute_train_pq` in YOLO code) for one grid point: **tiled detector proposals** → **cross-tile association** → **merged instance view** → **`MergedViewPqResult`**, without persisting a full **instance prediction set** or the **instance metric bundle**. Held-out whole predict uses the same postprocess module; held-out **eval** computes the full bundle (ADR 0005).
_Avoid:_ SAHI slice-merge + score-paint on the production path, requiring prediction-set JSON equality on every grid point, `compute_instance_metric_bundle` on the tune hot path

**Profile selection ground truth cache**:
The canonical train ground-truth **merged instance view** for a tune run, built once from vector labels and reused by all **profile selection scoring** tasks across input variants (label geometry is shared; channels are not). ADR 0005.
_Avoid:_ per-variant GT caches when label geometry is shared, using semantic TIFFs as GT for profile selection

**Profile promotion**:
Installing the **profile selection** winner’s train-selected **`conf`** and the fixed **`mask_threshold`** into the **test inference recipe** for git commit and held-out test (`rewrite_yolo_conf_in_recipe_text`; other recipe YOLO keys unchanged). ADR 0005.
_Avoid:_ partial promotion (only `conf` without the paired fixed `mask_threshold`), promoting legacy five-knob or SAHI-merge-grid winners

### Post-eval reporting

**Post-eval reporting**:
Aggregating finished test eval artifacts into comparison tables and thesis figures after cluster test jobs complete. Not part of inference or per-job metric computation. Headline charts use **whole-section PQ**; **PQ diagnostics** and **supporting test metrics** appear in separately labelled panels.
_Avoid:_ eval pipeline, in-job plotting, quality index (legacy composite)

**Eval run discovery**:
How **post-eval reporting** finds finished eval outputs before building the **reporting bundle**. Convention-based on scratch in v1 (see README); a future **eval run catalog** may replace path conventions.
_Avoid:_ submit script name as discovery key, required TSV before plotting

**Reporting bundle**:
The regenerated output of **post-eval reporting** (figures, derived tables, run summary). Not versioned in git.
_Avoid:_ repo `reports/` binaries, figures committed to git

**Reporting tier**:
A grouping inside the **reporting bundle** that separates thesis-facing results, supporting diagnostics, and artifact QA outputs. Core thesis results carry **whole-section PQ** as the headline; diagnostic outputs explain metric behavior; artifact QA outputs catch missing, stale, or suspicious eval artifacts.
_Avoid:_ treating every generated figure as equally thesis-facing, mixing QA checks into headline result ranking
