# Grain segmentation

Research codebase comparing U-Net (semantic segmentation + instance extraction) and YOLO (direct instance segmentation) on sandstone thin-section microscopy, across multi-modal input variants.

## Language

### Instance outputs & geometry

**Instance prediction set**:
The canonical per-sample model output: a list of non-overlapping grain instances, each with encoded mask geometry. YOLO entries also carry **score** (winning proposal after **score merge** at predict time); U-Net entries do not. On-disk layout: ADR 0001.
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
A single raster where each pixel has one instance id (or background). For YOLO, rasterizing the canonical **instance prediction set** after **score merge**; for U-Net, rasterizing **extracted grains**. Vector ground truth is painted into this form for metrics and for **profile selection ground truth cache**. Built transiently when a label map is required.
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
The full set of **detector proposals** from every **sliding window** slice, in whole-image coordinates, before slice-merge into non-overlapping grains. Overlapping; not the canonical **instance prediction set**. **Profile selection** may persist and reuse them when only merge knobs change (ADR 0005–0007).
_Avoid:_ pre-merge cache, SAHI pickle (implementation paths)

**Score**:
A detector-assigned value for how likely a YOLO **detector proposal** is a true grain instance. Required on YOLO grains in the canonical **instance prediction set**; must be absent on U-Net **extracted grains**.
_Avoid:_ confidence, probability (unless explicitly calibrated)

**Score merge**:
YOLO post-processing that resolves overlapping **detector proposals** into non-overlapping grains by painting higher-**score** masks over lower-score ones. Runs at predict time for both whole and patch **sample unit**; each surviving grain keeps the **score** of its winning proposal. Produces the canonical **instance prediction set**.
_Avoid:_ NMS, confidence merge (legacy name), instance map from masks (implementation)

**Slice-boundary duplicate**:
Two or more YOLO grains in the canonical **instance prediction set** that correspond to one ground-truth grain, often when adjacent **sliding window** tiles each detect part of the same grain and **score merge** cannot fuse them (no overlapping masks). Treated as a known limitation of the current YOLO system, not a separate post-processing stage.
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
Per-model watershed (or other) settings used after semantic prediction to produce **extracted grains** at test time. Selected from tune JSON or CLI defaults per checkpoint, not from the **test inference recipe**.
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
Copying every file a dataset manifest references into a local work directory and rewriting the manifest so paths are local. Downstream jobs should not depend on scratch or ephemeral paths once staged (ADR 0002).
_Avoid:_ partial staging (images only)

**Run provenance**:
Inference and post-processing parameters for a model run (e.g. **test inference recipe**, **YOLO inference profile**, **U-Net extraction profile**, **score merge**), stored once per run output directory alongside **prediction set directory** (ADR 0001).
_Avoid:_ metadata, .extract_meta (legacy filename)

### Test evaluation policy

**Variant test ranking**:
The primary ordering of **input configuration**s on held-out test: whole-section sliding-window inference, ranked by instance **AJI** with **F1@IoU50** reported alongside. **Supporting test metrics** are not the headline rank.
_Avoid:_ test mAP, patch mean AJI as headline (training-crop or detector-native, not deployment unit)

**Model test comparison**:
Comparing **producer** families (YOLO vs U-Net) on the same input variant and test mosaic: same headline as **variant test ranking** — whole-section instance **AJI** (and **F1@IoU50**) under the shared **test inference recipe**; both use non-overlapping grain lists (U-Net via extraction after semantic prediction, YOLO via **score merge** at predict time).
_Avoid:_ comparing models on patch mAP alone, YOLO-only **Mask AP** as the cross-model headline

**Supporting test metrics**:
Diagnostics bundled with default YOLO test jobs but not used for **variant test ranking**: patch-level instance AJI/F1 and patch detector mAP; whole-section **Mask AP** on the canonical YOLO **instance prediction set**.
_Avoid:_ patch metrics as headline rank, optional val on patch jobs (default bundle always includes val)

**Patch metric aggregate**:
Summary of patch-level instance metrics across the test split: (1) **unweighted mean** over patches that have at least one grain in ground truth, with empty-GT tiles excluded and counted separately; (2) **grain-weighted mean** over the same grain-bearing patches, weighting each patch by its GT instance count.
_Avoid:_ mean over all patches (includes empty tiles), simple average (ambiguous which rule)

**Mask AP**:
Instance mask average precision on the canonical YOLO **instance prediction set** against vector ground truth on whole-section test samples. Measures the full YOLO system including **score merge**, not raw **detector proposals**. Not used for U-Net or patch samples in default pipelines.
_Avoid:_ mask AP as cross-model headline, mask AP on overlapping proposals

**Test inference recipe**:
The single shared held-out test configuration for every **input configuration** and both **producer** families: whole-section **sliding window** geometry, patch crop size and batching, the frozen **YOLO inference profile**, and supporting YOLO val settings. Enables fair **variant test ranking** and **model test comparison**. U-Net instance extraction stays per **U-Net extraction profile**. Recorded in **run provenance** (ADR 0003).
_Avoid:_ per-variant inference settings on test, duplicating recipe constants in job scripts

### YOLO profile selection

**YOLO inference profile**:
The five train-selected YOLO inference knobs beyond shared **sliding window** geometry in the **test inference recipe**: slice-merge postprocess settings, minimum **score**, and **mask threshold**. One profile shared across all input variants on held-out test. Chosen via **profile selection**, then **profile promotion** into the recipe. Recorded in **run provenance**. Re-run **profile selection** when train labels or YOLO weights change materially.
_Avoid:_ per-variant profiles (confounds **variant test ranking**), tuning on overlapping **detector proposals** as the primary objective

**Profile selection**:
Train-side search for the shared **YOLO inference profile** on the whole **train** section, maximizing mean **AJI** (after **score merge**) averaged across input variants. Winner feeds **profile promotion**; audit trail on scratch (ADR 0005).
_Avoid:_ tuning on overlapping **detector proposals** as the headline objective

**Profile selection result row**:
One grid candidate’s audit record: profile knob values, per-variant and mean train **AJI**, and what inputs the score depended on. Rows assemble into the tune-run audit table (ADR 0005).
_Avoid:_ treating a stale row as valid after labels or weights change

**Profile selection scoring**:
Computing train **AJI** for one grid point from **tiled detector proposals** through slice-merge and **score merge** to a **merged instance view**, without persisting a full **instance prediction set** for that point. Held-out test still uses full predict and canonical prediction artifacts (ADR 0007).
_Avoid:_ skipping **score merge**, requiring prediction-set JSON equality on every grid point

**Profile selection ground truth cache**:
The canonical train ground-truth **merged instance view** for a tune run, built once from vector labels and reused by all **profile selection scoring** tasks across input variants (label geometry is shared; channels are not). ADR 0006.
_Avoid:_ per-variant GT caches when label geometry is shared, using semantic TIFFs as GT for profile selection

**Profile promotion**:
Installing the **profile selection** winner into the **test inference recipe** for git commit and held-out test (all five profile knobs). ADR 0005.
_Avoid:_ partial promotion (only merge or only score/mask without the full profile)

### Post-eval reporting

**Post-eval reporting**:
Aggregating finished test eval artifacts into comparison tables and thesis figures after cluster test jobs complete. Not part of inference or per-job metric computation. Headline charts use whole-section **AJI** and **F1@IoU50**; **supporting test metrics** appear in separately labelled panels.
_Avoid:_ eval pipeline, in-job plotting, quality index (legacy composite)

**Eval run discovery**:
How **post-eval reporting** finds finished eval outputs before building the **reporting bundle**. Convention-based on scratch in v1 (see README); a future **eval run catalog** may replace path conventions.
_Avoid:_ submit script name as discovery key, required TSV before plotting

**Reporting bundle**:
The regenerated output of **post-eval reporting** (figures, derived tables, run summary). Not versioned in git.
_Avoid:_ repo `reports/` binaries, figures committed to git
