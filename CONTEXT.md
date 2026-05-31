# Grain segmentation

Research codebase comparing U-Net (semantic segmentation + instance extraction) and YOLO (direct instance segmentation) on sandstone thin-section microscopy, across multi-modal input variants.

## Language

**Instance prediction set**:
The canonical per-sample model output: a list of grain instances, each with geometry (COCO RLE) and optional **score** (YOLO only), for one microscopy sample.
_Avoid:_ handover file, predictions JSON, NPZ (format names, not the concept)

**Instance label map**:
A single raster image where each pixel carries one instance id (or background); used internally when metrics need a painted map. Not persisted as the canonical prediction artifact.
_Avoid:_ instance map, instances TIFF (implementation paths)

**Grain instance**:
One segmented grain region in an image, represented as a mask or polygon regardless of which model produced it.

**Detector proposal**:
One entry in a YOLO instance prediction set: a detected grain mask and **score** from the instance segmentation model. Proposals may overlap each other before evaluation merges them.
_Avoid:_ detection, prediction row (too generic)

**Extracted grain**:
One entry in a U-Net instance prediction set: a grain region produced by instance extraction (connected components or watershed) on a semantic prediction. Extracted grains do not overlap each other.
_Avoid:_ watershed instance, CC label (method names)

**Score merge**:
Resolving overlapping YOLO detector proposals into a single non-overlapping view by painting higher-**score** masks over lower-score ones. Used for instance-level metrics and overlays, not stored in the prediction set.
_Avoid:_ NMS, confidence merge (legacy name), instance map from masks (implementation)

**Score**:
A detector-assigned value for how likely a YOLO detector proposal is a true grain instance. Stored as JSON field `score` on each detection (COCO-aligned). Required on YOLO proposals; must be absent on U-Net extracted grains.
_Avoid:_ confidence (use **score** in docs and schema), probability (unless explicitly calibrated)

**Eval manifest**:
A dataset manifest augmented after inference with paths to prediction artifacts and ground truth, used to drive instance and mask evaluation jobs. Its work directory is the eval manifest parent (the run output folder on scratch). Every path the eval pipeline reads — ground truth, anchor image, instance prediction sets — must resolve under that directory. `write-eval` copies assets out of ephemeral staging when needed; evaluation must not depend on `$TMPDIR` still existing after inference.
_Avoid:_ eval_manifest.json as a domain term (filename only)

**Prediction set directory**:
The per-run folder holding one instance prediction set file per sample (`prediction_sets/{sample_id}.json`).
_Avoid:_ instances/ (legacy label-map folder name)

**Semantic prediction**:
A U-Net per-pixel class label raster for one microscopy sample. Used for pixel-wise semantic metrics, not for cross-model instance comparison with YOLO.
_Avoid:_ semantic map, class TIFF (implementation)

**Grain class**:
The single object category for instance segmentation in this project (identifier `0`). All entries in an instance prediction set use this class until multi-class segmentation is introduced.
_Avoid:_ category, label id (ambiguous with instance ids)

**Merged instance view**:
The non-overlapping grain layout produced by applying **score merge** to YOLO detector proposals, or taken directly from U-Net extracted grains. Used for instance-level metrics and may be built transiently at evaluation time.
_Avoid:_ pred map, prediction raster (too vague)

**Prediction overlay**:
A visualization that blends predicted grain regions onto the microscopy image. For YOLO, overlapping proposals are resolved on the canvas in **score** order; for U-Net, extracted grains are drawn without overlap.
_Avoid:_ SAHI visualization (pipeline step name)

**Producer**:
Which model family wrote an instance prediction set: `yolo` (detector proposals) or `unet` (extracted grains). Declares overlap and score-field rules for consumers.
_Avoid:_ model_type, source (too vague)

**Sample unit**:
Whether a manifest row refers to a whole microscopy section or a single patch crop. Carried on dataset manifests (`patch` or `whole`), not duplicated inside instance prediction set files.
_Avoid:_ scale, tile (ambiguous)

**Sample id**:
The stable key for one manifest row and its instance prediction set file (`prediction_sets/{sample_id}.json`). For whole-section samples it is the data split (`train`, `test`), not the on-disk mosaic or channel stem. For patch samples it is the patch stem (e.g. `region_0001_y00000_x00000`).
_Avoid:_ TIFF filename, channel suffix, split directory name (paths, not ids)

**Run provenance**:
Inference and post-processing parameters for a model run (e.g. minimum detection score / `conf`, SAHI slice size, watershed settings), stored once per run output directory alongside prediction sets.
_Avoid:_ metadata, .extract_meta (legacy filename)

**Staging**:
Copying every file a dataset manifest references (images, masks, vector labels, YOLO segment labels, etc.) into a local work directory and rewriting the manifest so `path_base` is that directory. Downstream jobs should not depend on scratch paths once staged.
_Avoid:_ TMPDIR copy (implementation), partial staging (images only)

**Mask AP**:
COCO instance mask average precision computed from YOLO detector proposals (with **score**) against vector ground truth on whole-section test samples. A YOLO evaluation metric, not used for U-Net extracted grains or patch samples in the default pipelines.
_Avoid:_ mask_ap_metrics.json (filename), COCO AP (unless distinguishing from instance AJI/F1)

**Variant test ranking**:
The primary ordering of multi-modal input variants on held-out test: whole-section sliding-window inference, ranked by instance **AJI** with **F1@IoU50** reported alongside. Patch and Ultralytics metrics are supporting evidence, not the headline rank.
_Avoid:_ test mAP, patch mean AJI as headline (training-crop or detector-native, not deployment unit)

**Model test comparison**:
Comparing **producer** families (YOLO vs U-Net) on the same input variant and test mosaic: same headline as **variant test ranking** — whole-section instance **AJI** (and **F1@IoU50**) under the shared **test inference recipe**; U-Net reaches instances via extraction after semantic prediction, YOLO via detector proposals and **score merge**.
_Avoid:_ comparing models on patch mAP alone, YOLO-only **Mask AP** as the cross-model headline

**Supporting test metrics**:
Diagnostics bundled with default YOLO test jobs but not used for **variant test ranking**: patch-level instance AJI/F1 (custom **merged instance view**) plus Ultralytics segmentation mAP on the patch test split (always computed on patch test jobs); whole-section **Mask AP** on SAHI runs (COCO on detector proposals, same **grain class** as predictions).
_Avoid:_ optional val, secondary eval (vague), RUN_ULTRALYTICS_VAL (implementation flag; val is not optional in the default patch test job)

**Patch metric aggregate**:
Summary of patch-level instance metrics across the test split: (1) **unweighted mean** over patches that have at least one grain in ground truth, with empty-GT tiles excluded and counted separately; (2) **grain-weighted mean** over the same grain-bearing patches, weighting each patch by its GT instance count.
_Avoid:_ mean over all patches (includes empty tiles), simple average (ambiguous which rule)

**Test inference recipe**:
The single shared configuration for held-out test inference, used identically for every input variant and for both **producer** families (YOLO and U-Net): whole-section **sliding window** (window size, stride or overlap — YOLO via SAHI, U-Net via patch-and-stitch), patch-crop size and batching, and YOLO-specific thresholds (**score** / `conf`) and val settings. Enables fair **variant test ranking** and **model test comparison** on the same geometry. Does not include U-Net instance extraction (watershed); that stays per trained model. Recorded in **run provenance** on each eval run.
_Avoid:_ per-variant conf, hardcoded SLURM constants (duplicated or divergent recipes), SAHI-only config (U-Net whole-section must match the same window recipe)

**U-Net extraction profile**:
Per-model watershed (or other) settings used after semantic prediction to produce **extracted grains** at test time. Selected from tune JSON or CLI defaults per checkpoint, not from the **test inference recipe**.
_Avoid:_ global watershed in test recipe, extraction recipe (ambiguous with instance prediction set)
