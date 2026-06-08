# Grain segmentation

Research codebase comparing U-Net (semantic segmentation + instance extraction) and YOLO (direct instance segmentation) on sandstone thin-section microscopy, across multi-modal input variants.

## Language

### Instance outputs & geometry

**Instance prediction set**:
The canonical per-sample model output: a list of non-overlapping grain instances, each with encoded mask geometry. YOLO entries carry **score**; U-Net entries do not.
_Avoid:_ handover file, predictions JSON, NPZ (format names, not the concept)

**Instance label map**:
A raster where each pixel carries one instance id or background; used internally when metrics need a painted map.
_Avoid:_ instance map, instances TIFF (implementation paths)

**Grain instance**:
One segmented grain region in an image, as a mask or polygon regardless of which model produced it.

**Grain class**:
The single object category for instance segmentation in this project.
_Avoid:_ category, label id (ambiguous with instance ids)

**Merged instance view**:
A raster where each pixel has one instance id or background, built from an **instance prediction set** or vector ground truth.
_Avoid:_ pred map, prediction raster (too vague)

**Prediction overlay**:
A visualization that blends predicted grain regions onto the microscopy image.
_Avoid:_ SAHI visualization (pipeline step name)

**Prediction set directory**:
The per-run folder holding one **instance prediction set** per **sample id**.
_Avoid:_ instances/ (legacy label-map folder name)

### YOLO detection & tiling

**Detector proposal**:
A YOLO-detected grain mask and **score** before fusion into non-overlapping grains. Proposals may overlap.
_Avoid:_ detection, prediction row (too generic), canonical YOLO output (use **instance prediction set**)

**Tiled detector proposals**:
**Detector proposals** from every **sliding window** slice in whole-image coordinates, before **cross-tile association**. Overlapping; not the canonical **instance prediction set**.
_Avoid:_ pre-merge cache, SAHI pickle (implementation paths)

**Sliding window**:
Whole-section inference that runs the detector on overlapping crops and fuses results into one section output.
_Avoid:_ SAHI (library name), tile (ambiguous with patch **sample unit**)

**Cross-tile association**:
YOLO whole-section post-processing that fuses **tiled detector proposals** into non-overlapping grains.
_Avoid:_ slice-merge, score merge (whole-section path), NMS (legacy)

**Score**:
A detector-assigned value for how likely a YOLO **detector proposal** is a true grain instance.
_Avoid:_ confidence, probability (unless explicitly calibrated)

**Score merge**:
YOLO patch post-processing that resolves overlapping **detector proposals** by keeping higher-**score** masks over lower-score ones.
_Avoid:_ NMS, confidence merge (legacy name), using score merge for whole-section output (use **cross-tile association**)

**Slice-boundary duplicate**:
Two or more predicted grains that correspond to one ground-truth grain, often from adjacent **sliding window** tiles.
_Avoid:_ split grain, tile artifact (informal only)

**Mask threshold**:
Detector mask binarization cutoff at inference.
_Avoid:_ using **mask threshold** for minimum **score**

### U-Net outputs & extraction

**Semantic prediction**:
A U-Net per-pixel class label raster for one microscopy sample.
_Avoid:_ semantic map, class TIFF (implementation)

**Extracted grain**:
One grain region produced by instance extraction on a **semantic prediction**. Extracted grains do not overlap.
_Avoid:_ watershed instance, CC label (method names)

**U-Net extraction profile**:
Per-model settings used after semantic prediction to produce **extracted grains** at test time.
_Avoid:_ global watershed in test recipe, extraction recipe (ambiguous with **instance prediction set**)

### Models & input configurations

**Producer**:
Which model family wrote an **instance prediction set**: YOLO or U-Net.
_Avoid:_ model_type, source (too vague)

**Model display label**:
Thesis-facing name for a **producer** on figures and legends: **YOLO** and **U-Net**.
_Avoid:_ **Producer** on figure axes, using Model for **input configuration** (that is **variant display name**)

**Input configuration**:
A multi-modal microscopy input setup for training and evaluation (which channels or composites are fed to the model).
_Avoid:_ modality (ambiguous), All-Stack / All-Comp (legacy thesis names)

**Variant display name**:
Thesis-facing label for an **input configuration** on tables and figures.
_Avoid:_ registry key on plot axes, separate reporting config for labels only

**Input image count**:
The number of microscopy images in an **input configuration**, used as the thesis-facing complexity axis.
_Avoid:_ YOLO tensor channel count, RGB channel count

### Manifests & run layout

**Sample id**:
The stable key for one manifest row and its **instance prediction set**.
_Avoid:_ TIFF filename, channel suffix, split directory name (paths, not ids)

**Sample unit**:
Whether a manifest row refers to a whole microscopy section or a single patch crop.
_Avoid:_ scale, tile (ambiguous)

**Eval manifest**:
A dataset manifest augmented after inference with paths to predictions and ground truth, used to drive evaluation.
_Avoid:_ eval_manifest.json as a domain term (filename only)

**Staging**:
Mirroring data between persistent storage and a node-local work directory around a cluster job.
_Avoid:_ partial staging, pulling more than the job reads

**Run provenance**:
Inference and post-processing parameters for a model run, stored once per run output directory.
_Avoid:_ metadata, .extract_meta (legacy filename)

### Evaluation & metrics

**Individual grain instance recovery**:
The scientific target of recovering each annotated grain as a separate predicted instance on the whole section.
_Avoid:_ treating area coverage alone as sufficient

**Whole-section PQ**:
Panoptic Quality on the whole-section **merged instance view** for one-to-one matched grains.
_Avoid:_ patch-level PQ as the primary rank

**PQ diagnostics**:
Companion metrics reported alongside **whole-section PQ** on held-out evaluation.
_Avoid:_ treating tune-path audit records as the full diagnostic set

**MergedViewPqResult**:
The narrower PQ audit record persisted during train-side grid tuning.
_Avoid:_ calling this the full **PQ diagnostics** set

**Instance metric bundle**:
The standard metric set computed on held-out evaluation for each evaluated **sample unit**.
_Avoid:_ maintaining separate PQ definitions for tune vs eval

**AJI+**:
Supporting microscopy-style instance overlap metric using unique instance pairing.
_Avoid:_ treating AJI+ as sufficient evidence of one-for-one grain recovery

**Variant test ranking**:
The ordering of **input configuration**s on held-out test by **whole-section PQ**.
_Avoid:_ patch mean metrics as headline rank

**Model test comparison**:
Comparing **producer** families on the same **input configuration** and test mosaic under a shared **test inference recipe**.
_Avoid:_ comparing models on AP/mAP

**Supporting test metrics**:
Diagnostics bundled with eval jobs but not used for **variant test ranking**.
_Avoid:_ patch metrics as headline rank

**Patch metric aggregate**:
Summary of patch-level instance metrics across the test split.
_Avoid:_ mean over all patches (includes empty tiles), simple average (ambiguous which rule)

**Patch AP/mAP**:
YOLO-only detector diagnostics on patch data.
_Avoid:_ whole-section Mask AP, AP/mAP as cross-model evidence

**Test inference recipe**:
The shared held-out test configuration for every **input configuration** and both **producer** families.
_Avoid:_ per-variant inference settings on test

### YOLO profile selection

**YOLO inference profile**:
The train-selected detector minimum **score** plus fixed **mask threshold** used on held-out test.
_Avoid:_ per-variant profiles

**Profile selection**:
Train-side search for the shared **YOLO inference profile** on the whole train section.
_Avoid:_ tuning on overlapping **detector proposals** as the headline objective

**Profile selection scoring**:
Computing train **whole-section PQ** for one profile-selection grid point without persisting a full **instance prediction set**.
_Avoid:_ scoring with the full **instance metric bundle** on the tune hot path

**Profile selection ground truth cache**:
The shared train ground-truth **merged instance view** reused across profile-selection candidates.
_Avoid:_ per-variant GT caches when label geometry is shared

**Profile promotion**:
Installing the **profile selection** winner into the **test inference recipe** before held-out test.
_Avoid:_ partial promotion (only **score** without paired **mask threshold**)

### Post-eval reporting

**Post-eval reporting**:
Aggregating finished test eval artifacts into comparison tables and thesis figures after cluster jobs complete.
_Avoid:_ eval pipeline, in-job plotting, quality index (legacy composite)
