# Grain segmentation

Research codebase comparing U-Net (semantic segmentation + instance extraction) and YOLO (direct instance segmentation) on sandstone thin-section microscopy, across multi-modal input variants.

## Language

**Instance prediction set**:
The canonical per-sample model output: a list of non-overlapping grain instances, each with geometry (COCO RLE). YOLO entries also carry **score** (winning proposal after **score merge** at predict time); U-Net entries do not.
_Avoid:_ handover file, predictions JSON, NPZ (format names, not the concept)

**Instance label map**:
A single raster image where each pixel carries one instance id (or background); used internally when metrics need a painted map. Not persisted as the canonical prediction artifact.
_Avoid:_ instance map, instances TIFF (implementation paths)

**Grain instance**:
One segmented grain region in an image, represented as a mask or polygon regardless of which model produced it.

**Detector proposal**:
A detected grain mask and **score** from the YOLO instance segmentation model before **score merge**. Proposals may overlap; the default predict pipeline does not persist them as the canonical **instance prediction set**.
_Avoid:_ detection, prediction row (too generic), canonical YOLO output (use **instance prediction set**)

**Tiled detector proposals**:
The full set of **detector proposals** from every **sliding window** slice, each shifted to whole-image coordinates, before SAHI slice-merge postprocess (the merge knobs in **YOLO inference profile** selection). Overlapping; not the canonical **instance prediction set**. **Profile selection** may reuse them across grid candidates that share the same weights, train sample, window recipe, minimum **score**, and **mask threshold** so only slice-merge and **score merge** need to rerun.
_Avoid:_ pre-merge cache, SAHI pickle (implementation paths)

**Extracted grain**:
One entry in a U-Net instance prediction set: a grain region produced by instance extraction (connected components or watershed) on a semantic prediction. Extracted grains do not overlap each other.
_Avoid:_ watershed instance, CC label (method names)

**Score merge**:
YOLO post-processing that resolves overlapping **detector proposals** into non-overlapping grains by painting higher-**score** masks over lower-score ones. Runs at predict time for both whole and patch **sample unit**; each surviving grain keeps the **score** of its winning proposal. Produces the canonical **instance prediction set**.
_Avoid:_ NMS, confidence merge (legacy name), instance map from masks (implementation)

**Slice-boundary duplicate**:
Two or more YOLO grains in the canonical **instance prediction set** that correspond to one ground-truth grain, often when adjacent **sliding window** tiles each detect part of the same grain and **score merge** cannot fuse them (no overlapping masks). Treated as a known limitation of the current YOLO system, not a separate post-processing stage.
_Avoid:_ split grain, tile artifact (informal only)

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
A single raster where each pixel has one instance id (or background). For YOLO, equivalent to rasterizing the canonical **instance prediction set** after **score merge**; for U-Net, rasterizing **extracted grains**. Built transiently for instance-level metrics when a label map is required.
_Avoid:_ pred map, prediction raster (too vague)

**Prediction overlay**:
A visualization that blends predicted grain regions onto the microscopy image from the canonical **instance prediction set** (non-overlapping grains for both **producer** families).
_Avoid:_ SAHI visualization (pipeline step name)

**Producer**:
Which model family wrote an instance prediction set: `yolo` (non-overlapping grains with **score**) or `unet` (non-overlapping **extracted grains** without **score**). JSON and derived reporting tables use the field name `producer`; thesis figures do not show that word on axes (see **Model display label**).
_Avoid:_ model_type, source (too vague)

**Model display label**:
Thesis-facing name for a **producer** family on post-eval figures and legends: **YOLO** and **U-Net**. Axis titles read **Model**; tick and legend entries use these labels, not registry tokens (`yolo`, `unet`). Aligns with **Model test comparison** prose. Data columns and eval JSON keep `producer`.
_Avoid:_ Producer on figure axes, renaming `producer` in code, using Model for input configuration (that is **Variant display name**)

**Sample unit**:
Whether a manifest row refers to a whole microscopy section or a single patch crop. Carried on dataset manifests (`patch` or `whole`), not duplicated inside instance prediction set files.
_Avoid:_ scale, tile (ambiguous)

**Sample id**:
The stable key for one manifest row and its instance prediction set file (`prediction_sets/{sample_id}.json`). For whole-section samples it is the data split (`train`, `test`), not the on-disk mosaic or channel stem. For patch samples it is the patch stem (e.g. `region_0001_y00000_x00000`).
_Avoid:_ TIFF filename, channel suffix, split directory name (paths, not ids)

**Run provenance**:
Inference and post-processing parameters for a model run (e.g. minimum detection score / `conf`, SAHI slice size, **score merge**, watershed settings), stored once per run output directory alongside prediction sets.
_Avoid:_ metadata, .extract_meta (legacy filename)

**Staging**:
Copying every file a dataset manifest references (images, masks, vector labels, YOLO segment labels, etc.) into a local work directory and rewriting the manifest so `path_base` is that directory. Downstream jobs should not depend on scratch paths once staged.
_Avoid:_ TMPDIR copy (implementation), partial staging (images only)

**Mask AP**:
COCO instance mask average precision on the canonical YOLO **instance prediction set** (non-overlapping grains with **score**) against vector ground truth on whole-section test samples. Measures the full YOLO system including **score merge**, not raw overlapping **detector proposals**. Not used for U-Net or patch samples in the default pipelines.
_Avoid:_ mask_ap_metrics.json (filename), COCO AP (unless distinguishing from instance AJI/F1)

**Variant test ranking**:
The primary ordering of multi-modal input variants on held-out test: whole-section sliding-window inference, ranked by instance **AJI** with **F1@IoU50** reported alongside. Patch and Ultralytics metrics are supporting evidence, not the headline rank.
_Avoid:_ test mAP, patch mean AJI as headline (training-crop or detector-native, not deployment unit)

**Model test comparison**:
Comparing **producer** families (YOLO vs U-Net) on the same input variant and test mosaic: same headline as **variant test ranking** — whole-section instance **AJI** (and **F1@IoU50**) under the shared **test inference recipe**; both use non-overlapping grain lists (U-Net via extraction after semantic prediction, YOLO via **score merge** at predict time).
_Avoid:_ comparing models on patch mAP alone, YOLO-only **Mask AP** as the cross-model headline

**Supporting test metrics**:
Diagnostics bundled with default YOLO test jobs but not used for **variant test ranking**: patch-level instance AJI/F1 plus Ultralytics segmentation mAP on the patch test split (native detector output on patches, not the whole-section YOLO system); whole-section **Mask AP** on SAHI runs (COCO on the canonical YOLO **instance prediction set**, same **grain class** as predictions).
_Avoid:_ optional val, secondary eval (vague), RUN_ULTRALYTICS_VAL (implementation flag; val is not optional in the default patch test job)

**Patch metric aggregate**:
Summary of patch-level instance metrics across the test split: (1) **unweighted mean** over patches that have at least one grain in ground truth, with empty-GT tiles excluded and counted separately; (2) **grain-weighted mean** over the same grain-bearing patches, weighting each patch by its GT instance count.
_Avoid:_ mean over all patches (includes empty tiles), simple average (ambiguous which rule)

**Test inference recipe**:
The single shared configuration for held-out test inference, used identically for every input variant and for both **producer** families (YOLO and U-Net): whole-section **sliding window** (window size, stride or overlap — YOLO via SAHI, U-Net via patch-and-stitch), patch-crop size and batching, the frozen **YOLO inference profile**, and YOLO val settings. Enables fair **variant test ranking** and **model test comparison** on the same geometry. Does not include U-Net instance extraction (watershed); that stays per **U-Net extraction profile**. Recorded in **run provenance** on each eval run.
_Avoid:_ hardcoded SLURM constants (duplicated or divergent recipes), SAHI-only config (U-Net whole-section must match the same window recipe)

**Mask threshold**:
Detector mask binarization cutoff passed to Ultralytics or SAHI at inference (whole **sliding window** and patch predict). Not re-applied when building the **instance prediction set** from an already-binarized mask plane.
_Avoid:_ encode-time threshold (implies a second binarization pass), confidence (minimum **score** is separate)

**YOLO inference profile**:
Train-selected YOLO settings for **sliding window** inference beyond shared window geometry in the **test inference recipe**: SAHI slice-merge postprocess (`postprocess_type`, `match_metric`, `match_threshold`), minimum **score** (`conf`), and **mask threshold**. **One profile shared across all input variants** on held-out test. **Profile selection** runs a **full factorial grid** over all five knobs on the train section (search space in repo config), maximizing mean whole-section train **AJI** on the canonical **instance prediction set** (after **score merge**), averaged across registry variants; **tiled detector proposals** are reused across grid points that share the same weights, sample, window recipe, **conf**, and **mask threshold**. On the cluster, slice inference runs as **parallel detector jobs** (one SLURM job per variant and detector-key pair, writing shared `_work/` under a submit-assigned run directory); a **grid coordinator job** runs afterward (strict dependency on all detector jobs) to execute merge, eval, and audit under `grid/`. A tune job may **resume** after interrupt: by default it skips grid points that already have per-variant metrics on disk; pass `--no-resume` to re-run all merge+eval paths while still reusing valid tiled-proposal cache entries. Pre-merge metrics are diagnostic only. Chosen on the train section, promoted into **`configs/test_inference.yaml`** (the **test inference recipe**) and **committed to git** with the search grid spec; scratch tune runs hold supplementary audit tables; recorded in **run provenance** on each eval run. Re-run **profile selection** when train labels or YOLO weights change materially, not after every single-variant training job.
_Avoid:_ per-variant conf or merge settings (confounds **variant test ranking**), tuning on overlapping **detector proposals** as the primary objective, informal SAHI defaults without documenting selection, staged coordinate search (superseded by full grid with proposal reuse)

**Profile promotion**:
Copying the **profile selection** grid winner into the **test inference recipe** (`configs/test_inference.yaml`) for git commit and held-out test (all five profile knobs).
_Avoid:_ HITL promote (informal), partial profile (merge-only or score/mask-only commit without the full five knobs)

**U-Net extraction profile**:
Per-model watershed (or other) settings used after semantic prediction to produce **extracted grains** at test time. Selected from tune JSON or CLI defaults per checkpoint, not from the **test inference recipe**.
_Avoid:_ global watershed in test recipe, extraction recipe (ambiguous with instance prediction set)

**Post-eval reporting**:
Aggregating finished test eval artifacts on scratch and producing comparison tables and thesis figures. Runs after SLURM eval jobs complete; not part of inference or per-job metric computation. Ingests whole-section metrics for headline charts plus **supporting test metrics** (patch aggregates, **Mask AP**, Ultralytics val) in separately labelled panels. Headline figures rank by **AJI** with **F1@IoU50** alongside; derived efficiency views (gain vs PPL, per-input) use those metrics, not composite quality indices.
_Avoid:_ eval pipeline, in-job plotting (implementation placement), quality index (legacy composite)

**Eval run discovery**:
How **post-eval reporting** locates finished eval outputs on scratch. V1 uses path conventions per **producer**, **variant**, and **sample unit** (whole vs patch; latest job dir for patch runs). No catalog file in v1; a future **eval run catalog** may pin paths when conventions are insufficient.
_Avoid:_ submit script name as discovery key, yolo-only paths, required TSV before plotting

**Reporting bundle**:
The **post-eval reporting** output tree under the grainseg eval area on scratch (`figures/`, `derived/` tables, `analysis_summary.json`). Not versioned in git; regenerated by the reporting CLI after eval jobs finish. Initial figure set: headline AJI/F1 heatmap and model×input-configuration bars, PPL-relative delta heatmap, and one supporting YOLO patch-val panel; additional diagnostic charts deferred.
_Avoid:_ repo `reports/` binaries, figures committed to git

**Variant display name**:
Thesis-facing label for an input configuration on tables and figures. Stored on each registry variant in `config/variants.yaml` (`display_name`); join key remains the variant key. In prose, names are emphasised (e.g. *FullStack*). Figures use these strings on axes, not registry keys or legacy names (`All-Stack`, `All-Comp`).
_Avoid:_ modality (ambiguous), registry key on plot axes, separate reporting config for labels only

**PPL configuration**:
Single plane-polarized (PPL) image only. Registry variant `PPL`. Display name **PPL**.
_Avoid:_ PPL-only, 1-channel (implementation)

**FullStack configuration**:
PPL plus each cross-polarized (XPL) image as separate inputs. Registry variant `PPL+AllPPX`. Display name **FullStack**.
_Avoid:_ All-Stack, PPL+AllPPX on figures, seven-input (channel count)

**PPL+XPLComp configuration**:
PPL plus one screen-blend composite of all XPL images. Registry variant `PPL+PPXblend`. Display name **PPL+XPLComp**.()
_Avoid:_ PPL+XPL-Comp, PPL+PPXblend on figures (registry spelling)

**FullComp configuration**:
Single screen-blend composite of PPL and all XPL images. Registry variant `PPLPPXblend`. Display name **FullComp**.
_Avoid:_ All-Comp, PPLPPXblend on figures, PPLPPXblend (file stem, not thesis name)
