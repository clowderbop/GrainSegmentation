# Grain segmentation

Research codebase comparing U-Net (semantic segmentation + instance extraction) and YOLO (direct instance segmentation) on sandstone thin-section microscopy, across multi-modal input variants.

## Language

**Instance prediction set**:
The canonical per-sample model output: a list of grain instances, each with geometry (COCO RLE) and optional confidence, for one microscopy sample.
_Avoid:_ handover file, predictions JSON, NPZ (format names, not the concept)

**Instance label map**:
A single raster image where each pixel carries one instance id (or background); used internally when metrics need a painted map. Not persisted as the canonical prediction artifact.
_Avoid:_ instance map, instances TIFF (implementation paths)

**Grain instance**:
One segmented grain region in an image, represented as a mask or polygon regardless of which model produced it.

**Detector proposal**:
One entry in a YOLO instance prediction set: a detected grain mask and confidence from the instance segmentation model. Proposals may overlap each other before evaluation merges them.
_Avoid:_ detection, prediction row (too generic)

**Extracted grain**:
One entry in a U-Net instance prediction set: a grain region produced by instance extraction (connected components or watershed) on a semantic prediction. Extracted grains do not overlap each other.
_Avoid:_ watershed instance, CC label (method names)

**Confidence merge**:
Resolving overlapping YOLO detector proposals into a single non-overlapping view by painting higher-confidence masks over lower-confidence ones. Used for instance-level metrics and overlays, not stored in the prediction set.
_Avoid:_ NMS, instance map from masks (implementation)

**Confidence**:
A detector-assigned score for how likely a YOLO detector proposal is a true grain instance. Required on YOLO proposals; not defined for U-Net extracted grains.
_Avoid:_ score (in U-Net context), probability (unless explicitly calibrated)

**Eval manifest**:
A dataset manifest augmented after inference with paths to prediction artifacts and ground truth, used to drive instance and mask evaluation jobs.
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
The non-overlapping grain layout produced by applying confidence merge to YOLO detector proposals, or taken directly from U-Net extracted grains. Used for instance-level metrics and may be built transiently at evaluation time.
_Avoid:_ pred map, prediction raster (too vague)

**Prediction overlay**:
A visualization that blends predicted grain regions onto the microscopy image. For YOLO, overlapping proposals are resolved on the canvas in confidence order; for U-Net, extracted grains are drawn without overlap.
_Avoid:_ SAHI visualization (pipeline step name)

**Producer**:
Which model family wrote an instance prediction set: `yolo` (detector proposals) or `unet` (extracted grains). Declares overlap and confidence rules for consumers.
_Avoid:_ model_type, source (too vague)

**Sample unit**:
Whether a manifest row refers to a whole microscopy section or a single patch crop. Carried on dataset manifests (`patch` or `whole`), not duplicated inside instance prediction set files.
_Avoid:_ scale, tile (ambiguous)

**Run provenance**:
Inference and post-processing parameters for a model run (e.g. confidence threshold, SAHI slice size, watershed settings), stored once per run output directory alongside prediction sets.
_Avoid:_ metadata, .extract_meta (legacy filename)

**Mask AP**:
COCO instance mask average precision computed from YOLO detector proposals (with confidence) against vector ground truth on whole-section test samples. A YOLO evaluation metric, not used for U-Net extracted grains or patch samples in the default pipelines.
_Avoid:_ mask_ap_metrics.json (filename), COCO AP (unless distinguishing from instance AJI/F1)
