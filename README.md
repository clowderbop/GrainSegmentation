[Hippocratic License HL3-FULL](https://firstdonoharm.dev/version/3/0/full.html)

# GrainSegmentation

**Research question:** How do different multi-modal microscopy input configurations affect instance grain segmentation accuracy in sandstone thin-section images, when using U-Net semantic segmentation with postprocessing-based instance extraction versus YOLO direct instance segmentation?

The project compares four microscopy input variants across two model families, giving eight experiment pipelines:

- `PPL`: single-input baseline.
- `PPLPPXblend`: single blended composite input.
- `PPL+PPXblend`: two-input PPL plus PPX-blend configuration.
- `PPL+AllPPX`: seven-input configuration using PPL and all PPX images.

In this codebase, `PPX` refers to cross-polarized light (`XPL`).

For each variant, the intended experiment sequence is:

1. Tune YOLO hyperparameters.
2. Train the final YOLO model using the selected hyperparameters.
3. Evaluate YOLO on the held-out test set, patch-wise and whole-image-wise using sliding-window.
4. Tune U-Net hyperparameters.
5. Train the final U-Net model using the selected hyperparameters.
6. Tune watershed instance-extraction hyperparameters for the U-Net semantic predictions.
7. Compare connected components and tuned watershed instance extraction for U-Net outputs on the train set.
8. Evaluate the U-Net (using the best postprocessing method) on the held-out test set, patch-wise and whole-image-wise using sliding-window.
9. Compare results across all model families and input variants.

## Dataset contracts

Training and evaluation use two versioned layers on scratch (`$SCRATCH/GrainSeg` by default; see `SLURM/utils/paths.sh`):

1. **Variant registry** — `config/variants.yaml` defines inputs, channel counts, path templates, and naming slugs (`slugs.job` for watershed tune dirs, `yolo.yaml_name`, etc.). Python loads it via `src/common/variants.py`; SLURM uses `SLURM/utils/variants.sh` → `uv run python -m common.variants`.

2. **Dataset manifests** — JSON inventories under `dataset/` list exact files per sample. Contracts and field rules are in `docs/manifests.md`.

| Manifest | Path (relative to grainseg root) |
|----------|----------------------------------|
| U-Net train whole | `dataset/train/manifests/{variant}.whole.json` |
| U-Net test whole | `dataset/test/manifests/{variant}.whole.json` |
| YOLO / eval patches | `dataset/{train,test}/patches/{variant}/manifest.json` |
| U-Net patch test | `dataset/test/unet_from_yolo/{variant}/manifest.json` |

Replace `{variant}` with the registry key literally (`PPL+AllPPX`, not slug forms).

### Scratch filesystem layout

Persistent project data lives under **`$SCRATCH/GrainSeg`** (see `grainseg_root()` in `SLURM/utils/paths.sh`). The repo holds `config/variants.yaml` and code; scratch holds datasets, models, runs, and eval outputs. `{variant}` is one of `PPL`, `PPLPPXblend`, `PPL+PPXblend`, `PPL+AllPPX`. Watershed tune folders use registry **`slugs.job`** (e.g. `PPL_AllPPX`, not `PPL+AllPPX`).

```
$SCRATCH/GrainSeg/
├── dataset/
│   ├── uncropped.tar.lz4              # cached download (optional)
│   ├── uncropped/                     # extracted source mosaics + raw gpkg (preprocessing)
│   ├── train/
│   │   ├── train_PPL.tif … train_PPX6.tif
│   │   ├── train_PPXblend.tif, train_PPLPPXblend.tif
│   │   ├── train_PPL+PPXblend.tif, train_PPL+AllPPX.tif   # stacked YOLO mosaics (not in U-Net whole manifests)
│   │   ├── train_labels.gpkg, train_labels.tif
│   │   ├── manifests/
│   │   │   └── {variant}.whole.json                     # U-Net whole-section (per-channel images only)
│   │   └── patches/{variant}/
│   │       ├── manifest.json                            # patch inventory (train)
│   │       ├── {variant}.yaml                           # Ultralytics data.yaml (from write_patch_manifests)
│   │       ├── images/train/, images/val/
│   │       └── labels/train/, labels/val/
│   └── test/
│       ├── test_*.tif, test_labels.gpkg, test_labels.tif
│       ├── manifests/
│       │   └── {variant}.whole.json
│       ├── patches/{variant}/
│       │   ├── manifest.json
│       │   ├── images/test/, labels/test/
│       │   └── …
│       └── unet_from_yolo/{variant}/                     # U-Net patch test crops (from YOLO patches)
│           ├── manifest.json
│           ├── images/
│           └── masks/
├── models/
│   └── unet/
│       ├── pretrained/starting_point.keras
│       └── unet_finetuned_{variant}.keras               # default names in whole_eval_models.tsv
├── runs/
│   ├── yolo26-seg/{variant}/weights/best.pt             # YOLO training outputs
│   ├── yolo26-seg-val/{variant}/                       # YOLO patch eval run dirs (optional layout)
│   └── watershed_tune/{slugs.job}/                      # e.g. PPL_AllPPX, PPL_PlusPPXblend
│       └── watershed_best_*.json
├── tuning_logs/                                         # U-Net hyperparameter search (per run_name)
└── eval/                                                # SLURM eval job outputs (layout varies by script)
    ├── cc_val/, watershed_val/                          # train-section U-Net instance comparison
    ├── unet_test/                                       # whole-section test eval
    ├── unet_patches/{variant}/…/
    └── yolo_patches/{variant}/…/
```

On compute nodes, SLURM jobs copy **manifest-listed files only** into `$TMPDIR/…/manifest.json` (see [Staging on cluster nodes](#staging-on-cluster-nodes)); the tree above is the canonical scratch layout before staging.

### Registry CLI

From the repo root (after `uv sync`):

```bash
# Shell exports for one variant (NUM_INPUTS, IMAGE_SUFFIXES, paths, slugs, …)
uv run python -m common.variants --grainseg-root "$SCRATCH/GrainSeg" env --variant 'PPL+AllPPX'

# All variant names (used by submit scripts)
uv run python -m common.variants all-names

# Debug JSON
uv run python -m common.variants print-json --variant PPL
```

### Regenerating manifests on scratch

After preprocessing steps 1–8 in `SLURM/preprocessing/pipeline.md` (especially per-channel `train_*.tif` / `test_*.tif` and patch datasets):

```bash
GRAINSEG_ROOT="${SCRATCH:-/scratch/$USER}/GrainSeg}"

# Whole-section manifests (all four variants; U-Net uses per-channel images only)
uv run --directory src/data_prep python write_whole_manifests.py \
  --grainseg-root "$GRAINSEG_ROOT"

# Patch manifests + YOLO data.yaml files
uv run --directory src/data_prep python write_patch_manifests.py \
  --grainseg-root "$GRAINSEG_ROOT" \
  --write-yolo-yamls

# Test U-Net patch crops + unet_from_yolo manifests (needs test mosaics on disk)
uv run --directory src/data_prep python write_patch_manifests.py \
  --grainseg-root "$GRAINSEG_ROOT" \
  --split test \
  --write-unet-manifests
```

Validate paths without writing: add `--dry-run` to either script. After manifests exist:

```bash
uv run --directory src/data_prep python validate_dataset_manifests.py \
  --grainseg-root "$GRAINSEG_ROOT"
```

### Staging on cluster nodes

SLURM jobs copy only manifest-listed files to `$TMPDIR`, then pass `--manifest` to Python:

```bash
uv run python -m common.stage_manifest run \
  "$GRAINSEG_ROOT/dataset/train/manifests/PPL+AllPPX.whole.json" \
  "$TMPDIR/unet_inputs"
```

U-Net whole-section eval uses `SLURM/unet/whole_eval_models.tsv` (model basename + variant per row) and `--manifest-split train|test` with staged whole manifests. Patch and train jobs require the manifests above; they do not scan dataset directories for TIFFs.

### Manual smoke test (cluster)

After manifests exist and models are trained, compare CC vs watershed on the train section:

```bash
sbatch SLURM/unet/submit_cc_vs_watershed_train_eval.sh
```

Expect both jobs to evaluate all four variants via staged whole manifests (no stacked-TIFF directory scan).

## Dataset

The project starts from seven aligned high-resolution microscopy images of the same sandstone thin-section:

- 1 Plane-Polarized Light image (`PPL`)
- 6 Cross-Polarized Light images (`PPX1` to `PPX6`) captured at 15 degree angle increments between 0 and 75 degrees

Two subsections of the thin-section are used for the thesis experiments:

- A larger section for tuning, training, validation, and U-Net postprocessing selection.
- A smaller held-out section for final testing.

Initial labels were generated by running SAM2 on the PPL image using sliding-window inference. These mask polygons were then imported in QGIS, where they were manually corrected and refined before they were quality-checked by fixing invalid geometries and removing holes and fully-contained features. Finally, as the last step in QGIS, they were smoothed and buffered by 5 px.

As the QGIS processing introduced overlapping polygons, a custom overlap splitting step produces strictly non-overlapping grain masks where previously overlapping polygons touch at a shared boundary.

The algorithm resolves overlaps by:

1. Identifying connected components of overlapping polygons using a spatial index and bounding-box intersection graph.
2. For each overlapping pair, computing the exact intersection polygon.
3. Calculating a topological centerline that splits the overlap in two halves using Voronoi polygons handled by `pygeoops`.
4. Smoothing the centerline (using Taubin and Chaikin smoothing algorithms) to remove Voronoi-originated zigzags.
5. Snapping the centerline endpoints exactly to the outer boundaries of the intersecting grains.
6. Splitting the overlap along this smoothed centerline.
7. Assigning the resulting halves to the original adjacent polygons based on a ray-cast heuristic by building lines from the midpoint of the centerline to the exclusive areas of the polygons.

The corrected polygons and aligned microscopy images are cropped to the selected train and test sections. For U-Net, the polygons are rasterized into three semantic classes:

- background
- grain interior
- grain boundary (3px band around each grain)

Two composite variants are also derived:

- `PPLPPXblend`: a single composite input made from PPL and the PPX images using screen blending.
- `PPL+PPXblend`: a two-input variant using PPL plus a screen-blended PPX composite.

### Train/Validation Split and Patch Extraction

To process the large high-resolution training section, the data is spatially split and patchified:

1. **Spatial tiling:** The training section is divided into 4096px x 4096px spatial tiles.
2. **Coverage stratification:** Grain coverage is computed for each tile. Tiles with less than 10% grain coverage are assigned strictly to training. The remaining eligible tiles are binned by coverage and split using stratified sampling, with 80% used for training and 20% used for validation.
3. **Patch extraction:** The selected tiles are cropped into 1024px x 1024px patches with 50% overlap for training and validation. YOLO receives instance polygon labels; U-Net receives raster semantic masks.

## Research Pipeline

### Model Families

Two segmentation model families are evaluated:

- **U-Net:** semantic segmentation into background, grain interior, and grain boundary, followed by postprocessing to extract instances.
- **YOLO segmentation:** direct instance segmentation.

### YOLO Workflow

All input variants are loaded as a single TIFF image. `PPL` and `PPLPPXblend` are 3-channel TIFFs, `PPL+PPXblend` is a 6-channel stacked TIFF, and `PPL+AllPPX` is a 21-channel stacked TIFF.

For each input variant, the YOLO workflow is:

1. Tune hyperparameters with the Ultralytics tuner over learning rate and dropout.
2. Train the final YOLO segmentation model using the best selected hyperparameters.
3. Evaluate on the held-out test data in two ways:
  - patch-wise evaluation on the non-overlapping 1024px x 1024px patches
  - whole-section sliding-window evaluation (using SAHI)

### U-Net Workflow

Input variants are built by loading individual RGB TIFF images. `PPL` and `PPLPPXblend` both use one 3-channel TIFF. `PPL+PPXblend` uses two 3-channel TIFFs (`PPL` and `PPXblend`) that are concatenated inside the model to 6 channels. `PPL+AllPPX` uses seven 3-channel TIFFs (`PPL` and `PPX1`-`PPX6`) that are concatenated inside the model to 21 channels. The U-Net loader expects each input TIFF to be exactly 3 channels and will reject stacked 6/21-channel TIFFs.

For each input variant, the U-Net workflow is:

1. Tune model hyperparameters with Bayesian optimization over learning rate and dropout.
2. Train the final U-Net model using the best selected hyperparameters.
3. Tune watershed instance-extraction hyperparameters by maximizing AJI against the training-section ground-truth instances.
4. Run sliding-window inference on the training section and compare two instance extraction strategies:
  - connected components from the predicted grain interior class
  - tuned watershed using the predicted interior and boundary classes
5. Select the better U-Net instance extraction method.
6. Evaluate the selected U-Net pipeline on the held-out test data in two ways:
  - patch-wise segmentation evaluation on the test patch dataset
  - whole-section segmentation and semantic evaluation using custom sliding-window inference

### Metrics and Comparison

Evaluation metrics include:

- **AJI (Aggregated Jaccard Index):** An instance-aware metric specifically designed for microscopy and cell segmentation. AJI directly penalizes under-segmentation (merged grains) and over-segmentation (split grains) at the pixel level. It provides a holistic view of both detection and boundary adherence without relying on confidence thresholds.
- **Precision:** The ratio of correctly predicted positive observations to the total predicted positives. It indicates how many of the segmented grains are actually grains.
- **Recall:** The ratio of correctly predicted positive observations to all observations in actual class. It measures how many of the actual grains were successfully segmented.
- **F1 Score:** The harmonic mean of Precision and Recall, providing a single metric that balances both false positives and false negatives.
- **Mean P/R/F1 over IoU 0.50-0.95:** `mP_iou50_95`, `mR_iou50_95`, and `mF1_iou50_95` average precision, recall, and F1 at IoU thresholds 0.50, 0.55, ..., 0.95 using the same matching rule at each threshold.

For the YOLO models, **COCO-style mask AP (Average Precision)** is also used. It decouples detection performance from spatial accuracy by averaging across multiple IoU thresholds (`mAP@0.5:0.95`) and uses YOLO's prediction confidence scores. AP is not calculated for U-Net models because they don't output the needed confidence scores.
