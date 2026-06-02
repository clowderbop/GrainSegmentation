# Scratch filesystem layout

Persistent project data lives under **`$SCRATCH/GrainSeg`** by default (`grainseg_root()` in `SLURM/utils/paths.sh`). The git repo holds `config/variants.yaml` and code; scratch holds datasets, models, runs, eval outputs, and manifests.

`{variant}` is a registry key: `PPL`, `PPLPPXblend`, `PPL+PPXblend`, `PPL+AllPPX`. Watershed tune directories use **`slugs.job`** from the registry (e.g. `PPL_AllPPX`, not `PPL+AllPPX`).

## Directory tree

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
│   ├── yolo_inference_profile_tune/<run_id>/            # profile selection audit (see YOLO runbook)
│   └── watershed_tune/{slugs.job}/                      # e.g. PPL_AllPPX, PPL_PlusPPXblend
│       └── watershed_best_*.json
├── tuning_logs/                                         # U-Net hyperparameter search (per run_name)
└── eval/                                                # SLURM eval job outputs (layout varies by script)
    ├── instance_val_cc/, instance_val_watershed/        # train-section CC vs tuned watershed
    ├── unet_test/                                       # held-out whole-section test eval
    ├── unet_patches/{variant}/…/                        # patch test eval
    ├── yolo_patches/{variant}/…/
    ├── yolo_{variant}/                                  # whole SAHI test eval
    └── reporting/                                       # post-eval bundle (see analysis runbook)
```

On compute nodes, jobs copy **manifest-listed files only** into `$TMPDIR` before Python runs; see [`staging.md`](staging.md). The tree above is the canonical scratch layout before staging.

## Dataset contracts (summary)

Two versioned layers on scratch:

1. **Variant registry** — `config/variants.yaml` defines inputs, channel counts, path templates, and naming slugs. Python: `src/common/variants.py`; SLURM: `SLURM/utils/variants.sh` → `uv run python -m common.variants`.

2. **Dataset manifests** — JSON inventories under `dataset/` list exact files per sample. Schema: [`docs/manifests.md`](../manifests.md).

| Manifest | Path (relative to grainseg root) |
|----------|----------------------------------|
| U-Net train whole | `dataset/train/manifests/{variant}.whole.json` |
| U-Net test whole | `dataset/test/manifests/{variant}.whole.json` |
| YOLO / eval patches | `dataset/{train,test}/patches/{variant}/manifest.json` |
| U-Net patch test | `dataset/test/unet_from_yolo/{variant}/manifest.json` |

Replace `{variant}` with the registry key literally (`PPL+AllPPX`, not slug forms).

## Registry CLI

From the repo root (after `uv sync`):

```bash
# Shell exports for one variant (NUM_INPUTS, IMAGE_SUFFIXES, paths, slugs, …)
uv run python -m common.variants --grainseg-root "$SCRATCH/GrainSeg" env --variant 'PPL+AllPPX'

# All variant names (used by submit scripts)
uv run python -m common.variants all-names

# Debug JSON
uv run python -m common.variants print-json --variant PPL
```

## Regenerating manifests

After preprocessing produces per-channel TIFFs and patch datasets (see [`runbooks/preprocessing.md`](../runbooks/preprocessing.md)):

```bash
GRAINSEG_ROOT="${SCRATCH:-/scratch/$USER}/GrainSeg"

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
