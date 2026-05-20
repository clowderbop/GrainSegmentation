#!/bin/bash
# Compare connected-components vs tuned watershed on the train section.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GRAINSEG_ROOT="${SCRATCH:-/scratch/${USER}}/GrainSeg"
TRAIN_DIR="$GRAINSEG_ROOT/dataset/train"
MODELS_DIR="$GRAINSEG_ROOT/models/unet"
GT_GPKG="$TRAIN_DIR/train_labels.gpkg"
WATERSHED_TUNE_ROOT="$GRAINSEG_ROOT/runs/watershed_tune"
EVAL_ROOT="$GRAINSEG_ROOT/eval"

sbatch "$REPO_ROOT/SLURM/unet/run_unet_whole_test_eval.sh" \
  --model-dir "$MODELS_DIR" \
  --image-dir "$TRAIN_DIR" \
  --mask-dir "$TRAIN_DIR" \
  --gt-gpkg "$GT_GPKG" \
  --output-dir "$EVAL_ROOT/watershed_val" \
  --watershed-tune-root "$WATERSHED_TUNE_ROOT"

sbatch "$REPO_ROOT/SLURM/unet/run_unet_whole_test_eval.sh" \
  --model-dir "$MODELS_DIR" \
  --image-dir "$TRAIN_DIR" \
  --mask-dir "$TRAIN_DIR" \
  --gt-gpkg "$GT_GPKG" \
  --output-dir "$EVAL_ROOT/cc_val"

echo "Submitted CC vs watershed train-section eval jobs."
