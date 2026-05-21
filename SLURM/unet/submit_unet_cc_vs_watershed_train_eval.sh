#!/bin/bash
# Compare connected-components vs tuned watershed on the train section.

set -euo pipefail
# shellcheck source=SLURM/utils/repo_root.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/repo_root.sh"
cd "$REPO_ROOT"

GRAINSEG_ROOT="$(grainseg_root)"
TRAIN_DIR="$GRAINSEG_ROOT/dataset/train"
MODELS_DIR="$GRAINSEG_ROOT/models/unet"
GT_GPKG="$TRAIN_DIR/train_labels.gpkg"
WATERSHED_TUNE_ROOT="$GRAINSEG_ROOT/runs/watershed_tune"
EVAL_ROOT="$GRAINSEG_ROOT/eval"

sbatch --job-name=watershed_val "$REPO_ROOT/SLURM/unet/run_unet_whole_test_eval.sh" \
  --model-dir "$MODELS_DIR" \
  --image-dir "$TRAIN_DIR" \
  --mask-dir "$TRAIN_DIR" \
  --gt-gpkg "$GT_GPKG" \
  --output-dir "$EVAL_ROOT/watershed_val" \
  --instance-method watershed \
  --watershed-tune-root "$WATERSHED_TUNE_ROOT"

sbatch --job-name=cc_val "$REPO_ROOT/SLURM/unet/run_unet_whole_test_eval.sh" \
  --model-dir "$MODELS_DIR" \
  --image-dir "$TRAIN_DIR" \
  --mask-dir "$TRAIN_DIR" \
  --gt-gpkg "$GT_GPKG" \
  --output-dir "$EVAL_ROOT/cc_val" \
  --instance-method cc

echo "Submitted CC vs watershed train-section eval jobs."
