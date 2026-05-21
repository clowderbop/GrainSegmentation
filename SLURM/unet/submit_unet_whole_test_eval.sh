#!/bin/bash

set -euo pipefail
# shellcheck source=SLURM/utils/repo_root.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/repo_root.sh"
GRAINSEG_ROOT="$(grainseg_root)"
TEST_DIR="$GRAINSEG_ROOT/dataset/test"

sbatch "$REPO_ROOT/SLURM/unet/run_unet_whole_test_eval.sh" \
  --model-dir "$GRAINSEG_ROOT/models/unet" \
  --image-dir "$TEST_DIR" \
  --mask-dir "$TEST_DIR" \
  --gt-gpkg "$TEST_DIR/test_labels.gpkg" \
  --output-dir "$GRAINSEG_ROOT/eval/unet_test" \
  --watershed-tune-root "$GRAINSEG_ROOT/runs/watershed_tune"
