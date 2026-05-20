#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GRAINSEG_ROOT="${SCRATCH:-/scratch/${USER}}/GrainSeg"
TEST_DIR="$GRAINSEG_ROOT/dataset/test"

sbatch "$REPO_ROOT/SLURM/unet/run_unet_whole_test_eval.sh" \
  --model-dir "$GRAINSEG_ROOT/models/unet" \
  --image-dir "$TEST_DIR" \
  --mask-dir "$TEST_DIR" \
  --gt-gpkg "$TEST_DIR/test_labels.gpkg" \
  --output-dir "$GRAINSEG_ROOT/eval/unet_test" \
  --watershed-tune-root "$GRAINSEG_ROOT/runs/watershed_tune"
