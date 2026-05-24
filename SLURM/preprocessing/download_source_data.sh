#!/bin/bash
#SBATCH --job-name=GrainSeg_download_uncropped
#SBATCH --output=download-%j.log
#SBATCH --mem=8GB
#SBATCH --time=08:00:00

# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"
source "$SLURM_ROOT/prepare_env.sh"

GRAINSEG_ROOT="$(grainseg_root)"

module load lz4/1.9.4-GCCcore-12.3.0

cd "$REPO_ROOT/src/data_prep" && uv run python -u download_data.py \
  -o "$GRAINSEG_ROOT/dataset/uncropped"
