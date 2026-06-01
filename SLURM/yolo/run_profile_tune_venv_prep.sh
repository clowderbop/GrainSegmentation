#!/bin/bash
#SBATCH --job-name=yolo_prof_venv
#SBATCH --output=logs/yolo_prof_venv-%j.log
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

set -euo pipefail
# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"

source "$SLURM_ROOT/prepare_env.sh"
# shellcheck source=SLURM/utils/yolo_venv.sh
source "$SLURM_ROOT/utils/yolo_venv.sh"

export SHARED_VENV_ROOT="$(yolo_venv_shared_root)"
echo "Profile-tune shared YOLO venv → $SHARED_VENV_ROOT"
yolo_venv_prepare_shared
