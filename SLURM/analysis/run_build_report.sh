#!/bin/bash
#SBATCH --job-name=post_eval_report
#SBATCH --output=logs/post_eval_report-%j.log
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00

set -euo pipefail
# shellcheck source=SLURM/utils/enter_job.sh
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}/SLURM/utils/enter_job.sh"

module purge
module load Python/3.12.3-GCCcore-13.3.0
export PATH="$HOME/.local/bin:$PATH"

GRAINSEG_ROOT="${GRAINSEG_ROOT:-$(grainseg_root)}"
OUTPUT_DIR="${OUTPUT_DIR:-$GRAINSEG_ROOT/eval/reporting}"

REPORT_ARGS=(--grainseg-root "$GRAINSEG_ROOT" --output-dir "$OUTPUT_DIR")
if [[ "${REPORT_STRICT:-0}" == "1" ]]; then
    REPORT_ARGS+=(--strict)
fi
if [[ "${REPORT_NO_FIGURES:-0}" == "1" ]]; then
    REPORT_ARGS+=(--no-figures)
fi

echo "GrainSeg root: $GRAINSEG_ROOT"
echo "Reporting bundle: $OUTPUT_DIR"

cd "$REPO_ROOT"
uv sync --group analysis
uv run --group analysis python -m analysis.build_report "${REPORT_ARGS[@]}"
