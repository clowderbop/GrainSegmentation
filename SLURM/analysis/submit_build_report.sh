#!/bin/bash

set -euo pipefail
# shellcheck source=SLURM/utils/repo_root.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/repo_root.sh"
cd "$REPO_ROOT"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Submit post-eval reporting (tables + thesis figures) after test eval jobs finish.
Cluster workflow: docs/runbooks/analysis.md#build-report

Outputs: $SCRATCH/GrainSeg/eval/reporting/ (override with OUTPUT_DIR).

Environment (optional, forwarded via sbatch --export):
  GRAINSEG_ROOT     Scratch GrainSeg root (default: grainseg_root from paths.sh)
  OUTPUT_DIR        Reporting bundle directory
  REPORT_STRICT=1   Fail if whole-section eval artifacts are missing
  REPORT_NO_FIGURES=1  Skip matplotlib figures

Examples:
  bash SLURM/analysis/submit_build_report.sh
  OUTPUT_DIR=/scratch/$USER/GrainSeg/eval/reporting_v2 bash SLURM/analysis/submit_build_report.sh
EOF
    exit 0
fi

mkdir -p logs
sbatch \
    --export=ALL \
    --job-name=post_eval_report \
    "$REPO_ROOT/SLURM/analysis/run_build_report.sh"

echo "Submitted post-eval reporting job."
