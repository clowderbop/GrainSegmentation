#!/bin/bash

set -euo pipefail

# shellcheck source=SLURM/utils/repo_root.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../utils/repo_root.sh"
# shellcheck source=SLURM/utils/variants.sh
source "$SLURM_ROOT/utils/variants.sh"
# shellcheck source=SLURM/utils/slurm_export.sh
source "$SLURM_ROOT/utils/slurm_export.sh"
# shellcheck source=SLURM/utils/assertions.sh
source "$SLURM_ROOT/utils/assertions.sh"

GRID_CONFIG="${GRID_CONFIG:-$REPO_ROOT/config/watershed_tune_grid.yaml}"
VARIANT=""
RUN_TAG=""
DEPENDENCY=""

function usage {
    cat <<'EOF' >&2
Usage: submit_watershed_tune_merge.sh --variant NAME --run-tag TAG [--dependency JOBID] [--grid-config PATH]

Submit one watershed tune merge job for an existing shard run tag.

Use after shard arrays have finished when merge failed or was skipped.
EOF
    exit "${1:-1}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            VARIANT="$2"
            shift 2
            ;;
        --run-tag)
            RUN_TAG="$2"
            shift 2
            ;;
        --dependency)
            DEPENDENCY="$2"
            shift 2
            ;;
        --grid-config)
            GRID_CONFIG="$2"
            shift 2
            ;;
        --help)
            usage 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            ;;
    esac
done

if [ -z "$VARIANT" ] || [ -z "$RUN_TAG" ]; then
    echo "Both --variant and --run-tag are required." >&2
    usage
fi

if [ ! -f "$GRID_CONFIG" ]; then
    echo "Grid config not found: $GRID_CONFIG" >&2
    exit 1
fi

slug="$(job_slug "$VARIANT")"
subdir="$(watershed_tune_subdir_for_variant "$VARIANT")"
output_dir="$(grainseg_root)/runs/watershed_tune/$subdir"
require_dir "$output_dir" "Watershed tune output dir not found: $output_dir"

shard_glob="$output_dir/watershed_grid_${RUN_TAG}_shard_*.csv"
if ! compgen -G "$shard_glob" > /dev/null; then
    echo "No shard CSVs match: $shard_glob" >&2
    exit 1
fi

merge_export="$(slurm_export_line \
    "$(slurm_export_assign VARIANT "$VARIANT")" \
    "$(slurm_export_assign RUN_TAG "$RUN_TAG")" \
    "$(slurm_export_assign GRID_CONFIG "$GRID_CONFIG")")"
merge_cmd=(
    sbatch
    "--job-name=TuneWatershedMerge_${slug}"
    "--export=${merge_export}"
    "$REPO_ROOT/SLURM/unet/run_watershed_tune_merge.sh"
    --variant "$VARIANT"
    --run-tag "$RUN_TAG"
)
if [ -n "$DEPENDENCY" ]; then
    merge_cmd=("${merge_cmd[@]:0:2}" "--dependency=afterok:${DEPENDENCY}" "${merge_cmd[@]:2}")
fi

"${merge_cmd[@]}"
