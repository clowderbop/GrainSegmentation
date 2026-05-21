# shellcheck shell=bash
# Source from SLURM submit wrappers after set -euo pipefail.

# shellcheck source=SLURM/utils/paths.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/paths.sh"
