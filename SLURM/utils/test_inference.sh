# shellcheck shell=bash
# Load shared test inference recipe exports (config/test_inference.yaml).
# Requires REPO_ROOT (from enter_job.sh).

load_test_inference_exports() {
    # shellcheck disable=SC1090
    eval "$(uv run --directory "$REPO_ROOT" python -m common.test_inference)"
}
