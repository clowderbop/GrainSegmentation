# shellcheck shell=bash
_slurm_utils="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${REPO_ROOT:-}" ] || [ -z "${SLURM_ROOT:-}" ]; then
    # shellcheck source=SLURM/utils/paths.sh
    source "$_slurm_utils/paths.sh"
fi
# shellcheck source=SLURM/utils/assertions.sh
source "$_slurm_utils/assertions.sh"
unset _slurm_utils

TF_WHEEL_NAME="tensorflow-2.17.0+nv25.2-cp312-cp312-linux_x86_64.whl"

install_unet_tensorflow_wheel() {
    local wheel_root="${1:-$(grainseg_root)/wheels}"
    local wheel_path="$wheel_root/$TF_WHEEL_NAME"
    require_file "$wheel_path" "TensorFlow wheel not found"
    echo "Installing TensorFlow wheel..."
    uv pip install \
        nvidia-cudnn-cu12~=9.0 \
        nvidia-nccl-cu12 \
        nvidia-cuda-runtime-cu12~=12.8.0 \
        nvidia-cusparse-cu12 \
        nvidia-cufft-cu12 \
        nvidia-cusolver-cu12 \
        nvidia-cuda-nvcc-cu12 \
        nvidia-cuda-nvrtc-cu12 \
        "$wheel_path"
}
