echo "Loading modules..."
module purge
module load Python/3.12.3-GCCcore-13.3.0
module list

echo "Preparing new job-specific environment..."

export PATH="$HOME/.local/bin:$PATH"
echo "using uv: $(uv --version && which uv)"

export UV_PROJECT_ENVIRONMENT="$TMPDIR/.venv"
export VIRTUAL_ENV="$UV_PROJECT_ENVIRONMENT"

export UV_LINK_MODE=copy

export LD_LIBRARY_PATH="$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cudnn/lib:$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib:$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cublas/lib:$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cufft/lib:$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cusparse/lib:$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cusolver/lib:$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/nccl/lib:$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

export PATH="$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cuda_nvcc/bin:$PATH"

export XLA_FLAGS="--xla_gpu_cuda_data_dir=$UV_PROJECT_ENVIRONMENT/lib/python3.12/site-packages/nvidia/cuda_nvcc"
