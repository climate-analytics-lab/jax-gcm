#!/usr/bin/env bash

set -euo pipefail

GPU_INDEX="${1:-0}"
OUTPUT_TAG="${2:-idle_gpu${GPU_INDEX}}"
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
cd "$REPO_ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "$REPO_ROOT/.benchmark-venv/bin/python" ]]; then
  PY="$REPO_ROOT/.benchmark-venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  printf 'No Python interpreter found. Activate the project environment or set PYTHON=/path/to/python.\n' >&2
  exit 1
fi

if ! "$PY" -c 'import jax, jcm' >/dev/null 2>&1; then
  printf 'Python %s cannot import jax and jcm. Activate the project environment or set PYTHON=/path/to/python.\n' "$PY" >&2
  exit 1
fi

SITE_PACKAGES="$("$PY" -c 'import site; print(site.getsitepackages()[0])')"
NVIDIA_ROOT="$SITE_PACKAGES/nvidia"

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
# Model.resume uses jax.debug.callback, which requires a host CPU device even
# though the model arrays and computation remain on the default CUDA backend.
export JAX_PLATFORMS=cuda,cpu
if [[ -d "$NVIDIA_ROOT" ]]; then
  export LD_LIBRARY_PATH="$NVIDIA_ROOT/cublas/lib:$NVIDIA_ROOT/cuda_cupti/lib:$NVIDIA_ROOT/cuda_nvrtc/lib:$NVIDIA_ROOT/cuda_runtime/lib:$NVIDIA_ROOT/cudnn/lib:$NVIDIA_ROOT/cufft/lib:$NVIDIA_ROOT/cusolver/lib:$NVIDIA_ROOT/cusparse/lib:$NVIDIA_ROOT/nccl/lib:$NVIDIA_ROOT/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
fi

"$PY" -c 'import jax; print("Benchmark device:", jax.devices())'

"$PY" experiments/armbe_sgp/benchmark_evolution_cost.py \
  --scheme speedy-t31l8 \
  --cycle-steps 10 \
  --repeats 10 \
  --output "experiments/armbe_sgp/outputs/evolution_cost_speedy_t31l8_${OUTPUT_TAG}.json"

"$PY" experiments/armbe_sgp/benchmark_evolution_cost.py \
  --scheme echam-rrtmgp-t63l47 \
  --cycle-steps 10 \
  --repeats 10 \
  --no-compute-cre \
  --output "experiments/armbe_sgp/outputs/evolution_cost_echam_rrtmgp_t63l47_${OUTPUT_TAG}.json"

"$PY" experiments/armbe_sgp/benchmark_evolution_cost.py \
  --scheme echam-rrtmgp-t63l47 \
  --cycle-steps 10 \
  --repeats 10 \
  --compute-cre \
  --output "experiments/armbe_sgp/outputs/evolution_cost_echam_rrtmgp_cre_t63l47_${OUTPUT_TAG}.json"
