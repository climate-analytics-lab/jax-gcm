# Environment for the ARMBE single-column experiment.
#
# Source this before running any script here:
#     source /data/MOSAIC/jax-gcm/experiments/armbe_sgp/env.sh
#
# Why each line matters on this box:
# - The venv's jax[cuda12] ships its own CUDA 12.9 libs. The system
#   LD_LIBRARY_PATH points at /usr/local/cuda-12.1, whose older libcusparse
#   shadows the bundled one and forces jax back to CPU. We drop it so jax uses
#   its self-contained libraries.
# - The 8 GPUs are shared. Pin to one and disable XLA's default ~75% memory
#   pre-allocation so we don't fight other users (the SCM needs almost nothing).

export VIRTUAL_ENV=/data/MOSAIC/.venv
export PATH="/data/MOSAIC/.venv/bin:$PATH"

unset LD_LIBRARY_PATH                       # use jax's bundled CUDA 12.9 libs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # one GPU by default
export XLA_PYTHON_CLIENT_PREALLOC=false      # don't grab 75% up front
export XLA_PYTHON_CLIENT_ALLOCATOR=platform  # on-demand allocation, polite on shared GPUs
