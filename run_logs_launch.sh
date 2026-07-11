#!/bin/bash
# Launch one ne30L47 pySES+ECHAM+RRTMGP climatology on a given GPU.
# Usage: ./run_logs_launch.sh <gpu> <config: 1m|2m|2m-jam>
set -euo pipefail
GPU=$1; CFG=$2
PY=/home/dwatsonparris/micromamba/envs/jcm/bin/python
TS=$(date +%y%m%d_%H%M%S)
OUT=/scr/dwatsonparris/jcm_runs/pyses_ne30
mkdir -p $OUT
PREFIX=$OUT/ne30_rrtmgp_${CFG}_${TS}
LOG=$OUT/ne30_rrtmgp_${CFG}_${TS}.log
nohup env CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false PYSES_BACKEND=jax \
  $PY tools/run_pyses_climatology.py --config $CFG \
    --nx 30 --days 365 --chunk-days 10 --save-interval 5 \
    --physics-dt 1800 --nu-top 2.5e4 \
    --prefix $PREFIX \
  > $LOG 2>&1 &
echo "PID=$! GPU=$GPU CFG=$CFG PREFIX=$PREFIX LOG=$LOG"
