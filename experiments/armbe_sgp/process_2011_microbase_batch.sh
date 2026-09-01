#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 START_DATE END_DATE_EXCLUSIVE" >&2
  exit 2
fi

root=/data/MOSAIC/jax-gcm/experiments/armbe_sgp
export JAX_PLATFORMS=cpu
export PYTHONPATH="/data/MOSAIC/jax-gcm:/data/MOSAIC/.venv/lib/python3.12/site-packages${PYTHONPATH:+:$PYTHONPATH}"

exec /data/MOSAIC/tools/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12 \
  "$root/process_microbase_month.py" \
  --microbase-dir /home/ubuntu/globus-staging/order-268516 \
  --atmosphere "$root/data/order-267892/ftp.archive.arm.gov/fisherm1/267892/sgparmbeatmC1.c1/sgparmbeatmC1.c1.20110101.000000.cdf" \
  --cldrad "$root/data/order-267892/ftp.archive.arm.gov/fisherm1/267892/sgparmbecldradC1.c1/sgparmbecldradC1.c1.20110101.003000.nc" \
  --start "$1" \
  --end "$2" \
  --output "$root/outputs/echam_layer_cloud_2011" \
  --delete-raw-after-verify
