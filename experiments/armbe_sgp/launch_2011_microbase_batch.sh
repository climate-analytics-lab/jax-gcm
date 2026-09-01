#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 START_DATE END_DATE_EXCLUSIVE" >&2
  exit 2
fi

root=/data/MOSAIC/jax-gcm/experiments/armbe_sgp
log="$root/outputs/echam_layer_cloud_2011_pipeline.log"
nohup "$root/process_2011_microbase_batch.sh" "$1" "$2" \
  >"$log" 2>&1 </dev/null &
echo $!
