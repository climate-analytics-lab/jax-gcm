#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DIRECTORY_CONTAINING_JUNE_24_TO_30_RAW_FILES" >&2
  exit 2
fi

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
raw_source=${1%/}
config="$root/microbase_benchmark.toml"
python="$root/.venv-microbase/bin/python"
raw_stage="$root/benchmark-workspace/raw/2011/2011-06"

if [[ ! -x $python ]]; then
  echo "run ./setup_microbase_mac.sh first" >&2
  exit 1
fi

mkdir -p "$raw_stage"
for day in 24 25 26 27 28 29 30; do
  filename="sgpmicrobaseC1.c1.201106${day}.000000.nc"
  if [[ ! -f "$raw_source/$filename" ]]; then
    echo "missing $raw_source/$filename" >&2
    exit 1
  fi
  ln -sfn "$raw_source/$filename" "$raw_stage/$filename"
done

"$python" "$root/microbase_mac_pipeline.py" run \
  --config "$config" \
  --month 2011-06 \
  --start 2011-06-24 \
  --end 2011-07-01 \
  --stop-after processed_verified

for day in 24 25 26 27 28 29 30; do
  date="2011-06-${day}"
  "$python" "$root/microbase_mac_pipeline.py" compare \
    --candidate "$root/benchmark-workspace/reduced/schema4/sgp/C1/2011/2011-06/$date" \
    --reference "$root/benchmark/reference/$date"
done

"$python" "$root/microbase_mac_pipeline.py" status \
  --config "$config" --month 2011-06
echo "seven-day Apple Silicon benchmark passed"
