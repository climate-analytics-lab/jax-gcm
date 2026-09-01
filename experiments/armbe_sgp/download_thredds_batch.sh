#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 START_DATE END_DATE_EXCLUSIVE OUTPUT_DIRECTORY" >&2
  echo "example: $0 2011-06-24 2011-07-01 ~/Downloads/arm-microbase" >&2
  exit 2
fi

start_date=$1
end_date=$2
output_directory=$3
base_url="https://thredds-ui.svcs.arm.gov/thredds/fileServer/orders/fisherm1/268516/sgpmicrobaseC1.c1"

if [[ -z ${THREDDS_COOKIE:-} ]]; then
  read -r -s -p "Paste the _oauth2_proxy cookie value: " THREDDS_COOKIE
  echo
fi
if [[ $THREDDS_COOKIE != _oauth2_proxy=* ]]; then
  THREDDS_COOKIE="_oauth2_proxy=$THREDDS_COOKIE"
fi
trap 'unset THREDDS_COOKIE' EXIT

mkdir -p "$output_directory"

valid_microbase_file() {
  local path=$1
  local size
  size=$(stat -f %z "$path")
  [[ $size -gt 600000000 ]] || return 1
  if command -v ncdump >/dev/null 2>&1; then
    ncdump -h "$path" >/dev/null
  fi
}

python3 - "$start_date" "$end_date" <<'PY' | while IFS= read -r filename; do
import sys
from datetime import date, timedelta

start = date.fromisoformat(sys.argv[1])
end = date.fromisoformat(sys.argv[2])
if end <= start:
    raise SystemExit("END_DATE_EXCLUSIVE must be later than START_DATE")
while start < end:
    print(f"sgpmicrobaseC1.c1.{start:%Y%m%d}.000000.nc")
    start += timedelta(days=1)
PY
  final_path="$output_directory/$filename"
  partial_path="$final_path.part"
  if [[ -f $final_path ]]; then
    if valid_microbase_file "$final_path"; then
      echo "skip verified $filename"
      continue
    fi
    echo "remove invalid existing file $filename" >&2
    rm "$final_path"
  fi

  echo "download $filename"
  effective_url=$(curl \
    --fail \
    --location \
    --continue-at - \
    --retry 10 \
    --retry-all-errors \
    --retry-delay 10 \
    --connect-timeout 30 \
    --speed-limit 1024 \
    --speed-time 120 \
    --cookie "$THREDDS_COOKIE" \
    --output "$partial_path" \
    --write-out '%{url_effective}' \
    "$base_url/$filename")

  if [[ $effective_url != "$base_url/$filename" ]] || ! valid_microbase_file "$partial_path"; then
    rm -f "$partial_path"
    echo "authentication failed or THREDDS returned a non-NetCDF response" >&2
    echo "refresh the browser login, obtain a new _oauth2_proxy cookie, and rerun" >&2
    exit 1
  fi
  mv "$partial_path" "$final_path"
  echo "verified $filename ($(stat -f %z "$final_path") bytes)"
done
