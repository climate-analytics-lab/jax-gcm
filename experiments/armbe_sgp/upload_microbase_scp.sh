#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 LOCAL_DIRECTORY [FILENAME]" >&2
  exit 2
fi

local_directory=${1%/}
host=${MOSAIC_HOST:-roselab1.ucsd.edu}
port=${MOSAIC_PORT:-12000}
identity=${MOSAIC_IDENTITY:-$HOME/.ssh/mfisher}
remote_directory=/home/ubuntu/globus-staging/order-268516

shopt -s nullglob
if [[ $# -eq 2 ]]; then
  files=("$local_directory/$2")
else
  files=("$local_directory"/sgpmicrobaseC1.c1.*.nc)
fi
if [[ ${#files[@]} -eq 0 ]]; then
  echo "no MICROBASE files found in $local_directory" >&2
  exit 1
fi

for file in "${files[@]}"; do
  filename=${file##*/}
  local_size=$(stat -f %z "$file")
  remote_size=$(
    ssh -p "$port" -i "$identity" "ubuntu@$host" \
      "stat -c %s '$remote_directory/$filename' 2>/dev/null || true"
  )
  if [[ $remote_size == "$local_size" ]]; then
    echo "skip verified remote $filename"
    continue
  fi

  echo "upload $filename ($local_size bytes)"
  scp -P "$port" -i "$identity" \
    "$file" "ubuntu@$host:$remote_directory/$filename.part"
  remote_size=$(
    ssh -p "$port" -i "$identity" "ubuntu@$host" \
      "stat -c %s '$remote_directory/$filename.part'"
  )
  if [[ $remote_size != "$local_size" ]]; then
    echo "size mismatch for $filename: local=$local_size remote=$remote_size" >&2
    exit 1
  fi
  ssh -p "$port" -i "$identity" "ubuntu@$host" \
    "mv '$remote_directory/$filename.part' '$remote_directory/$filename'"
  echo "verified remote size $filename"
done
