#!/usr/bin/env bash
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive CLOUDSDK_CORE_DISABLE_PROMPTS=1 PYTHONUNBUFFERED=1
WORK=/var/lib/jcm-rtx
mkdir -p "$WORK/results"
export RESULT_DIR="$WORK/results"
exec > >(tee -a "$RESULT_DIR/job.log" /dev/console) 2>&1
metadata() {
  curl --fail --silent --show-error --retry 3 --max-time 15 \
    -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}
RESULT_URI=$(metadata result-uri)
UPLOADER=''
finish() {
  local code=$?
  trap - EXIT
  set +e
  if [[ -n "$UPLOADER" ]]; then kill "$UPLOADER"; wait "$UPLOADER"; fi
  printf '{"exit_code":%d,"finished_at":"%s"}\n' "$code" "$(date -u +%FT%TZ)" > "$RESULT_DIR/status.json"
  tar -czf "$WORK/results.tar.gz" -C "$RESULT_DIR" .
  if timeout 180 gcloud storage cp "$WORK/results.tar.gz" "$RESULT_URI/results.tar.gz" --quiet; then
    timeout 60 gcloud storage cp "$RESULT_DIR/status.json" "$RESULT_URI/status.json" --quiet
  fi
  shutdown -h now
}
trap finish EXIT
(
  while sleep 60; do
    timeout 30 gcloud storage cp "$RESULT_DIR/job.log" "$RESULT_URI/logs/$(date -u +%Y%m%dT%H%M%SZ).log" --quiet || true
  done
) &
UPLOADER=$!
cd "$WORK"
gcloud storage cp "$(metadata source-uri)" source.tar.gz --quiet
printf '%s  source.tar.gz\n' "$(metadata source-sha256)" | sha256sum --check -
mkdir source
tar -xzf source.tar.gz -C source
metadata constraints > constraints.txt
nvidia-smi > "$RESULT_DIR/nvidia-smi.txt"
apt-get update
apt-get install -y --no-install-recommends python3-venv python3-pip
python3 -m venv venv
PYTHON="$WORK/venv/bin/python"
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -c constraints.txt -e ./source 'jax[cuda12]==0.10.2'
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze > "$RESULT_DIR/pip-freeze.txt"
cp "$WORK/source/tools/rtx_spot/plan.json" "$RESULT_DIR/plan.json"
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cuda JAX_ENABLE_X64=0 \
  timeout --signal=TERM --kill-after=30s 100m "$PYTHON" -u "$WORK/source/tools/rtx_spot/sweep.py"
