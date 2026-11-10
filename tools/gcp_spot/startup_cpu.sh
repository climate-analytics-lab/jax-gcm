#!/usr/bin/env bash
set -Eeuo pipefail
export CLOUDSDK_CORE_DISABLE_PROMPTS=1
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1

WORK=/var/lib/jax-gcm-spot
mkdir -p "$WORK/results"
export RESULT_DIR="$WORK/results"
exec > >(tee -a "$RESULT_DIR/job.log" /dev/console) 2>&1

metadata() {
  curl --fail --silent --show-error --retry 3 --max-time 15 \
    -H 'Metadata-Flavor: Google' \
    "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"
}

RESULT_URI=''
UPLOADER=''
finish() {
  local code=$?
  trap - EXIT
  set +e
  if [[ -n "$UPLOADER" ]]; then kill "$UPLOADER"; wait "$UPLOADER"; fi
  echo "Runner finished with exit code $code at $(date -u +%FT%TZ)"
  printf '{"exit_code":%d,"finished_at":"%s"}\n' "$code" "$(date -u +%FT%TZ)" > "$RESULT_DIR/status.json"
  if [[ -n "$RESULT_URI" ]]; then
    tar -czf "$WORK/results.tar.gz" -C "$RESULT_DIR" .
    # Publish the completion marker only after the full bundle is durable.
    if timeout 120 gcloud storage cp "$WORK/results.tar.gz" "$RESULT_URI/results.tar.gz" --quiet; then
      timeout 60 gcloud storage cp "$RESULT_DIR/status.json" "$RESULT_URI/status.json" --quiet
    fi
  fi
  # The submitting process deletes the VM and its boot disk after reading status.
  # Server-side max-run-duration also deletes a VM if this runner hangs.
  shutdown -h now
}
trap finish EXIT

RESULT_URI=$(metadata result-uri)
# Plain CPU images do not contain Google Cloud CLI or NVIDIA drivers.
apt-get -o DPkg::Lock::Timeout=300 update
apt-get -o DPkg::Lock::Timeout=300 install -y ca-certificates curl gnupg
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor --yes -o /usr/share/keyrings/cloud.google.gpg
echo 'deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main' > /etc/apt/sources.list.d/google-cloud-sdk.list
apt-get update
apt-get install -y google-cloud-cli

SOURCE_URI=$(metadata source-uri)
EXPECTED_SHA=$(metadata source-sha256)
export TOTAL_DAYS SAVE_DAYS REPEATS TRUNCATION
TOTAL_DAYS=$(metadata total-days)
SAVE_DAYS=$(metadata save-days)
REPEATS=$(metadata repeats)
TRUNCATION=$(metadata truncation)
metadata benchmark-constraints > "$WORK/constraints.txt"
JOB_MODE=$(metadata job-mode)
export RESUME_FROM
RESUME_FROM=$(metadata resume-from 2>/dev/null || true)
export RESULT_URI

# New object names let the service account use objectCreator without overwrite
# or delete access. Logs survive preemption even without an EXIT trap.
(
  while sleep 60; do
    timeout 30 gcloud storage cp "$RESULT_DIR/job.log" \
      "$RESULT_URI/logs/$(date -u +%Y%m%dT%H%M%SZ).log" --quiet || true
  done
) &
UPLOADER=$!

cd "$WORK"
gcloud storage cp "$SOURCE_URI" source.tar.gz --quiet
printf '%s  source.tar.gz\n' "$EXPECTED_SHA" | sha256sum --check -
tar -xzf source.tar.gz
cp source.json "$RESULT_DIR/source.json"
if [[ -n "$RESUME_FROM" ]]; then
  gcloud storage cp "$RESUME_FROM/checkpoint.msgpack" "$WORK/resume.msgpack" --quiet
  gcloud storage cp "$RESUME_FROM/progress.json" "$WORK/resume-progress.json" --quiet
fi
ENTRYPOINT="$WORK/benchmark.py"
if [[ "$JOB_MODE" == month ]]; then
  metadata month-runner > "$WORK/month.py"
  RUNNER_SHA=$(metadata runner-sha256)
  printf '%s  %s\n' "$RUNNER_SHA" "$WORK/month.py" | sha256sum --check -
  cp "$WORK/month.py" "$RESULT_DIR/month.py"
  ENTRYPOINT="$WORK/month.py"
fi
lscpu > "$RESULT_DIR/lscpu.txt"
free -b > "$RESULT_DIR/memory-before.txt"

apt-get update
apt-get install -y --no-install-recommends python3-venv python3-pip
python3 -m venv "$WORK/venv"
PYTHON="$WORK/venv/bin/python"
"$PYTHON" -m pip install --upgrade -c "$WORK/constraints.txt" pip setuptools wheel
"$PYTHON" -m pip install -c "$WORK/constraints.txt" -e "$WORK/jax-gcm" 'jax==0.10.2'
"$PYTHON" -m pip check
"$PYTHON" -m pip freeze > "$RESULT_DIR/pip-freeze.txt"

# Use pip's CUDA libraries instead of a conflicting system LD_LIBRARY_PATH.
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu EXPECTED_BACKEND=cpu JAX_ENABLE_X64=0 OPENBLAS_NUM_THREADS=1 \
  timeout --signal=TERM --kill-after=30s 75m "$PYTHON" -u "$ENTRYPOINT"
