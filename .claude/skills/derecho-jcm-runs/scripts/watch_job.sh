#!/usr/bin/env bash
# watch_job.sh <jobid> <logfile> <completion-marker> [poll_seconds]
#
# Emits one line per event; exits 0 on completion, 1 on failure. Intended as
# the command of a persistent Monitor.
#
# Encodes four lessons from real runs:
#   * grep the log ONCE per check into a variable, then decide and report from
#     those same bytes (live NFS logs give stale re-reads -> phantom failures
#     that print empty)
#   * debounce: a failure signature must persist across two checks
#   * 3-strike qstat: PBS requeues and transient qstat errors are not death
#   * Lmod "unknown module" noise on these nodes is harmless
set -uo pipefail

JOB="${1:?usage: watch_job.sh <jobid> <logfile> <marker> [poll]}"
LOG="${2:?}"
MARKER="${3:?}"
POLL="${4:-300}"

FAIL_RE='Traceback|unhealthy|RESOURCE_EXHAUSTED|not in struct|Error executing|CUDA_ERROR|Killed|Invalid horizontal'
bad=0; gone=0

while true; do
  sleep "$POLL"

  if [ -f "$LOG" ] && grep -q "$MARKER" "$LOG"; then
    echo "COMPLETE: $JOB"
    grep -E "sim days/hr|NaN vars|SETTLED" "$LOG" | tail -4
    exit 0
  fi

  # single read: decide and report from the same bytes
  matches=$(grep -aiE "$FAIL_RE" "$LOG" 2>/dev/null | grep -av Lmod | tail -4)
  if [ -n "$matches" ]; then
    bad=$((bad + 1))
    if [ "$bad" -ge 2 ]; then
      echo "FAILED: $JOB (signature persisted 2 checks)"
      echo "$matches"
      exit 1
    fi
  else
    bad=0
  fi

  if qstat "$JOB" > /dev/null 2>&1; then
    gone=0
  else
    gone=$((gone + 1))
    if [ "$gone" -ge 3 ]; then
      if grep -q "$MARKER" "$LOG" 2>/dev/null; then
        echo "COMPLETE (late log): $JOB"; exit 0
      fi
      echo "GONE: $JOB left the queue without completing"
      tail -4 "$LOG" 2>/dev/null
      exit 1
    fi
  fi
done
