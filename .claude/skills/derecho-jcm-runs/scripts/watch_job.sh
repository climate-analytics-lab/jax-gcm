#!/usr/bin/env bash
# watch_job.sh <jobid> <logfile> <completion-marker> [poll_seconds]
#
# Emits one line per event; exits 0 on completion, 1 on failure. Intended as
# the command of a persistent Monitor.
#
# Encodes five lessons from real runs:
#   * read the log ONCE per check into a variable, then decide and report from
#     those same bytes (live NFS logs give stale re-reads -> phantom failures
#     that print empty)
#   * THE MARKER IS NOT SUCCESS: jcm.runners.run_chunked logs "unhealthy" and
#     returns NORMALLY when the health gate trips, so the job script goes on to
#     write its completion marker. Failure signatures are therefore checked
#     BEFORE the marker is accepted, and the NaN count is verified to be zero.
#   * debounce: a failure signature must persist across two checks -- but only
#     while the job is still running; once the marker is down the log is final
#     and there is nothing to debounce against
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

  # ONE read per check; every decision below comes from these same bytes.
  content=$(cat "$LOG" 2>/dev/null || true)
  matches=$(printf '%s\n' "$content" | grep -aiE "$FAIL_RE" | grep -av Lmod | tail -4)
  # Any "NaN vars: N/M" with N != 0 is a failed run even if nothing matched
  # FAIL_RE (the health line is printed per chunk, before the gate reacts).
  nanbad=$(printf '%s\n' "$content" \
           | grep -aoE "NaN vars:[[:space:]]*[0-9]+" \
           | grep -avE "NaN vars:[[:space:]]*0$" | tail -2)

  if printf '%s\n' "$content" | grep -q "$MARKER"; then
    # Marker present: the log is final, so judge it now with no debounce.
    if [ -n "$matches" ] || [ -n "$nanbad" ]; then
      echo "FAILED: $JOB (completion marker present, but the run is unhealthy)"
      [ -n "$matches" ] && printf '%s\n' "$matches"
      [ -n "$nanbad" ] && printf '%s\n' "$nanbad"
      exit 1
    fi
    echo "COMPLETE: $JOB"
    printf '%s\n' "$content" | grep -E "sim days/hr|NaN vars|SETTLED" | tail -4
    exit 0
  fi

  if [ -n "$matches" ] || [ -n "$nanbad" ]; then
    bad=$((bad + 1))
    if [ "$bad" -ge 2 ]; then
      echo "FAILED: $JOB (signature persisted 2 checks)"
      [ -n "$matches" ] && printf '%s\n' "$matches"
      [ -n "$nanbad" ] && printf '%s\n' "$nanbad"
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
      # Same rule as above: a late marker still has to be a HEALTHY one.
      if printf '%s\n' "$content" | grep -q "$MARKER"; then
        if [ -n "$matches" ] || [ -n "$nanbad" ]; then
          echo "FAILED (late log): $JOB — marker present but run unhealthy"
          exit 1
        fi
        echo "COMPLETE (late log): $JOB"; exit 0
      fi
      echo "GONE: $JOB left the queue without completing"
      tail -4 "$LOG" 2>/dev/null
      exit 1
    fi
  fi
done
