#!/usr/bin/env bash
# Behaviour test for the run-completion gate that mkrun.py generates.
#
# This extracts the gate straight out of the generated manifest and runs it
# under the SAME `set -euo pipefail` the container uses. That matters: an
# earlier version of this test reimplemented the gate in a plain function, so
# it passed while the real script aborted at the first no-match `grep` —
# pipefail propagates the 1 out of a command substitution and `set -e` kills
# the script before the empty-LAST branch can run. Test the artefact, under
# the artefact's shell options, or the test is theatre.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${JCM_PYTHON:-python}"
DAYS=365
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
RUNDIR="$TMP/runs/testrun"
mkdir -p "$RUNDIR"

# Extract the generated script, then keep only the gate (everything from the
# attempt-slice onward) and repoint it at the temp rundir.
"$PY" "$HERE/mkrun.py" --name testrun --days "$DAYS" 2>/dev/null \
  | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["spec"]["template"]["spec"]["containers"][0]["command"][-1])' \
  > "$TMP/full.sh"
sed -n '/^tail -c +\$((ATTEMPT_START/,$p' "$TMP/full.sh" \
  | sed "s#/runs/testrun#$RUNDIR#g" > "$TMP/gate.sh"
[ -s "$TMP/gate.sh" ] || { echo "FAILED to extract gate from generated script"; exit 1; }

fails=0
run_gate() {  # $1 = bytes already in run.log before this attempt
  ( set -euo pipefail
    ATTEMPT_START="$1"
    RC=0
    # shellcheck disable=SC1090
    source "$TMP/gate.sh" ) >"$TMP/out" 2>&1
  echo $?
}

check() {  # name want_rc got_rc
  if [ "$2" = "$3" ]; then
    printf '  PASS  %-46s (rc=%s)\n' "$1" "$3"
  else
    printf '  FAIL  %-46s want rc=%s got rc=%s\n' "$1" "$2" "$3"
    sed 's/^/          /' "$TMP/out"
    fails=$((fails + 1))
  fi
}

L="$RUNDIR/run.log"

# 1. Genuine completion, then the pod restarts and re-runs the container.
#    THIS is the case codex caught: no _dayN.nc in the new slice.
printf 'Saved predictions to run_day365.nc\n' > "$L"
off=$(stat -c%s "$L")
printf 'Resumed from checkpoint %s/testrun.ckpt at sim-day 365.0\n' "$RUNDIR" >> "$L"
check "restart after genuine completion" 0 "$(run_gate "$off")"

# 2. No-op resume: stale/foreign checkpoint behind the target, no progress.
printf 'Saved predictions to run_day365.nc\n' > "$L"
off=$(stat -c%s "$L")
printf 'Resumed from checkpoint %s/testrun.ckpt at sim-day 6.0\n' "$RUNDIR" >> "$L"
check "no-op resume from stale checkpoint" 1 "$(run_gate "$off")"

# 3. Attempt produced nothing at all and never even logged a resume.
printf 'Saved predictions to run_day365.nc\n' > "$L"
off=$(stat -c%s "$L")
printf 'some unrelated chatter\n' >> "$L"
check "no output and no resume line" 1 "$(run_gate "$off")"

# 4. Normal completion within this attempt.
printf 'Resumed at sim-day 300.0\nSaved predictions to run_day365.nc\n' > "$L"
check "normal completion" 0 "$(run_gate 0)"

# 5. Evicted mid-year.
printf 'Saved predictions to run_day120.nc\n' > "$L"
check "evicted mid-year is incomplete" 1 "$(run_gate 0)"

# 6. Health trip in THIS attempt.
printf 'Saved predictions to run_day365.nc\natmosphere unhealthy\n' > "$L"
check "unhealthy this attempt" 1 "$(run_gate 0)"

# 7. Health trip in a PREVIOUS attempt, recovered since.
printf 'NaN vars: 3/176\n' > "$L"
off=$(stat -c%s "$L")
printf 'Saved predictions to run_day365.nc\n' >> "$L"
check "recovered-from NaN in earlier attempt" 0 "$(run_gate "$off")"

if [ "$fails" -eq 0 ]; then echo "all gate tests passed"; else
  echo "$fails gate test(s) failed"; exit 1; fi
