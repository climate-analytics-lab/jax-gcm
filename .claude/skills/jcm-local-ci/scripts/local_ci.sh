#!/bin/bash
# Local jax-gcm CI: lint on this node, both test gates in one PBS job.
#   local_ci.sh [worktree]                 lint here, gates via qsub
#   local_ci.sh --local-fast [worktree]    also run the fast gate on this node
set -uo pipefail
LOCAL_FAST=0
if [ "${1:-}" = "--local-fast" ]; then LOCAL_FAST=1; shift; fi
REPO=$(cd "${1:-.}" && pwd)
VENV=${JCM_VENV:-$HOME/.venvs/jaxgcm}
ACCOUNT=${PBS_ACCOUNT:-UCSD0085}
source "$VENV/bin/activate"
cd "$REPO"

# Share XLA compiles across the pytest-xdist workers (and successive gate
# runs on this machine): bit-identical jitted modules hit the on-disk JAX
# cache instead of recompiling once per worker. Derecho nodes are
# homogeneous so a persistent dir is safe; a miss just recompiles. Same
# location as jcm.runners.maybe_enable_compilation_cache uses for runs.
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-${SCRATCH:-$HOME/.cache/jcm}/jcm-jax-cache}

# The dinosaur backend requires the semi-Lagrangian fork (neuralgcm/dinosaur
# PR #135); without it on the path every model-construction test raises and
# ~100 unrelated failures bury the ones that matter. The worktree comes first
# so it wins over any editable install in the venv.
export JCM_DINOSAUR=${JCM_DINOSAUR:-$HOME/dinosaur-sl}
export PYTHONPATH=$JCM_DINOSAUR:$REPO${PYTHONPATH:+:$PYTHONPATH}

echo "=== lint (here) ==="
ruff check . || { echo "LINT FAILED"; exit 1; }

# Runs before the qsub, never alongside it: pytest-cov erases and recombines
# .coverage.* per worktree, so two concurrent suites destroy each other's data.
LOCAL_FAST_STATUS=0
if [ "$LOCAL_FAST" = 1 ]; then
    echo "=== fast gate (this node, -n 2) ==="
    echo "WARNING: interactive work on a login node lives in a 10 GiB per-user"
    echo "         memory cgroup. -n 2 is the ceiling there, heavy packages"
    echo "         (jcm/physics/radiation/, the JAM tests) may still be"
    echo "         OOM-killed, and a killed worker looks like unrelated test"
    echo "         failures. The PBS gate below is the authoritative run."
    JAX_PLATFORMS=cpu pytest -n 2 -m "not slow" --cov=jcm --cov-fail-under=90 -q \
        || LOCAL_FAST_STATUS=$?
    echo "LOCAL_FAST_EXIT=$LOCAL_FAST_STATUS"
fi

echo "=== submitting both gates to the develop queue ==="
JOB=$(mktemp --suffix=.pbs)
cat > "$JOB" <<EOF
#!/bin/bash
#PBS -N jcm_ci
#PBS -A $ACCOUNT
#PBS -q develop
#PBS -l select=1:ncpus=16:mem=200GB
#PBS -l walltime=03:00:00
#PBS -m abe
#PBS -j oe
#PBS -o $REPO/jcm_ci.log
set -uo pipefail
source $VENV/bin/activate
cd $REPO
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR
export PYTHONPATH=$JCM_DINOSAUR:$REPO

# Sequential, not concurrent: the two gates share this worktree's .coverage.*.
# Each status is kept rather than left in \$? (the next echo would replace it),
# so the slow gate still runs after a fast-gate failure and the job's own exit
# code still reports it -- a green job must mean both gates passed.
FAST_STATUS=0
SLOW_STATUS=0

echo "=== fast gate (not slow, cov>=90) ==="
pytest -n 12 -m "not slow" --cov=jcm --cov-fail-under=90 -q || FAST_STATUS=\$?
echo "FAST_EXIT=\$FAST_STATUS"

echo "=== slow gate (slow only, cov>=80 vs .coveragerc-pr) ==="
pytest -n 4 -m "slow" --cov=jcm --cov-config=.coveragerc-pr --cov-fail-under=80 \
    || SLOW_STATUS=\$?
echo "SLOW_EXIT=\$SLOW_STATUS"

if [ "\$FAST_STATUS" -ne 0 ] || [ "\$SLOW_STATUS" -ne 0 ]; then
    echo "GATES FAILED (fast=\$FAST_STATUS slow=\$SLOW_STATUS)"
    exit 1
fi
echo "GATES PASSED"
EOF
qsub "$JOB"
echo "watch: grep -E 'FAST_EXIT|SLOW_EXIT|GATES' $REPO/jcm_ci.log"

# A failed local fast gate must not look like a clean wrapper run; the job is
# still submitted, since it is the authoritative gate.
if [ "$LOCAL_FAST_STATUS" -ne 0 ]; then
    echo "LOCAL FAST GATE FAILED (exit $LOCAL_FAST_STATUS) — job submitted anyway"
    exit "$LOCAL_FAST_STATUS"
fi
