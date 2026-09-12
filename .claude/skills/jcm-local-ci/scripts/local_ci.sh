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

# The gate's purpose is dependency PARITY with CI, so the dinosaur it tests
# against must be the one `pip install -e .` resolves — requirements.txt pins
# `dinosaur>=1.5.0`, which carries the semi-Lagrangian transport
# (neuralgcm/dinosaur#135) jcm's backend requires. A fork checkout is used
# ONLY when JCM_DINOSAUR is set explicitly; nothing is auto-detected, or a
# stale checkout sitting in $HOME would silently displace the pinned package
# and the gate would measure a dependency CI never sees.
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
if [ -n "${JCM_DINOSAUR:-}" ]; then
    [ -d "$JCM_DINOSAUR/dinosaur" ] || {
        echo "JCM_DINOSAUR=$JCM_DINOSAUR has no dinosaur package in it."
        exit 1
    }
    export PYTHONPATH=$JCM_DINOSAUR:$PYTHONPATH
fi
# Gate on the invariant, not on a path: without semi-Lagrangian transport
# every model-construction test raises and ~100 unrelated failures bury the
# ones that matter.
if ! python -c "
from dinosaur import primitive_equations as pe
import sys
sys.exit(0 if hasattr(pe, 'SemiLagrangianPrimitiveEquations') else 1)" 2>/dev/null; then
    echo "the dinosaur on this PYTHONPATH has no semi-Lagrangian transport."
    echo "Either install the pinned dependency:   pip install -e ."
    echo "  (requirements.txt pins dinosaur>=1.5.0, which carries it)"
    echo "or point at a checkout that has it:     JCM_DINOSAUR=<checkout>"
    exit 1
fi
if [ -n "${JCM_DINOSAUR:-}" ]; then
    echo "dinosaur: JCM_DINOSAUR override $JCM_DINOSAUR @ $(git -C "$JCM_DINOSAUR" rev-parse --short HEAD 2>/dev/null || echo non-git) — NOT the installed package, so this run is not at CI dependency parity"
else
    echo "dinosaur: installed package $(python -c 'import dinosaur,importlib.metadata as m;print(m.version("dinosaur"))' 2>/dev/null || echo '?') at $(python -c 'import dinosaur;print(dinosaur.__file__)')"
fi

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
export PYTHONPATH=$PYTHONPATH

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
