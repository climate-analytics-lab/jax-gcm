#!/bin/bash
# Local jax-gcm CI: lint + fast gate here, slow gate via PBS develop queue.
#   local_ci.sh [worktree]   (default: current directory)
set -uo pipefail
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

echo "=== lint ==="
ruff check . || { echo "LINT FAILED"; exit 1; }

echo "=== fast gate (not slow, cov>=90) ==="
JAX_PLATFORMS=cpu pytest -n 12 -m "not slow" --cov=jcm --cov-fail-under=90 -q
FAST=$?
echo "FAST_EXIT=$FAST"

echo "=== slow gate: submitting to develop queue ==="
JOB=$(mktemp --suffix=.pbs)
cat > "$JOB" <<EOF
#!/bin/bash
#PBS -N jcm_slow_ci
#PBS -A $ACCOUNT
#PBS -q develop
#PBS -l select=1:ncpus=8:mem=120GB
#PBS -l walltime=02:00:00
#PBS -m abe
#PBS -j oe
#PBS -o $REPO/jcm_slow_ci.log
set -uo pipefail
source $VENV/bin/activate
cd $REPO
export JAX_PLATFORMS=cpu
export JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR
pytest -v -s -m "slow" --cov=jcm --cov-config=.coveragerc-pr --cov-fail-under=80 2>&1 | tail -40
echo SLOW_EXIT=\${PIPESTATUS[0]}
EOF
qsub "$JOB"
echo "watch: grep SLOW_EXIT $REPO/jcm_slow_ci.log"
exit $FAST
