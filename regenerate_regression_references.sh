#!/usr/bin/env bash
#
# Regenerate the composable-physics regression-test reference snapshots
# (jcm/data/test/composable_physics_regression/*.npz).
#
# WHAT THIS IS: a convenience wrapper that regenerates the references in a clean,
# CI-like environment (Python 3.11 + `pip install -r requirements.txt &&
# pip install -e .` in a throwaway prefix, as in .github/workflows/run_test.yaml).
#
# NOTE: the regression test now compares snapshots with a per-field normalized-RMS
# metric (3% tolerance) that is robust to cross-hardware / cross-jaxlib XLA
# reduction-order drift (measured <=~0.7% between runner CPU types). A reference is
# therefore valid regardless of the exact jax/jaxlib it was generated with — a
# plain `REGENERATE=1 JAX_PLATFORMS=cpu pytest ...` on any reasonable machine works
# too. This script just keeps regeneration reproducible and isolated from a messy
# dev env; it is no longer required to bit-match CI.
#
# Usage:  ./regenerate_regression_references.sh
# Then:   git add jcm/data/test/composable_physics_regression/*.npz && commit.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PREFIX="${REGEN_ENV_PREFIX:-/tmp/jcm-regen-ci}"
PYVER="${REGEN_PYVER:-3.11}"   # match .github/workflows/run_test.yaml

# Prefer micromamba/conda for a clean Python; fall back to `python${PYVER}`.
if command -v micromamba >/dev/null 2>&1; then
  micromamba create -y -p "$ENV_PREFIX" -c conda-forge "python=${PYVER}" pip >/dev/null
  PY="$ENV_PREFIX/bin/python"
elif command -v "python${PYVER}" >/dev/null 2>&1; then
  "python${PYVER}" -m venv "$ENV_PREFIX"
  PY="$ENV_PREFIX/bin/python"
else
  echo "Need micromamba or python${PYVER} to build a CI-matching env." >&2
  exit 1
fi

echo ">> Installing deps exactly as CI does (requirements.txt + -e .) ..."
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r "$REPO/requirements.txt"
"$PY" -m pip install --quiet -e "$REPO"
"$PY" -c "import jax, jaxlib; print(f'>> jax {jax.__version__} / jaxlib {jaxlib.__version__} (must match CI)')"

echo ">> Regenerating references ..."
cd "$REPO"
REGENERATE=1 JAX_PLATFORMS=cpu "$PY" -m pytest \
  jcm/physics/composable_physics_regression_test.py -m slow -p no:cacheprovider -q

echo ">> Verifying the fresh references pass (no REGENERATE) ..."
JAX_PLATFORMS=cpu "$PY" -m pytest \
  jcm/physics/composable_physics_regression_test.py -m slow -p no:cacheprovider -q

echo ">> Done. Review and commit jcm/data/test/composable_physics_regression/*.npz"
