#!/usr/bin/env bash
#
# Regenerate the composable-physics regression-test reference snapshots
# (jcm/data/test/composable_physics_regression/*.npz).
#
# WHY A SCRIPT (and not just `REGENERATE=1 pytest`): those references are
# compared with a tight tolerance (rtol=1e-3, atol=1e-4) that absorbs ULP-level
# reduction-order drift but NOT a change of jax/jaxlib version — a different
# jaxlib emits different XLA CPU code and the 1-day integration amplifies the
# difference past the tolerance. So a reference is only valid against CI if it is
# generated with the SAME jax/jaxlib (and Python) that CI installs. CI runs
# Python 3.11 and `pip install -r requirements.txt && pip install -e .` on
# ubuntu-latest (see .github/workflows/run_test.yaml), which currently resolves
# jaxlib 0.10.2. This script reproduces that environment in a throwaway prefix so
# the regenerated references match CI.
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
