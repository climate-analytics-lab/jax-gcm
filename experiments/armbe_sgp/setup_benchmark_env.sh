#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
REPO_ROOT="$(readlink -f "$SCRIPT_DIR/../..")"
ENV_PATH="${1:-$REPO_ROOT/.benchmark-venv}"

if command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
else
  UV="$HOME/.local/bin/uv"
  if [[ ! -x "$UV" ]]; then
    if ! command -v curl >/dev/null 2>&1; then
      printf 'uv is unavailable and curl is not installed. Install uv from https://docs.astral.sh/uv/ and rerun.\n' >&2
      exit 1
    fi
    curl -LsSf https://astral.sh/uv/install.sh | sh
  fi
fi

cd "$REPO_ROOT"
"$UV" venv --python 3.12 "$ENV_PATH"
"$UV" pip install --python "$ENV_PATH/bin/python" --editable "$REPO_ROOT"
"$UV" pip install --python "$ENV_PATH/bin/python" --upgrade 'jax[cuda12]'

"$ENV_PATH/bin/python" -c 'import jax, jcm; print("Environment ready:", jax.__version__)'
printf 'Run the benchmark without activation:\n  %s/experiments/armbe_sgp/run_evolution_cost_benchmark.sh GPU_INDEX OUTPUT_TAG\n' "$REPO_ROOT"
