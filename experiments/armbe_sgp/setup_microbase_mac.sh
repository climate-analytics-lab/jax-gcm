#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
venv="$root/.venv-microbase"

if [[ ! -f "$root/microbase_campaign.toml" ]]; then
  cp "$root/microbase_campaign.example.toml" "$root/microbase_campaign.toml"
  echo "created microbase_campaign.toml from the example; inspect it before production"
fi

if [[ $(uname -s) != Darwin || $(uname -m) != arm64 ]]; then
  echo "warning: this bundle is validated for Apple Silicon macOS" >&2
fi

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 "$venv"
  uv pip sync --python "$venv/bin/python" "$root/requirements-microbase-mac.txt"
elif command -v python3.12 >/dev/null 2>&1; then
  python3.12 -m venv "$venv"
  "$venv/bin/python" -m pip install --upgrade pip
  "$venv/bin/python" -m pip install -r "$root/requirements-microbase-mac.txt"
else
  echo "Python 3.12 is required. Install uv (recommended) or Python 3.12, then rerun." >&2
  exit 1
fi

"$venv/bin/python" "$root/microbase_mac_pipeline.py" doctor \
  --config "$root/microbase_campaign.toml"

echo "environment ready: $venv"
