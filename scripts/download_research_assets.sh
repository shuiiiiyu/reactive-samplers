#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p sampler_runs/setup/logs
uv run --no-project --with huggingface-hub==0.34.4 python scripts/download_research_assets.py \
  > sampler_runs/setup/logs/research-assets-download.log 2>&1
status=$?
printf '%s\n' "$status" > sampler_runs/setup/logs/research-assets-download.exit-code
exit "$status"
