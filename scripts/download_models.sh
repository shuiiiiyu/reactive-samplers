#!/usr/bin/env bash
# Start inside tmux so downloads survive the SSH/desktop session.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p sampler_runs/setup/logs
uv run --no-project --with huggingface-hub==0.34.4 python scripts/download_models.py --model both \
  > sampler_runs/setup/logs/model-download.log 2>&1
status=$?
printf '%s\n' "$status" > sampler_runs/setup/logs/model-download.exit-code
exit "$status"
