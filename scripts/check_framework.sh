#!/usr/bin/env bash
# CPU control-flow checks only. Run this inside a detached tmux session on RunPod.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p sampler_runs/setup/logs
uv run --no-project python -m unittest discover -s tests -v > sampler_runs/setup/logs/framework-check.log 2>&1
status=$?
printf '%s\n' "$status" > sampler_runs/setup/logs/framework-check.exit-code
exit "$status"
