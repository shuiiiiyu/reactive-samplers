#!/usr/bin/env bash
# Start this in tmux. Only official local-weight pipeline inference is tested here.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p sampler_runs/setup/logs
exec > sampler_runs/setup/logs/inference-smoke.log 2>&1
system_packages=$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')
export PYTHONPATH="$PWD:$system_packages"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Runtime environment is ignored; the versioned pyproject remains under tests/.
export UV_PROJECT_ENVIRONMENT="$PWD/.venv-inference"
uv run --project tests/inference_runtime python tests/smoke_generate.py "$@"
