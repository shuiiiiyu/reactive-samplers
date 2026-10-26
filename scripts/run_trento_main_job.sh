#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
config=$(realpath "${1:?experiment config JSON is required}")
scratch=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/reactive-sampling.XXXXXX")
finish() {
    code=$?
    printf '%s\n' "$code" > "${config%.json}.exit"
    rm -rf -- "$scratch"
}
trap finish EXIT
unset PYTHONPATH PYTHONHOME
tar -xzf .venv-trento/runtime.tar.gz -C "$scratch"
env="$scratch/env"
tar -xzf .venv-trento/metrics-overlay.tar.gz -C "$env/lib/python3.11/site-packages"
cuda_libs=$(find "$env/lib/python3.11/site-packages/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:')
export LD_LIBRARY_PATH="${cuda_libs}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
"$env/bin/python" scripts/run_config_json.py "$config"
if [[ "$config" == "$PWD/sampler_runs/mini20-20261026/frozen_holdout.json" ]]; then
    "$env/bin/python" scripts/compare_mini20.py
fi
