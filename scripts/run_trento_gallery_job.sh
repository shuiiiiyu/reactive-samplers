#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
request=$(realpath "${1:?request.json is required}")
scratch=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/reactive-gallery.XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
unset PYTHONPATH PYTHONHOME
tar -xzf .venv-trento/runtime.tar.gz -C "$scratch"
env="$scratch/env"
cuda_libs=$(find "$env/lib/python3.11/site-packages/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:')
export LD_LIBRARY_PATH="${cuda_libs}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
"$env/bin/python" -m reactive_samplers.generation --request "$request"
