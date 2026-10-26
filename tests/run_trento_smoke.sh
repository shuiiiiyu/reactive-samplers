#!/usr/bin/env bash
# Execute inside a Slurm GPU allocation after loading the configured modules.
set -euo pipefail
cd "$(dirname "$0")/.."
project="$PWD"
run="$project/sampler_runs/trento_smoke/${SLURM_JOB_ID:?Run this test through Slurm}"
mkdir -p "$run"
scratch=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/reactive-samplers.XXXXXX")
finish() {
    code=$?
    printf '%s\n' "$code" > "$run/job.exit"
    rm -rf -- "$scratch"
}
trap finish EXIT
export UV_CACHE_DIR="$scratch/uv-cache"
base_python=$(command -v python)
# Site modules inject an older torch through PYTHONPATH; use the locked runtime.
unset PYTHONPATH PYTHONHOME
env="$scratch/env"
archive="$project/.venv-trento/runtime.tar.gz"
if test -f "$archive"; then
    tar -xzf "$archive" -C "$scratch"
else
    uv venv --relocatable --python "$base_python" "$env"
    uv pip sync --python "$env/bin/python" --no-index \
        --find-links "$project/.venv-trento-wheels" tests/trento-requirements.lock
    uv pip install --python "$env/bin/python" --no-index \
        --find-links "$project/.venv-trento-wheels" --no-build-isolation -e "$project"
fi
# Prefer CUDA libraries bundled with the pinned torch wheels over site modules.
cuda_libs=$(find "$env/lib/python3.11/site-packages/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:' )
export LD_LIBRARY_PATH="${cuda_libs}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$project"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
nvidia-smi > "$run/nvidia-smi.txt"
"$env/bin/python" -c 'import json,sys,torch,diffusers,transformers; assert torch.cuda.is_available(); assert torch.__version__ == "2.5.1+cu121"; print(json.dumps(dict(python=sys.version,torch=torch.__version__,torch_path=torch.__file__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),diffusers=diffusers.__version__,transformers=transformers.__version__)))' > "$run/environment.json"
"$env/bin/python" -m unittest discover -s tests -v > "$run/unit-tests.log" 2>&1
"$env/bin/python" tests/smoke_generate.py --model both --output-dir "$run/images"
if ! test -f "$archive"; then
    mkdir -p "$(dirname "$archive")"
    tar -czf "$archive.partial" -C "$scratch" env
    mv "$archive.partial" "$archive"
    sha256sum "$archive" > "$archive.sha256"
fi
