#!/usr/bin/env bash
set -euo pipefail
project=/gpfs/scratch/ehpc1056/TengLong/deepresearch/workspace/reactive-samplers
run="$project/sampler_runs/agent-cli-smoke-20261026-r4"
setup="$project/sampler_runs/setup/agent-cli-smoke-20261026"
job_log="$setup/job-${SLURM_JOB_ID:?}"
mkdir -p "$job_log"
scratch=$(mktemp -d "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/reactive-agent-smoke.XXXXXX")
finish() { code=$?; printf '%s\n' "$code" > "$job_log/job.exit"; rm -rf -- "$scratch"; }
trap finish EXIT
unset PYTHONPATH PYTHONHOME
tar -xzf "$project/.venv-trento/runtime.tar.gz" -C "$scratch"
env="$scratch/env"
uv pip install --python "$env/bin/python" --no-index --find-links "$project/.venv-trento-wheels" 'hydra-core>=1.3,<1.4'
cuda_libs=$(find "$env/lib/python3.11/site-packages/nvidia" -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:' )
export LD_LIBRARY_PATH="${cuda_libs}${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$project" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
nvidia-smi > "$job_log/nvidia-smi.txt"
"$env/bin/python" -c 'import json,torch,diffusers,hydra; assert torch.cuda.is_available(); assert torch.__version__ == "2.5.1+cu121"; print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),diffusers=diffusers.__version__,hydra=hydra.__version__)))' > "$job_log/environment.json"
"$env/bin/python" "$project/scripts/run_main.py" \
  mode=explore model_path="$project/models/sd3-medium" device=cuda dtype=bfloat16 \
  height=1024 width=1024 guidance_scale=7.0 max_sequence_length=256 \
  reference_steps=28 max_nfe=28 prompts="$setup/prompts.jsonl" expected_count=1 \
  output_dir="$run" agent.backend=mailbox agent.timeout_seconds=1800 agent.poll_seconds=1 \
  agent.schedule=checkpoints 'agent.checkpoints=[0]' metrics.names='[]'
