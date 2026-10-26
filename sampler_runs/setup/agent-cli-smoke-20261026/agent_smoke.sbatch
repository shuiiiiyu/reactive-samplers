#!/usr/bin/env bash
#SBATCH --job-name=reactive-agent-smoke
#SBATCH --account=ehpc1056
#SBATCH --partition=acc
#SBATCH --qos=acc_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/gpfs/scratch/ehpc1056/TengLong/deepresearch/workspace/reactive-samplers/sampler_runs/setup/agent-cli-smoke-20261026/%j.out
#SBATCH --error=/gpfs/scratch/ehpc1056/TengLong/deepresearch/workspace/reactive-samplers/sampler_runs/setup/agent-cli-smoke-20261026/%j.err
set -euo pipefail
module load gcc/11.4.0 mkl/2024.0 impi/2021.11 hdf5/1.14.1-2-gcc python/3.11.5-gcc nvidia-hpc-sdk/23.11-cuda11.8 openblas/0.3.27-gcc cudnn/9.0.0-cuda11 tensorrt/10.0.0-cuda11 nccl/2.19.4 pytorch/2.4.0 uv/0.10.11
export SLURM_CPU_BIND=none
bash /gpfs/scratch/ehpc1056/TengLong/deepresearch/workspace/reactive-samplers/sampler_runs/setup/agent-cli-smoke-20261026/run_agent_smoke.sh
