# Trento inference smoke

The pinned Python 3.11 runtime was built and tested on Trento, with torch
2.5.1+cu121, diffusers 0.35.1 and transformers 4.44.2. No Agent is used.

Submit from the project root, using the generated site-specific Slurm launcher:

```bash
sbatch sampler_runs/setup/trento/smoke.sbatch
```

`tests/run_trento_smoke.sh` runs inside that allocation after the configured
modules are loaded. It restores `.venv-trento/runtime.tar.gz` into node-local
temporary storage, then runs the existing unit tests and official SD3/SDXL smoke.
If the archive is missing it builds from the pinned offline wheels first.
Dependencies are recorded in `trento-requirements.in` and `.lock` in this directory.
Outputs are unique per Slurm job under `sampler_runs/trento_smoke/<job_id>/`.

Successful initial job: 48036668 (COMPLETED, exit 0:0). See its REPORT.md.
This verifies inference health, not the adaptive method or acceleration claims.
