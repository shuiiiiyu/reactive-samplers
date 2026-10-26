# Reactive Samplers

The user's instructions are the authority for this project. Follow the OPAL-inspired
layout below; do not introduce parent DeepResearch/SUE structure or workflow requirements.
Reference: https://github.com/xingjian-bai/opal

## Code and runtime boundaries
- `reactive_samplers/` owns the wrapper, Agent, program loader, context and schedules.
- `reactive_samplers/prompts/analysis.py` and `updater.py` are initial templates.
- An experiment copies them into `sampler_runs/<run>/workspace/`; the Agent edits those
  run-local copies, not the framework templates. Versioned programs and decisions live
  alongside workspace/, as in OPAL.
- Base SD3/SDXL weights live in models/. Baseline-specific sources/weights live in
  baselines/<method>/. Benchmarks keep their own code/data/checkpoints under
  benchmarks/<name>/. Shared metric resources live under metrics/<name>/.
- Logs, generated images and experiment/download records live in sampler_runs/.
  Python environments use hidden .venv* directories. Do not recreate artifacts/,
  preliminary_code/, preliminary_latex/ or SUE orchestration assets.
- Imported source/ directories are upstream references. Keep adaptations in our own
  project code and preserve upstream version metadata.

## Research contract
- Current method scope: SD3-Medium + official FlowMatch Euler; SDXL Base + official
  Euler supports fixed-grid skipping only. Continuous SDXL scheduling remains future work.
- Freeze base weights; do not train a timestep network or modify solver formulas, CFG,
  caching, data, evaluation or precision to improve method results.
- The Agent may repeatedly modify/run analysis and modify updater inside ONE image.
  Support both every-step checks and fixed checkpoints.
- Explore on 10 images; evaluate a frozen policy without LLM calls on 40 disjoint images.
  Do not invent benchmark records. GenEval is reserved for independent final evaluation.
- Reset per-image memory, retain strategy code across exploration images.
- Validate physical sigma actions and preserve scheduler semantics.
- Never claim quality preservation or acceleration from a smoke check.

## Development
Use uv. Run `bash scripts/check_environment.sh` inside tmux for the CPU checks.
GPU generation and paid Agent calls require a user request covering that work.
