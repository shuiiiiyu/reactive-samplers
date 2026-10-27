# Reactive Samplers

The user's instructions are the authority for this project. Follow the OPAL-inspired
layout below; do not introduce parent DeepResearch/SUE structure or workflow requirements.
Reference: https://github.com/xingjian-bai/opal

## Code and runtime boundaries
- `reactive_samplers/` owns the wrapper, Agent, program loader, context and schedules.
- `reactive_samplers/prompts/<method>/analysis.py` and `updater.py` are initial templates.
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
- Three methods: grid_skip, prediction_cache and residual_cache.
  All decide before the current full prediction. grid_skip omits the current
  prediction and latent update; cache methods update every original interval.
  SD3-Medium uses official FlowMatch Euler; SDXL Base uses official epsilon Euler.
- Freeze base weights; do not train a timestep network or modify solver formulas, CFG,
  data, evaluation or precision to improve method results. Cache methods use only the
  framework-owned final-prediction or whole-backbone-residual cache; Agent code must
  not alter the cache boundary, CFG or cache payload to improve results.
- The Agent may repeatedly modify/run analysis and modify updater inside ONE image.
  Support both every-step checks and fixed checkpoints.
- Explore on 10 images; evaluate a frozen policy without LLM calls on 40 disjoint images.
  Do not invent benchmark records. GenEval is reserved for independent final evaluation.
- Reset per-image memory, retain strategy code across exploration images.
- Updaters return SkipAction(bool); they never choose sigma or scan future nodes.
- Default observations and their English mathematical definitions come from
  observations.py. Grid skipping compares with last-real inputs; cache methods
  compare with adjacent-grid inputs. Definitions follow any Agent-added signals.
- Every completed image supplies CLIPScore and ImageReward feedback. Every Agent
  call writes its four-part English report and actual execution records into
  agent_decisions/, which the RunPod gallery reads automatically.
- Never claim quality preservation or acceleration from a smoke check.

## Development
Use the existing Python environment. Run `python -m unittest discover -s tests -v`
inside tmux for CPU regression checks. Environment setup and smoke-test scripts
were intentionally removed; do not recreate them unless requested.
GPU generation and paid Agent calls require a user request covering that work.
