# Reactive Samplers

Agent-in-the-loop discovery of training-free, adaptive diffusion sampling rules.
Code organization follows [OPAL](https://github.com/xingjian-bai/opal): a small wrapper,
an Agent session, editable programs, context, decision schedules and per-run records.

## Directory layout

```text
reactive_samplers/           Our framework implementation
  wrapper.py                Sampling / Agent check / reload / action / original solver update
  agent.py                  Agent session and repeated within-step diagnostics
  programs.py               Strategy loading, versions, rejection and freezing
  context.py                Current state, past trajectory, observation and sigma action
  schedule.py               Every-step or fixed-checkpoint Agent intervention
  backends/                 Agent bridges
  models/                   Model adapter CODE (currently adaptive SD3)
  prompts/
    sampling_agent.md       Agent instructions
    analysis.py             Initial observation template
    updater.py              Initial timestep-selection template
  evaluation.py             Quality scoring adapters
  report.py                 Event/results records
  cli.py                    inspect / freeze / report utilities
conf/                       Main-experiment Hydra configuration
scripts/                    Entry points, checks and resumable resource downloads
tests/                      CPU checks and opt-in two-model GPU smoke script
models/                     Base SD3-Medium and SDXL weight directories
baselines/                  TPDM / AYS / OYS source, schedules and baseline weights
benchmarks/                 COCO / HPS / GenEval code, inputs and evaluator weights
metrics/                    Shared CLIP / CLIPScore / FID code and checkpoints
experiments/main/           Exploration/evaluation manifest contract
sampler_runs/               Generated run products and setup/download records
docs/                       Status and resource map
```

## Where are analysis.py and updater.py?

Initial versions are `reactive_samplers/prompts/analysis.py` and
`reactive_samplers/prompts/updater.py`. Each run starts with its own copies:

```text
sampler_runs/<run>/
  workspace/analysis.py      Agent edits this file during that image's sampling
  workspace/updater.py       Agent edits this file during that image's sampling
  programs/<source-hash>/    Complete retained strategy versions
  decisions/                State / request / Agent answer / timing
  trajectories/             Actual sigma boundaries and strategy version per step
  images/
  config.json
  results.json
  frozen_policy/            Export for evaluation without an Agent
```

The initial updater follows the official reference schedule. A check pauses sampling
after the current model prediction and before its latent update. The Agent may repeatedly
edit analysis, run `inspect`, view the diagnostic and then edit updater before returning.
It does not need to wait for a finished image or a batch. The Agent bridge must preserve
its provider/CLI conversation across the run. The real provider bridge is not configured yet.

Observation fields are open-ended. `FEATURE_LAYERS` requests captures on future model
forwards; uncaptured past features require explicit replay. `HISTORY_LIMIT` controls
history retained in frozen inference. Exploration keeps the full existing trajectory.
Rules receive only current/past information and must not mutate model state.

Actions choose the next physical sigma after scheduler shifting. The adapter calls the
unchanged official FlowMatch Euler update. It never reapplies shift to an action.
No model training, learned timestep network, CFG tuning or cache optimization is included.

## Run and verify

```bash
uv sync
uv run reactive-samplers --help
uv run python scripts/run_main.py --cfg job
bash scripts/check_environment.sh
```

`conf/sd3_main.yaml` is the main-experiment template. `???` means a required experimental
setting; it must be supplied before a real run. The downloaded local SD3 model path is
already filled. Prompt JSONL rows require `id`, `prompt` and integer `seed`.
The initial research plan uses 10 exploration images and 40 disjoint evaluation images.

Run modes are default / explore / evaluate. Exploration requires an explicitly configured
Agent command; evaluation verifies a frozen manifest and makes no LLM calls.
Utility examples:

```bash
uv run reactive-samplers inspect STATE.pt --workspace sampler_runs/RUN/workspace
uv run reactive-samplers freeze sampler_runs/RUN/programs/HASH sampler_runs/RUN/frozen_policy
uv run reactive-samplers report sampler_runs/RUN
```

Use tmux for remote checks, downloads and inference that should survive disconnects.
The inference smoke is opt-in, separate from unittest discovery:

```bash
bash tests/run_inference_smoke.sh --output-dir sampler_runs/new_smoke
```

It runs the official SD3 and SDXL pipelines with local weights, one image each.
The test uses CPU offload for the server's 16 GB GPU. Its isolated environment lives
in `.venv-inference`; the launcher reuses server PyTorch rather than installing another
CUDA stack. All model components remain present.

## Validation scope

Eleven CPU checks pass, including exact tested scheduler-interval parity at shift 1 and 3.
SD3 and SDXL each generated a valid 1024x1024 smoke image. These facts do not establish
full adaptive-Pipeline parity, baseline reproduction, quality preservation or acceleration.
The full SD3 Pipeline parity harness is `scripts/check_sd3_parity.py` and remains unrun.

Quality scoring is explicit; empty metric configuration means no quality measurement.
NFE counts denoiser calls and denoiser_items records CFG batch items. Sampling time includes
diagnostics and Agent waits. End-to-end generation time adds text encoding and VAE decoding
but excludes model loading, image saving and scoring. Compare frozen/no-Agent timings when
making deployment claims. Generated strategies are trusted Python, not a security sandbox.

See docs/RESOURCES.md for downloaded sources, data, checkpoints and remaining gaps.

## Clone this repository

```bash
git clone --recurse-submodules https://github.com/vibe-hust/reactive-samplers.git
cd reactive-samplers
```

Downloaded third-party source trees are pinned Git submodules. Base/baseline model weights,
benchmark datasets, environments and private run records are not stored in Git. The exact
resource manifest is docs/resource-lock.json; the download scripts restore those resources.
OYS official code has not been located. AYS's published schedule is included; its official
Quickstart and the OYS paper can be fetched with the resource downloader.

The two official-pipeline smoke images and measured run parameters are in docs/images/
and docs/smoke-results.json. They are inference-health examples, not acceleration results.
