# Reactive Samplers

Agent-in-the-loop discovery of training-free, adaptive diffusion sampling rules.
Code organization follows [OPAL](https://github.com/xingjian-bai/opal): a small wrapper,
an Agent session, editable programs, context, decision schedules and per-run records.

Three mechanisms are available: `grid_skip` (omit the current prediction and latent
update), `prediction_cache` (reuse the last final prediction), and `residual_cache`
(reuse the whole-backbone hidden residual with the current output head). All decide
before prediction. Cache methods update every original interval. See
[three-methods.md](docs/three-methods.md) for boundaries and observations.

## Directory layout

```text
reactive_samplers/           Our framework implementation
  wrapper.py                Shared analysis and dispatch
  grid_sampling.py          Current-node skip; no prediction or update when skipped
  cache_sampling.py         Prediction/residual caching; update every interval
  observations.py           Shared signals and English mathematical definitions
  decision_records.py       Four-part Agent reports and trusted execution outcomes
  agent.py                  Agent session and repeated within-step diagnostics
  programs.py               Strategy loading, versions, rejection and freezing
  context.py                Current state, references and SkipAction
  schedule.py               Every-step or fixed-checkpoint Agent intervention
  backends/                 Agent bridges
  models/                   SD3/SDXL adapters and cache boundaries
  prompts/
    <method>/              Per-method analysis.py, updater.py and sampling_agent.md
  evaluation.py             Quality scoring adapters
  report.py                 Event/results records
  cli.py                    inspect / freeze / report utilities
conf/                       Main-experiment Hydra configuration
scripts/                    Entry points, checks and resumable resource downloads
tests/                      CPU regression checks
models/                     Base SD3-Medium and SDXL weight directories
baselines/                  TPDM / AYS / OYS source, schedules and baseline weights
benchmarks/                 COCO / HPS / GenEval code, inputs and evaluator weights
metrics/                    CLIPScore / ImageReward; retained FID references
sampler_runs/               Generated run products and setup/download records
docs/                       Status and resource map
```

## Where are analysis.py and updater.py?

Initial versions live in `reactive_samplers/prompts/<method>/`. Each run starts with its own copies:

```text
sampler_runs/<run>/
  workspace/analysis.py      Agent edits this file during that image's sampling
  workspace/updater.py       Agent edits this file during that image's sampling
  programs/<source-hash>/    Complete retained strategy versions
  decisions/                State / request / Agent answer / timing
  agent_decisions/          English reports and actual executions, JSON + Markdown
  trajectories/             Actual sigma boundaries and strategy version per step
  images/
  config.json
  results.json
  frozen_policy/            Export for evaluation without an Agent
```

The initial updater computes every node. Sampling pauses BEFORE the current full
prediction. The Agent may edit analysis, run inspect on the saved state, and edit
or keep updater. It writes a reusable state-dependent rule, not a one-step toggle.
The bridge preserves the Agent conversation across the experiment.

Default observations are latent_rel_change, feature_rel_change, sigma,
step_delta_sigma, since_real_delta_sigma and consecutive_skips. Cache methods also
expose has_reference. Grid skipping compares current inputs with the last real
prediction's inputs; cache methods compare with the adjacent grid inputs. JSON
requests and inspect include English meanings, mathematical definitions and exact
reference coordinates. New signals need matching analysis.OBSERVATION_SPEC entries.
FEATURE_LAYERS is empty; no current full output is needed to decide whether to compute.

Every image is scored with CLIPScore and ImageReward after generation. New results,
paired reference deltas when available, and computation cost reach the next Agent
request. Reports and actual actions are archived in agent_decisions/ and shown in
the RunPod gallery. Scoring time is separate from sampling time.

All updaters return SkipAction(bool). They do not select sigma, scan future nodes,
train model weights, alter scheduler formulas, CFG or cache boundaries.

## Run and verify

```bash
uv run reactive-samplers --help
uv run python scripts/run_main.py --cfg job
python -m unittest discover -s tests -v
```

`conf/sd3_main.yaml` holds concrete SD3 defaults; `sd3_grid_skip.yaml` inherits them
and changes only the method. `conf/sdxl_main.yaml` inherits shared settings and sets
SDXL's model path, dtype, guidance and schedule length.

Change SD3's original step count at `reference_steps` in `conf/sd3_main.yaml`.
`max_nfe: ${reference_steps}` follows automatically; override max_nfe separately
to impose a smaller real-model-call budget. SD3 defaults to 28 steps, SDXL to 30.
Run these commands from the project root on the host holding the model weights:

```bash
# Print the resolved configuration without loading a model or calling an Agent.
uv run python scripts/run_main.py --cfg job --resolve reference_steps=40

# Run one of the three mechanisms; default mode has no Agent calls.
uv run python scripts/run_main.py method=prediction_cache reference_steps=40
uv run python scripts/run_main.py --config-name sdxl_main method=residual_cache

# Prepare a NEW paired COCO-10 experiment using the same YAML defaults.
uv run python scripts/prepare_coco10_grid.py --name coco10-next --runtime-root /path/to/runtime/project
```

Previously generated `sampler_runs/<experiment>/<arm>.json` files are fixed
experiment snapshots. Editing YAML does not change them. To change an existing,
not-yet-run JSON job, edit its `reference_steps` and `max_nfe` directly and choose
a fresh `output_dir`; completed experiment records retain their original settings.
Prompt JSONL rows require `id`, `prompt` and integer `seed`.

Run modes are default / explore / evaluate. Exploration requires an explicitly configured
Agent command; evaluation verifies a frozen manifest and makes no LLM calls.
Utility examples:

```bash
uv run reactive-samplers inspect STATE.pt --workspace sampler_runs/RUN/workspace
uv run reactive-samplers freeze sampler_runs/RUN/programs/HASH sampler_runs/RUN/frozen_policy
uv run reactive-samplers report sampler_runs/RUN
```

Use tmux for remote checks, downloads and inference that should survive disconnects.
Environment setup and one-off smoke tools have been retired. Use the existing
runtime for normal inference and CPU regression checks.

## Validation scope

CPU checks cover scheduler-interval parity, all three methods, cache behavior and configuration.
SD3 and SDXL each generated a valid 1024x1024 smoke image. These facts do not establish
full adaptive-Pipeline parity, baseline reproduction, quality preservation or acceleration.

CLIPScore and ImageReward are enabled by default; an explicitly empty metric list disables scoring.
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
benchmark datasets and environments are not stored in Git. The current sampler_runs/ records and outputs are tracked. The exact
resource manifest is docs/resource-lock.json; the download scripts restore those resources.
OYS official code has not been located. AYS's published schedule is included; its official
Quickstart and the OYS paper can be fetched with the resource downloader.

The two official-pipeline smoke images and measured run parameters are in docs/images/
and docs/smoke-results.json. They are inference-health examples, not acceleration results.
