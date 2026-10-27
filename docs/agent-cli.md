# RunPod Codex CLI bridge

Exploration uses the OPAL-style coding-agent CLI, not a raw model API. Trento pauses
at Agent decision points; the RunPod bridge resumes one Codex conversation per run.
The current experiment explicitly uses `danger-full-access`. Workspace restrictions
are the research contract and mailbox validation, not OS sandbox isolation. The bridge transfers only the JSON request/feedback and policy Python files;
`state.pt`, latent history, images and model weights stay on Trento. Repeated
inspection requests are served by the paused Trento process against its in-memory
state and an isolated diagnostic workspace.

## Existing RunPod runtime

Use the already installed Codex CLI and matching `codex-code-mode-host` under
`$HOME/.codex-opal/reactive-samplers-tools/bin/`. Existing authentication remains
under `$HOME/.codex-opal`. Installation tooling has been intentionally removed.
The per-run bridge below is still required for Agent exploration.

## Per-run bridge

Copy the two small bridge scripts from the persistent Trento project to RunPod:

```bash
project=/gpfs/scratch/ehpc1056/TengLong/deepresearch/workspace/reactive-samplers
mkdir -p $HOME/.codex-opal/reactive-samplers-tools/scripts
ssh trento "cat '$project/scripts/runpod_codex_bridge.py'" > $HOME/.codex-opal/reactive-samplers-tools/scripts/runpod_codex_bridge.py
ssh trento "cat '$project/scripts/agent_inspect_helper.py'" > $HOME/.codex-opal/reactive-samplers-tools/scripts/agent_inspect_helper.py
```

Start one bridge for the exploration output directory **before** submitting its
Slurm job. Replace `<run-name>` with the configured unique output directory name.
Run it inside a tmux session so it stays attached while the local SSH window closes.

```bash
project=/gpfs/scratch/ehpc1056/TengLong/deepresearch/workspace/reactive-samplers
run="$project/sampler_runs/<run-name>"
tmux new -s reactive-samplers-agent "CODEX_HOME='$HOME/.codex-opal' python3 $HOME/.codex-opal/reactive-samplers-tools/scripts/runpod_codex_bridge.py --project-root '$project' --run-dir '$run'"
```

Detach with `Ctrl-B`, then `D`; keep this bridge running for the whole exploration. Configure `mode: explore`,
`agent.backend: mailbox`, and the desired `every_step` or `checkpoints` schedule in
the Slurm job's resolved Hydra configuration. Do not start two bridges for the same
run. CLI failures and timeouts are returned to Trento, where the sampler retains its
previous valid policy and continues.
