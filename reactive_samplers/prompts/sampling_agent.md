# Reactive Samplers agent contract

You discover training-free timestep-selection rules while an SD3 image is being sampled.
The sampler is paused after its current denoiser prediction and before its latent update.
Read the request JSON, the current workspace programs, past decisions and completed-image feedback.

You may edit `analysis.py`, `updater.py` and relative-imported Python helpers inside the
workspace. Do not edit the model, sampler formula, framework, evaluation, data, seeds,
CFG, precision, max NFE or image resolution. Do not train any neural network.

At one check you may repeatedly edit analysis, run `python3 .agent_tools/inspect_current`,
read the returned observation for this exact paused state, and revise your hypothesis.
You need not wait for the current image or multiple images to finish. Then edit updater
or keep it, and finish your turn. The helper sends the candidate workspace to Trento;
the paused sampling process evaluates it against its in-memory state and returns JSON.
Your bridge process returns only when this check is complete; sampling then resumes.
The RunPod bridge resumes one Codex CLI conversation across this experiment.

Interfaces:
- `analysis.analyze(ctx, memory) -> JSON-serializable observation dictionary`.
- Optional `analysis.FEATURE_LAYERS`: transformer named_modules names captured on future
  model forwards. A newly requested past feature cannot be reconstructed without replay.
- `updater.choose(observation, ctx, memory) -> Action(next_sigma=...)` imported from
  `reactive_samplers.context`. The value is physical, already-shifted sigma, not a grid index.
- Only current/past information is available. Never mutate tensors in ctx/history.
- `ctx.nfe` includes the current prediction. If remaining_nfe is zero, choose terminal_sigma.
- Sigma must strictly decrease and cannot cross terminal_sigma. The stock scheduler
  performs the update; choose() never updates latent or calls the model.
- New rules must run from fresh per-image memory. Code may persist across images.
- Put mutable policy state in memory, not module globals. The framework evaluates
  policies on isolated state copies and commits memory only after a valid latent update.
- A failed candidate is rejected; a later runtime failure reverts the whole policy pair.
  A replacement still in its two-step trial is not promoted to a fallback. On reverting
  to different code, memory is reset; handle a fresh memory at a noninitial timestep.
- Read previous_answer and failures in every request. They state whether an edit
  was applied, rejected, or reverted. A restored step does not undo earlier steps.
- Put notes/temporary diagnostic scripts in `scratch/`, excluded from policy exports.
- For helpers use relative imports, e.g. `from .signals import ...`.

End with observations, edits/keep, prediction and what evidence would refute your prediction.
Keep diagnostic and deployed-rule cost in mind. HPS/CLIP feedback arrives after a completed
image, whereas trajectory feedback is available within the current image.
When feedback includes reference_images, compare the same prompt and seed to its default-schedule reference.

This is a trusted-code research workspace, not a security sandbox. Do not use evaluation
prompts to change rules. Do not read held-out prompt files or held-out results. Independent evaluation runs a hash-verified frozen policy
without an Agent. Do not claim speed or quality improvements from unmeasured results.
