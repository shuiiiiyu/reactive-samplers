# Reactive Samplers agent contract

You discover training-free timestep-selection rules while an SD3 image is being sampled.
The sampler is paused after its current denoiser prediction and before its latent update.
Read the request JSON, the current workspace programs, past decisions and completed-image feedback.

You may edit `analysis.py`, `updater.py` and relative-imported Python helpers inside the
workspace. Do not edit the model, sampler formula, framework, evaluation, data, seeds,
CFG, precision, max NFE or image resolution. Do not train any neural network.

At one check you may repeatedly edit analysis, execute the request's inspect_command,
read its results, and revise your hypothesis. You need not wait for the current image
or multiple images to finish. Then edit updater or keep it, and finish your turn.
Your bridge process returns only when this check is complete; sampling then resumes.
The bridge is responsible for reusing one provider/CLI conversation across the run.

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
- Put notes/temporary diagnostic scripts in `scratch/`, excluded from policy exports.
- For helpers use relative imports, e.g. `from .signals import ...`.

End with observations, edits/keep, prediction and what evidence would refute your prediction.
Keep diagnostic and deployed-rule cost in mind. HPS/CLIP feedback arrives after a completed
image, whereas trajectory feedback is available within the current image.

This is a trusted-code research workspace, not a security sandbox. Do not use evaluation
prompts to change rules. The final 40-image evaluation runs a hash-verified frozen policy
without an Agent. Do not claim speed or quality improvements from unmeasured results.
