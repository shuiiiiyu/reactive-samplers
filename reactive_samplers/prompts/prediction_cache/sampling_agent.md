# Method 3: final prediction cache (prediction_cache)

Sampling is paused BEFORE the current model prediction. `ctx.prediction.guided`
is None, and nfe counts only completed REAL predictions. Never fabricate a current
velocity, noise prediction, clean latent or deep feature. `inspect` reruns analysis
on the same saved cheap input feature without any model calls.

Edit run-local analysis.py, updater.py and relative Python helpers only. You may
repeat inspect within this pause. Return SkipAction(skip=bool): True reuses the
last real CFG-combined prediction; False calls the official model and refreshes
the cache. EVERY original grid interval still receives its official Euler update.
This differs from grid_skip, which removes integration nodes.

Default observation: sigma, step_delta_sigma, since_real_delta_sigma,
consecutive_skips, has_reference, feature_rel_change. Current input_feature is
SD3 pos_embed output or SDXL scaled-input conv_in output; reference_feature is
from the last REAL prediction, not the preceding reused node. Neither entry
feature includes timestep modulation; use sigma metadata. Compare tensors before
spatial/token reduction. There is no accumulated-path-change observation.

The first node and reuse-limit refreshes are forced real. Budget logic reserves
enough future refreshes; it can force reuse before exhausting NFE. These trusted
constraints, the fixed schedule and cache boundary cannot be modified by policy.
FEATURE_LAYERS must be empty: deep current features do not exist before forward.
Per-image policy memory resets; the framework privately owns prediction caches.

Read prior outcomes, failures and completed-image feedback. Keep base weights,
solver, CFG, precision, budget, resolution, metrics and data unchanged. Do not
read holdout data or borrow another method's strategy/session. Explain the signal,
code change or retention, expected effect and evidence that would refute it.
