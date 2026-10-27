# Method 4: whole-backbone residual cache (residual_cache)

Sampling is paused BEFORE the current backbone prediction. ctx.prediction.guided
is None. Return SkipAction(skip=bool): False runs the official full forward and
caches R = Z - H; True uses CURRENT H + cached R, then the current output head.
This is a hidden-feature residual, NOT a final velocity/noise residual.

SD3: H is pos_embed output; Z is the input to norm_out after all transformer
blocks. Reuse still executes time_text_embed, norm_out, proj_out and unpatchify.
SDXL: H is conv_in output after official input scaling; Z precedes conv_norm_out
after the entire down/mid/up U-Net. Reuse executes norm/activation/conv_out. SDXL's
cached delta retains OLD time/text effects; its output head is unconditioned.
Do not claim this is the full TeaCache method or a proven SDXL quality result.

Every original grid node still gets a Scheduler update. NFE counts full backbone
refreshes; output-head calls and entry-feature costs are reported separately.

Edit run-local analysis.py, updater.py and relative Python helpers only. Inspect
can be repeated at the same pause using saved current input_feature and historical
reference_feature; it cannot run the current model. Default observation consists
of sigma, step_delta_sigma, since_real_delta_sigma, consecutive_skips,
has_reference and feature_rel_change. Compare full feature tensors before scalar
reduction. These entry features are not timestep-modulated. FEATURE_LAYERS must
remain empty; no current deep features or model prediction exists at decision time.

First-node initialization, maximum consecutive reuse and prediction-budget guards
belong to the framework. Cache state resets per image and remains independent of
policy memory/recovery. Keep solver, weights, CFG, precision, budget, resolution,
data, metrics and cache boundaries unchanged. Never read holdout data or borrow
another method's strategy/session. Read past outcomes, failures and image feedback;
explain changes or retention, expected effect and what evidence would refute it.
