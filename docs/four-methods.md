# Four sampling mechanisms

| ID | method | Decision position | Reuse/step mechanism | Implementation |
|---|---|---|---|---|
| 1 | adaptive_sigma | after current prediction | choose arbitrary next physical sigma | wrapper.sample_adaptive_sigma |
| 2 | grid_skip | after current prediction | keep/skip later grid nodes, Euler across removed nodes | wrapper.sample_grid_skip |
| 3 | prediction_cache | before current prediction | last real final CFG-combined prediction, update every node | cache_sampling.sample_prediction_cache |
| 4 | residual_cache | before current backbone | current entry + last whole-backbone delta, current head, update every node | cache_sampling.sample_residual_cache |

Methods 1/2 keep their previous execution semantics and policy contracts. SDXL
supports 2/3/4; only SD3 supports arbitrary physical-sigma actions (method 1).
Methods 3/4 have separate starter programs, frozen method tags and Agent contracts.
Agent edits code; Python chooses reuse at sampling time. The initial templates
keep all nodes real; no unmeasured reuse threshold is presented as an improvement.

## Cache lifecycle and observation

For each image, initialize empty caches. At a grid node, extract the cheap entry
feature and pause BEFORE any full prediction. Expose six scalar observation fields:
sigma, step_delta_sigma, since_real_delta_sigma, consecutive_skips, has_reference,
feature_rel_change. Distance uses the full current/reference feature tensors before
scalar reduction. Reference is the last REAL prediction, not the previous grid
node. No accumulated feature-path metric, current model output or deep hook is used.

On refresh, replace reference feature/sigma and prediction/residual caches. On
reuse, retain the reference and increment reuse count. First node and reuse-limit
refreshes are forced real. Budget guards reserve future mandatory refreshes;
infeasible budgets fail explicitly. max_consecutive_skips defaults to 4 in config.
Policy rollback resets private policy memory but retains committed trusted cache
state. Inspection serializes the pre-prediction context; no model replay occurs.

## Model boundaries

Implementation is in models/cache.py, pinned to Diffusers 0.35.1:

- SD3: H = pos_embed(latent); Z = norm_out input after all joint blocks. Reuse
  adds cached Z-H to current H, then computes the current conditioned output head.
- SDXL: H = conv_in(scale_model_input(latent)); Z = conv_norm_out input after
  the U-Net's down/mid/up blocks. Reuse adds Z-H, then norm/activation/conv_out.
  Its old delta contains old timestep/text effects; the head has no time condition.
- Method 3 caches final CFG-combined flow velocity (SD3) or epsilon (SDXL), before
  scheduler-specific conversion. Conditional/unconditional values are retained for
  diagnostic completeness; no prediction/feature cache crosses image boundaries.

Starter entry features are not timestep-modulated: this is NOT a full TeaCache
reimplementation. SDXL whole-U-Net residual reuse is an experimental adaptation.
No checkpoint-scale quality or acceleration claim follows from CPU parity tests.

## Running and accounting

Use scripts/run_main.py with method=prediction_cache or method=residual_cache;
the same default/explore/evaluate modes, policy export and mailbox bridge apply.
For a reduced max_nfe, allow enough budget for mandatory refreshes. Default mode
uses all-keep unless a policy is supplied or a budget guard forces reuse.

NFE denotes full backbone evaluations, including one batched CFG call; denoiser_items
counts CFG batch items on those calls. Cached nodes are reused_indices, NOT removed
skipped_indices. Log scheduler_updates, output_head_evaluations, feature_extractions,
feature_seconds, policy_seconds, cache_reuse_seconds and agent_seconds separately.

References: AdaptiveDiffusion (NeurIPS 2024) motivates final-prediction reuse;
TeaCache (CVPR 2025, official FLUX implementation) motivates whole-backbone deltas.
Our shallow-feature predicate and SDXL boundary need independent validation.
