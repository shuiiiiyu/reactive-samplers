# Three sampling mechanisms

| ID | Method | Current-node decision | Implementation |
|---|---|---|---|
| 2 | grid_skip | Skip both prediction and latent update, or compute/update this original interval | grid_sampling.sample_grid_nodes |
| 3 | prediction_cache | Reuse the last real CFG-combined prediction or recompute; always update the latent | cache_sampling.sample_prediction_cache |
| 4 | residual_cache | Current entry feature + cached backbone residual + current output head, or recompute; always update | cache_sampling.sample_residual_cache |

All decisions precede the current full prediction. grid_skip visits every original
node once; it never scans future candidates or widens the update interval. Its
first and last intervals are protected, and the NFE budget reserves the final call.
Cache methods protect initialization and maximum consecutive reuse; their budget
guards reserve mandatory future refreshes. Initial updaters compute every node.

## Observations

Default fields: latent_rel_change, feature_rel_change, sigma, step_delta_sigma,
since_real_delta_sigma, consecutive_skips; has_reference exists only in cache methods.

For current input x[k] and shallow feature F[k], change is
`mean(abs(current-reference)) / max(mean(abs(reference)), 1e-8)`.
The first reference is missing and changes are null. Grid skipping uses inputs at
the last completed real prediction; cache methods use the immediately preceding
grid input, including reused nodes. References belong to the framework and survive
policy-memory resets. Prediction/residual caches separately update only on refresh.

`step_delta_sigma = sigma[k]-sigma[k+1]` is the current original interval width.
`since_real_delta_sigma = sigma[q]-sigma[k]`, where q is the last real prediction.
Request JSON and inspect include full English definitions, units, missing-value
semantics and exact reference step/sigma. New signals need OBSERVATION_SPEC entries.

## Cache boundaries

Diffusers 0.35.1 adapters keep the original model and scheduler formulas:

- SD3: H = pos_embed output; Z = norm_out input after all transformer blocks.
  Residual reuse runs current time/text embedding, norm_out, proj_out and unpatchify.
- SDXL: H = conv_in output after official scaling/centering; Z = conv_norm_out
  input after down/mid/up blocks. Reuse runs norm/activation/conv_out. The cached
  delta contains old time/text effects; the output head is unconditioned.
- Prediction caching stores final CFG-combined velocity (SD3) or epsilon (SDXL).

Shallow observation features are not timestep-modulated. These are experimental
cache adaptations, not a complete TeaCache implementation or a quality guarantee.

## Feedback and records

CLIPScore and ImageReward are computed once per completed image. feedback.json
contains completed results and paired prompt/seed reference deltas when available.
The next Agent request includes new_completed_images. Each call writes four English
report sections to agent_decisions/ as JSON and Markdown; trusted acceptance,
rejection, reversion and actual execution outcomes are recorded separately.
The RunPod gallery reads these records directly and offers per-image downloads.

NFE counts full predictions. Cache methods log every scheduler update; grid_skip
logs omitted nodes and only the updates actually performed. Agent waits, feature
cost and scoring cost must be accounted for when assessing acceleration. Frozen
evaluation executes the exported rules without an LLM. Older policies built for
the previous grid-skip semantics require new exploration and validation.
