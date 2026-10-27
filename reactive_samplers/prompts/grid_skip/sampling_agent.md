# Method 2: current-node skipping (grid_skip)

At each original grid node k, return SkipAction(skip=bool) from
reactive_samplers.context. True omits BOTH the current full prediction AND the
current latent update: x[k+1] = x[k]. False computes at sigma[k] and applies the
official update only over [sigma[k], sigma[k+1]]. Never scan future candidates,
choose a target sigma, widen an interval or reuse a prediction.
Use decision_context.node_index for the current node.
Every grid node gets its own decision; NFE counts only real predictions.
The first and last grid intervals are forced compute. Budget reserves the final
prediction; nonfinal nodes may be forced skip. These are framework constraints.

latent_rel_change and feature_rel_change compare CURRENT inputs to inputs at the
LAST COMPLETED REAL PREDICTION, even across several skipped nodes. References update
only after a real prediction and successful update. since_real_delta_sigma uses
that same real node. A skipped node leaves latent unchanged; do not mistake this
for convergence. has_reference is absent. Sigma labels advance despite omitted
updates: this mechanism is not shortened-grid integration.

## Objective
Reduce full denoiser/backbone computation while preserving natural, realistic images
and prompt fidelity. Lower NFE alone is not success. Read both CLIPScore and
ImageReward (scores.clip_score and scores.image_reward) in new_completed_images and feedback.json; use matching prompt/seed
reference score_deltas and NFE reduction when available. Neither score guarantees
photorealism; do not add their raw values or compare unrelated prompts as a controlled
experiment. Final-image feedback is delayed and evaluates the whole trajectory,
possibly produced by several code versions. Form testable hypotheses, not claims
that one particular skip caused a score change. Exclude failed/recovered pairs from
quality/speed claims. No scorer, VAE or extra full denoiser call belongs in choose().

## Observation protocol
All decisions happen BEFORE the current complete prediction. ctx.prediction.guided
is None. Read request.observation_definitions: it contains full English meanings,
mathematical definitions, units, missing-value semantics and the exact reference
step/sigma. Never infer a field's meaning from its name. Default signals are
latent_rel_change, feature_rel_change, sigma, step_delta_sigma,
since_real_delta_sigma, consecutive_skips; has_reference exists only in cache methods.
The two change signals use mean absolute differences normalized by the reference
mean absolute magnitude (denominator floor 1e-8). Missing references produce null,
not zero. Raw latent/features remain in ctx for inspect; they are not dumped into JSON.
The shallow feature is SD3 pos_embed or SDXL officially scaled/centered conv_in,
including CFG input duplication. It is not timestep-modulated or a quality score.

You may edit run-local analysis.py, updater.py and relative Python helpers only.
FEATURE_LAYERS must remain empty. Inspect may reanalyze this saved state without a
model call; its JSON contains observation and observation_definitions. If you add,
remove or redefine signals, update analysis.OBSERVATION_SPEC in the same edit:
each field needs description, mathematical_definition, reference, range_and_units
and missing_value, all in English. New definitions accompany subsequent requests
and inspect results. Do not use undocumented signals. Read failures and previous_answer.

## Reusable rules
Write a complete policy describing which STATE CONDITIONS trigger skip/reuse or
computation. Do not encode the currently desired action as an unconditional toggle,
a hard-coded current index/image/prompt, or a copied action sequence. Do not rely
on a later Agent edit to reverse the action. Keep evidence-supported prior rules.
A constant all-compute program is a reference or explicit diagnostic, not a learned
adaptive result. A fixed stage schedule may supplement state feedback; label it
as such and express it in sigma/progress, never relative to the edit time.

Explain each rule's inputs, condition and reason. Reason through both compute and
skip/reuse cases. The code must run on a new image with fresh memory and with the
Agent disabled, and handle a memory reset during recovery. Mutable policy state
belongs in memory, not module globals. Framework observation references and caches
are separate from policy memory; changing the policy does not reset those references.
Keep weights, scheduler formulas, solver, CFG, precision, resolution, budget,
metrics, seeds and data unchanged. Do not read holdout data or another method's
strategy/session. Do not claim unmeasured improvements. When evidence is weak,
keep the existing rules and explain why.

## Required final report (English only)
End EVERY call with exactly these four headings, with a colon and your explanation.
The action here means code/rule changes, not an unobserved runtime action. Include
thresholds/relationships, whether updater.py changed, and available completed-image
scores/sample IDs in the appropriate explanation. Write "None" explicitly where
there was no change/addition. Include the expected computation/quality effect and
what evidence would refute it under the rationale heading. Framework execution and
accept/reject/revert outcomes are recorded separately; never invent them.

Signals used this round:
Actions this round (rules modified or added; None if unchanged):
Why these signal conditions support these actions:
New signals added (None if none):
