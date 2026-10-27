# Method 4: whole-backbone residual cache

Return SkipAction(skip=bool). False performs the official full prediction and caches
R=Z-H. True computes current H + cached R, then the original current output head.
This is a hidden-space residual, not a final velocity/noise residual. SD3 uses
pos_embed -> norm_out input; SDXL uses scaled/centered conv_in -> conv_norm_out
input after all down/mid/up blocks. SDXL's residual contains old time/text effects
and its output head is unconditioned. Do not claim equivalence to TeaCache.
Every original interval still gets its official scheduler update.
Both change signals compare CURRENT inputs to the IMMEDIATELY PRECEDING original
grid node's inputs, including reused nodes. These adjacent references are separate
from the residual/prediction cache. since_real_delta_sigma measures real-cache age;
has_reference reports initialization. First-node and reuse-limit refreshes are
forced real; budget guards reserve mandatory refreshes. No current deep feature
or complete prediction is available for the decision.

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
