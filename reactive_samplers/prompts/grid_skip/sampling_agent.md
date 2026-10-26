# Fixed-grid timestep skipping

Discover a Python keep/skip rule on the original SD3 Scheduler grid. This is a
separate experiment from adaptive_sigma: do not read its policy or conversation.

The model has already predicted at ctx.sigma. ctx.candidate_index identifies a
later reference node being considered, not an already-generated latent. Return
SkipAction(skip=True/False), imported from reactive_samplers.context. The framework
alone advances the grid cursor and selects the target sigma. Never return Action,
a timestep, a sigma, an index or a step size. The first model node and terminal
boundary are protected. Remaining-budget constraints are enforced by the engine.

You may freely edit analysis.py, updater.py and relative Python helpers in this
run's workspace. Add, remove or combine observations; FEATURE_LAYERS may request
existing transformer named modules on future forwards. Initial features compare
pos_embed output to the final block's image hidden states before the output head.
The current complete output can only guide later candidates. There is no prediction
or feature from a skipped node; do not fabricate one or run an extra model forward.

Read the request, programs, prior outcomes/failures and completed-image feedback.
Within this pause, repeatedly edit analysis and run the provided inspect command.
Then edit or keep updater. Agent checkpoints follow actual predictions; the Python
rule is called for each candidate, including several candidates at the same NFE.
Keep repeated analysis at the same NFE consistent; it is not a new model observation.

Use only current/past state. Mutable policy state belongs in memory; it resets per
image and may reset on recovery at a noninitial timestep. The full policy pair is
versioned; a new version needs two successful latent updates to qualify as fallback.
Rejected candidate scans are discarded before retrying from the committed grid cursor.

Do not change the model, solver, CFG, precision, evaluation, prompts or seeds. Do not
read holdout prompts/results. When reference_images are provided, compare matching
prompt/seed quality and NFE. Frozen inference runs these rules without an LLM.
End with observations, edits/keep, a prediction, and evidence that would refute it.
