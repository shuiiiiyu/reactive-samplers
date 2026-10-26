# SDXL fixed-grid timestep skipping

Discover a training-free Python keep/skip rule for SDXL Base and its official
EulerDiscreteScheduler. This run is independent of all SD3 runs: do not read their
policies, conversations, prompts or results. No solver or model formula changes.

The current UNet noise prediction already exists; sampling is paused before the
latent update. Return only SkipAction(skip=True/False). The candidate is a future
node of the original grid, not a latent or prediction already evaluated there.
Never return a sigma, timestep, index, step size or Action. The engine protects
the first model node, terminal boundary and remaining NFE budget.

You may edit analysis.py, updater.py and relative Python helpers in this workspace.
Use the provided inspect command repeatedly to reanalyze this exact paused state.
Change, combine, remove or add observations. FEATURE_LAYERS names UNet modules;
newly requested hooks apply on future forwards. Initial conv_in and conv_norm_out
features compare input-projection and final hidden activations before the output
head. Model outputs are epsilon, not SD3 flow velocity. The clean estimate is
x0 = current unscaled latent - sigma * guided epsilon. Sigma is not bounded by 1.
Skipped nodes have no prediction or features; never fabricate them or run another
model forward. Several candidate checks at the same NFE share one observation.

Read the request, previous outcomes/failures and completed-image feedback. Match
reference images by prompt and seed. You may keep a rule when evidence is weak.
Do not claim quality or speed gains without measurements. Keep rules inexpensive.
Mutable state belongs in memory, which resets per image and may reset mid-image
on recovery. Policy edits undergo same-step validation; two committed updates
are required for fallback eligibility. Earlier committed steps are retained.

Do not modify weights, CFG, solver, precision, NFE budget, resolution, data, seeds
or evaluation. Do not read heldout prompts/results. End with a concise observation,
edit/keep rationale, prediction and evidence that would refute it. This final
answer is logged with the actual kept/skipped grid decisions; do not invent actions.
