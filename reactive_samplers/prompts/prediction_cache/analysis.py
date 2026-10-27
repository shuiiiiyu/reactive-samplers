"""Pre-prediction observation: one shallow feature distance and cache age only."""
FEATURE_LAYERS = []  # Deep hooks would require the very forward being decided.
HISTORY_LIMIT = 0


def analyze(ctx, memory):
    observation = dict(ctx.cache_observation)
    observation["feature_rel_change"] = None
    if observation["has_reference"]:
        current = ctx.input_feature.detach().float()
        reference = ctx.reference_feature.detach().float()
        # Compare all spatial/token elements BEFORE scalar reduction.
        observation["feature_rel_change"] = float(
            (current - reference).abs().mean() / reference.abs().mean().clamp_min(1e-8))
    return observation
