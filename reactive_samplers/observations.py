"""Shared pre-prediction signals and English mathematical definitions."""


def relative_change(current, reference):
    if reference is None:
        return None
    import torch
    current = torch.as_tensor(current).detach().float()
    reference = torch.as_tensor(reference).detach().float()
    return float((current - reference).abs().mean() / reference.abs().mean().clamp_min(1e-8))


def analyze(ctx, memory):
    return {**ctx.cache_observation,
            "latent_rel_change": relative_change(ctx.latent, ctx.reference_latent),
            "feature_rel_change": relative_change(ctx.input_feature, ctx.reference_feature)}


def definitions(ctx, analysis, observation):
    reference = "last_real_prediction_input" if ctx.method == "grid_skip" else "previous_adjacent_grid_input"
    common = {
        "reference": reference,
        "reference_step": ctx.observation_reference_step,
        "reference_sigma": ctx.observation_reference_sigma,
        "index_convention": "k increases in execution order; sigma[k] decreases. All inputs are observed before the current full prediction.",
    }
    specs = {
        "latent_rel_change": {
            "description": "Relative mean absolute change of the current noisy latent x[k] from the reference INPUT latent x[r], not a decoded image or a prediction of final quality.",
            "mathematical_definition": "mean(abs(x[k]-x[r])) / max(mean(abs(x[r])), 1e-8)",
            "reference": reference, "range_and_units": "Dimensionless, nonnegative, no fixed upper bound.",
            "missing_value": "null when no reference input exists; zero means equal tensors, not proven convergence.",
        },
        "feature_rel_change": {
            "description": "Relative mean absolute change of shallow denoiser INPUT features F[k] and F[r]. SD3 uses pos_embed; SDXL uses conv_in after official scaling/centering and CFG input preparation. Compare full tensors elementwise before reduction. These features do not include timestep modulation and are not deep backbone outputs.",
            "mathematical_definition": "mean(abs(F[k]-F[r])) / max(mean(abs(F[r])), 1e-8)",
            "reference": reference, "range_and_units": "Dimensionless, nonnegative, no fixed upper bound.",
            "missing_value": "null when no reference input exists; zero does not prove that the current denoiser output is unchanged.",
        },
        "sigma": {
            "description": "Current physical scheduler noise coordinate, after scheduler shifting. SDXL sigma may exceed 1. The scheduler label advances even when grid_skip leaves the latent unchanged.",
            "mathematical_definition": "sigma[k], from the original scheduler grid",
            "range_and_units": "Scheduler noise coordinate, decreasing toward terminal sigma; not an integer timestep or NFE.",
            "missing_value": "Never null.",
        },
        "step_delta_sigma": {
            "description": "Width of the CURRENT original grid interval. It is not the gap to a future selected candidate. Cache methods update this interval; grid_skip either updates this interval or omits it completely.",
            "mathematical_definition": "sigma[k] - sigma[k+1]",
            "range_and_units": "Positive scheduler sigma units.", "missing_value": "Never null at a decision node.",
        },
        "since_real_delta_sigma": {
            "description": "Sigma distance from the most recent completed REAL full prediction to the current grid node. Cache methods still compare latent/features to the adjacent input: this field separately measures cache age.",
            "mathematical_definition": "sigma[q] - sigma[k], where q is the last completed real-prediction node; 0 before any real prediction",
            "range_and_units": "Nonnegative scheduler sigma units.", "missing_value": "Zero before initialization; consult null change values / has_reference.",
        },
        "consecutive_skips": {
            "description": "Number of consecutive omitted nodes (grid_skip) or reused predictions/backbone residuals (cache methods) immediately BEFORE this decision. A successful real prediction resets it. It is not total skips or NFE.",
            "mathematical_definition": "c[k+1] = c[k]+1 after skip/reuse; c[k+1] = 0 after real computation; c[0] = 0",
            "range_and_units": "Nonnegative integer node count.", "missing_value": "Never null.",
        },
        "has_reference": {
            "description": "Cache methods only: whether a real prediction cache has been initialized and is available for reuse. It does not mean the cached result is accurate enough for this node. Residual caching also requires the corresponding backbone residual.",
            "mathematical_definition": "cached_prediction is not None",
            "range_and_units": "Boolean.", "missing_value": "Omitted in grid_skip; false before cache initialization.",
        },
    }
    custom = getattr(analysis, "OBSERVATION_SPEC", {})
    specs.update(custom)
    return {"context": common, "fields": {
        key: specs.get(key, {"description": "Undocumented Agent-defined signal. Read its analysis implementation and supply OBSERVATION_SPEC before using it; do not infer meaning from its name.",
                             "mathematical_definition": None}) for key in observation}}
