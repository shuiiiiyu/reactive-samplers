"""Grid metadata and current-forward features; never probe a skipped node."""
from reactive_samplers.prompts.analysis import analyze as common_analysis, _change


FEATURE_LAYERS = ["pos_embed", "transformer_blocks.23"]
HISTORY_LIMIT = 2


def analyze(ctx, memory, input_layer="pos_embed", output_layer="transformer_blocks.23"):
    # Several candidates can share one prediction; update temporal feature state once.
    if memory.get("feature_nfe") != ctx.nfe:
        observation = common_analysis(ctx, memory)
        initial = ctx.prediction.features.get(input_layer)
        final = ctx.prediction.features.get(output_layer)
        if isinstance(final, (tuple, list)):
            final = final[1] if len(final) == 2 else None
        observation.update(feature_transform=None, input_feature_change=None, output_feature_change=None,
                           feature_delta_sigma=None)
        if initial is not None and final is not None and initial.shape == final.shape:
            names = ("unconditional", "conditional") if initial.shape[0] == 2 else ("conditional",)
            observation["feature_transform"] = {
                name: _change(final[i:i + 1], initial[i:i + 1]) for i, name in enumerate(names)}
            axes = (1,) if initial.ndim == 3 else (2, 3)
            descriptors = {"input": initial.detach().float().mean(dim=axes),
                           "output": final.detach().float().mean(dim=axes)}
            previous = memory.get("feature_descriptors")
            if previous is not None:
                observation["feature_delta_sigma"] = memory["feature_sigma"] - ctx.sigma
                observation["input_feature_change"] = _change(descriptors["input"], previous["input"])
                observation["output_feature_change"] = _change(descriptors["output"], previous["output"])
            memory["feature_descriptors"] = descriptors
            memory["feature_sigma"] = ctx.sigma
        else:
            memory.pop("feature_descriptors", None)
            memory.pop("feature_sigma", None)
        memory.update(feature_nfe=ctx.nfe, feature_observation=observation)
    observation = dict(memory["feature_observation"])
    observation["grid"] = {
        "model_index": ctx.model_index, "candidate_index": ctx.candidate_index,
        "candidate_sigma": ctx.candidate_sigma, "candidate_timestep": ctx.candidate_timestep,
        "source_sigma": ctx.sigma, "source_nfe": ctx.nfe,
        "sigma_gap": ctx.sigma - ctx.candidate_sigma,
        "skipped_since_prediction": ctx.candidate_index - ctx.model_index - 1,
    }
    return observation
