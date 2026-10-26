"""Editable online observation rule. No extra denoiser or VAE call is made."""

import math

import torch
import torch.nn.functional as F


# Agent may replace this list with any SD3 transformer named module(s).
FEATURE_LAYERS = ["transformer_blocks.11"]
HISTORY_LIMIT = 2


def _number(value):
    result = float(value.detach().float().item()) if isinstance(value, torch.Tensor) else float(value)
    return result if math.isfinite(result) else None


def _rms(value):
    return _number(value.float().square().mean().sqrt())


def _stats(value):
    value = value.detach().float()
    return {"mean": _number(value.mean()), "std": _number(value.std(unbiased=False)),
            "rms": _rms(value), "max_abs": _number(value.abs().max())}


def _change(current, previous):
    current, previous = current.detach().float(), previous.detach().float()
    difference = current - previous
    cosine = F.cosine_similarity(current.flatten()[None], previous.flatten()[None]).item()
    return {"rms": _rms(difference), "relative_rms": _number(
        difference.square().mean().sqrt() / previous.square().mean().sqrt().clamp_min(1e-8)),
        "cosine": cosine if math.isfinite(cosine) else None}


def _spatial(current, previous):
    delta = (current.detach().float() - previous.detach().float()).square().mean(dim=1, keepdim=True)
    tiles = F.adaptive_avg_pool2d(delta, (4, 4)).sqrt().flatten().tolist()
    return {"tile_rms_4x4": [float(x) for x in tiles], "max_tile_rms": max(tiles)}


def _feature(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), **_stats(value)}
    if isinstance(value, (tuple, list)):
        return [_feature(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _feature(item) for key, item in value.items()}
    return {"type": type(value).__name__}


def analyze(ctx, memory):
    latent, velocity = ctx.latent, ctx.prediction.guided
    clean = latent - ctx.sigma * velocity  # SD3 flow matching x0 estimate
    observation = {
        "sigma": ctx.sigma, "timestep": ctx.timestep, "step": ctx.step,
        "nfe": ctx.nfe, "remaining_nfe": ctx.remaining_nfe,
        "history_length": len(ctx.history), "latent": _stats(latent),
        "velocity": _stats(velocity), "predicted_clean": _stats(clean),
        "velocity_change": None, "latent_change": None,
        "predicted_clean_change": None, "spatial_latent_change": None,
        "spatial_predicted_clean_change": None, "cfg": None,
        "layer_features": {name: _feature(value) for name, value in ctx.prediction.features.items()},
    }
    if ctx.history:
        previous = ctx.history[-1]
        prior_latent = previous["latent"]
        prior_velocity = previous["prediction"].guided
        prior_clean = prior_latent - previous["sigma"] * prior_velocity
        observation.update({
            "velocity_change": _change(velocity, prior_velocity),
            "latent_change": _change(latent, prior_latent),
            "predicted_clean_change": _change(clean, prior_clean),
            "spatial_latent_change": _spatial(latent, prior_latent),
            "spatial_predicted_clean_change": _spatial(clean, prior_clean),
        })
    if ctx.prediction.conditional is not None and ctx.prediction.unconditional is not None:
        conditional = ctx.prediction.conditional
        unconditional = ctx.prediction.unconditional
        observation["cfg"] = {"conditional": _stats(conditional),
                              "unconditional": _stats(unconditional),
                              "difference": _change(conditional, unconditional)}
    return observation
