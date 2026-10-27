"""Shared isolated analysis and dispatch for three pre-prediction mechanisms."""
import json

from .context import METHODS
from .recovery import clone_state, policy_deadline


def _analyze(programs, ctx, memory, timeout_seconds=30):
    working, proposed_memory = clone_state(ctx), clone_state(memory)
    with policy_deadline(timeout_seconds):
        observation = programs.analysis.analyze(working, proposed_memory)
    if not isinstance(observation, dict):
        raise TypeError("analysis.analyze must return a JSON observation dictionary")
    json.dumps(observation, allow_nan=False)
    limit = getattr(programs.analysis, "HISTORY_LIMIT", 2)
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("HISTORY_LIMIT must be a nonnegative integer or None")
    return observation, proposed_memory, working


def sample(adapter, programs, prompt, seed, reference_steps, max_nfe, schedule, recorder, agent=None,
           policy_timeout_seconds=30, method="grid_skip", max_consecutive_skips=4):
    """Three explicit mechanisms; changing one never changes another method's solver path."""
    if method not in METHODS or programs.method != method:
        raise ValueError("Sampling method and policy method must match")
    args = (adapter, programs, prompt, seed, reference_steps, max_nfe, schedule, recorder, agent)
    if method == "grid_skip":
        return sample_grid_skip(*args, policy_timeout_seconds=policy_timeout_seconds)
    from .cache_sampling import sample_prediction_cache, sample_residual_cache
    run = sample_prediction_cache if method == "prediction_cache" else sample_residual_cache
    return run(*args, policy_timeout_seconds=policy_timeout_seconds,
               max_consecutive_skips=max_consecutive_skips)


def sample_grid_skip(*args, **kwargs):
    """2. Decide on the current node; skip performs no prediction and no latent update."""
    from .grid_sampling import sample_grid_nodes
    return sample_grid_nodes(*args, **kwargs)
