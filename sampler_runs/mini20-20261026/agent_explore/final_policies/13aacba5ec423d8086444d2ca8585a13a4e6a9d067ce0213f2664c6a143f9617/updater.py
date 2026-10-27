"""Editable starting rule: follow the official reference schedule."""
from reactive_samplers.context import Action


def choose(observation, ctx, memory):
    if ctx.remaining_nfe == 0:
        return Action(ctx.terminal_sigma)

    future = [s for s in ctx.reference_sigmas if s < ctx.sigma]
    if (0.35 <= ctx.sigma <= 0.85 and len(future) > 1
            and observation.get("history_length", 0) > 0):
        clean_change = observation.get("predicted_clean_change") or {}
        clean_relative = clean_change.get("relative_rms")
        clean_cosine = clean_change.get("cosine")
        velocity_change = observation.get("velocity_change") or {}
        velocity_relative = velocity_change.get("relative_rms")
        velocity_cosine = velocity_change.get("cosine")

        # A stable x0 estimate can justify one coarser interval even while
        # the vector field itself continues to drift across timesteps.
        clean_is_stable = (clean_relative is not None and clean_cosine is not None
                           and clean_relative < 0.10 and clean_cosine > 0.995)
        velocity_is_stable = (velocity_relative is not None and velocity_cosine is not None
                              and velocity_relative < 0.04 and velocity_cosine > 0.999)
        if clean_is_stable or velocity_is_stable:
            return Action(future[1])

    return Action(future[0])
