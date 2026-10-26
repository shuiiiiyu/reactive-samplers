"""Editable starting rule: follow the official reference schedule."""
from reactive_samplers.context import Action


def choose(observation, ctx, memory):
    if ctx.remaining_nfe == 0:
        return Action(ctx.terminal_sigma)

    future = [s for s in ctx.reference_sigmas if s < ctx.sigma]
    if (0.35 <= ctx.sigma <= 0.85 and len(future) > 1
            and observation.get("history_length", 0) > 0):
        change = observation.get("velocity_change") or {}
        relative = change.get("relative_rms")
        cosine = change.get("cosine")
        # When adjacent vector-field predictions are nearly unchanged, take
        # one larger Euler interval. The high thresholds keep this rare.
        if (relative is not None and cosine is not None
                and relative < 0.04 and cosine > 0.999):
            return Action(future[1])

    return Action(future[0])
