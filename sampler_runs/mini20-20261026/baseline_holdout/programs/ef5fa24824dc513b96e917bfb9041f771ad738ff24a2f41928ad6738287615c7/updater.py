"""Editable starting rule: follow the official reference schedule."""
from reactive_samplers.context import Action


def choose(observation, ctx, memory):
    if ctx.remaining_nfe == 0:
        return Action(ctx.terminal_sigma)
    return Action(next(s for s in ctx.reference_sigmas if s < ctx.sigma))
