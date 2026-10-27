"""Start with a real backbone on every node; Agent can learn when to reuse its delta."""
from reactive_samplers.context import SkipAction


def choose(observation, ctx, memory):
    return SkipAction(skip=False)  # True uses current input + cached backbone residual.
