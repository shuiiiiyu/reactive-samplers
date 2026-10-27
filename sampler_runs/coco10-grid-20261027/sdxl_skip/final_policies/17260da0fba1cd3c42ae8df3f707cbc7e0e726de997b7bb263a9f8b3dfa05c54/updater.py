"""Starting SDXL rule: keep all official Euler grid nodes."""
from reactive_samplers.context import SkipAction


def choose(observation, ctx, memory):
    return SkipAction(False)
