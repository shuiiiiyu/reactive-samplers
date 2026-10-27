"""Start with all real predictions; Agent discovers a reuse rule from evidence."""
from reactive_samplers.context import SkipAction


def choose(observation, ctx, memory):
    return SkipAction(skip=False)  # True reuses the last final prediction.
