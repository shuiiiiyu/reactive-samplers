"""Editable starting rule: keep every candidate from the official grid."""
from reactive_samplers.context import SkipAction


def choose(observation, ctx, memory):
    return SkipAction(skip=False)
