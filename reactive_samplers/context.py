from dataclasses import dataclass, field
from typing import Any
import math

METHODS = ("adaptive_sigma", "grid_skip")


@dataclass
class Prediction:
    guided: Any
    conditional: Any = None
    unconditional: Any = None
    features: dict = field(default_factory=dict)


@dataclass
class Context:
    prompt: str
    seed: int
    step: int                       # zero-based; this step's prediction already exists
    nfe: int                        # includes the current prediction
    max_nfe: int
    sigma: float                    # physical sigma after scheduler shifting
    terminal_sigma: float
    timestep: float
    reference_sigmas: tuple
    latent: Any
    prediction: Prediction
    history: list                  # completed steps only; no future results
    analysis_memory: dict = field(default_factory=dict)  # state before this step's initial analysis
    sample_id: str = ""
    method: str = "adaptive_sigma"
    model_index: int | None = None
    candidate_index: int | None = None
    candidate_timestep: float | None = None

    @property
    def candidate_sigma(self):
        return self.reference_sigmas[self.candidate_index] if self.candidate_index is not None else None

    @property
    def remaining_nfe(self):
        return self.max_nfe - self.nfe


@dataclass(frozen=True)
class Action:
    next_sigma: float


@dataclass(frozen=True)
class SkipAction:
    skip: bool


def validate_skip(action, ctx):
    if not isinstance(action, SkipAction) or type(action.skip) is not bool:
        raise TypeError("grid_skip updater.choose must return SkipAction(skip=bool)")
    if type(ctx.model_index) is not int or type(ctx.candidate_index) is not int or not (
            0 <= ctx.model_index < ctx.candidate_index < len(ctx.reference_sigmas)):
        raise ValueError("Grid candidates must follow the current model node in the reference schedule")
    if ctx.candidate_index == len(ctx.reference_sigmas) - 1 and action.skip:
        raise ValueError("The terminal boundary cannot be skipped")
    return action.skip


def validate_action(action, ctx):
    if not isinstance(action, Action):
        raise TypeError("updater.choose must return Action(next_sigma=...)")
    value = float(action.next_sigma)
    if not math.isfinite(value) or not ctx.terminal_sigma <= value < ctx.sigma:
        raise ValueError("next_sigma must be finite, strictly decreasing and within the terminal boundary")
    if ctx.remaining_nfe == 0 and value != ctx.terminal_sigma:
        raise ValueError("The final available prediction must advance to terminal_sigma")
    return value
