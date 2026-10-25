from dataclasses import dataclass, field
from typing import Any
import math


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

    @property
    def remaining_nfe(self):
        return self.max_nfe - self.nfe


@dataclass(frozen=True)
class Action:
    next_sigma: float


def validate_action(action, ctx):
    if not isinstance(action, Action):
        raise TypeError("updater.choose must return Action(next_sigma=...)")
    value = float(action.next_sigma)
    if not math.isfinite(value) or not ctx.terminal_sigma <= value < ctx.sigma:
        raise ValueError("next_sigma must be finite, strictly decreasing and within the terminal boundary")
    if ctx.remaining_nfe == 0 and value != ctx.terminal_sigma:
        raise ValueError("The final available prediction must advance to terminal_sigma")
    return value
