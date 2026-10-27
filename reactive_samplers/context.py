from dataclasses import dataclass, field
from typing import Any

CACHE_METHODS = ("prediction_cache", "residual_cache")
METHODS = ("grid_skip", *CACHE_METHODS)


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
    step: int                       # zero-based; cache methods pause BEFORE prediction
    nfe: int                        # completed real predictions; current prediction is not yet available
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
    method: str = "grid_skip"
    # All methods expose shallow current features and trusted historical references.
    # prediction.guided is None during every pre-prediction decision.
    input_feature: Any = None
    reference_feature: Any = None
    cache_observation: dict = field(default_factory=dict)
    reference_latent: Any = None
    observation_reference_step: int | None = None
    observation_reference_sigma: float | None = None

    @property
    def remaining_nfe(self):
        return self.max_nfe - self.nfe


@dataclass(frozen=True)
class SkipAction:
    skip: bool


def validate_skip(action):
    if not isinstance(action, SkipAction) or type(action.skip) is not bool:
        raise TypeError("updater.choose must return SkipAction(skip=bool)")
    return action.skip
