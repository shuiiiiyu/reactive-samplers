from dataclasses import dataclass


@dataclass(frozen=True)
class CheckSchedule:
    mode: str = "every_step"
    checkpoints: tuple = ()

    def __post_init__(self):
        if self.mode not in {"every_step", "checkpoints"}:
            raise ValueError("Agent schedule must be every_step or checkpoints")
        if any(type(i) is not int or i < 0 for i in self.checkpoints):
            raise ValueError("Checkpoints are nonnegative, zero-based executed-step indices")

    def due(self, step):
        return self.mode == "every_step" or step in self.checkpoints
