"""Frozen contracts shared by the generator, model, evaluation harness and demo."""

import json
from dataclasses import asdict, dataclass

import numpy as np

# Channels, in fixed order. Encoders, attention weights and the demo index by position.
CHANNELS = (
    "chamber_pressure",
    "turbopump_speed",
    "turbopump_vibration",
    "bearing_temp",
    "nozzle_temp",
    "coolant_flow",
    "feed_pressure",
    "mixture_ratio",
)

COMMANDS = ("throttle",)

# Run shape: 60 s at 10 ms per sample.
DT_S = 0.01
RUN_S = 60.0
N_STEPS = round(RUN_S / DT_S)  # 6000

# Model dimensions.
WINDOW = 100  # samples per encoder input (1 s)
D_EMBED = 16  # numbers per channel embedding


@dataclass(frozen=True, eq=False)  # eq=False: == on arrays is elementwise, so compare fields explicitly
class Run:
    """One engine run. sensors[:, i] is CHANNELS[i]; throttle is kept separate."""

    throttle: np.ndarray  # (N_STEPS,)
    sensors: np.ndarray  # (N_STEPS, len(CHANNELS))
    seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "throttle", np.asarray(self.throttle, dtype=float))
        object.__setattr__(self, "sensors", np.asarray(self.sensors, dtype=float))
        if self.throttle.shape != (N_STEPS,):
            raise ValueError(f"throttle must have shape ({N_STEPS},), got {self.throttle.shape}")
        if self.sensors.shape != (N_STEPS, len(CHANNELS)):
            raise ValueError(
                f"sensors must have shape ({N_STEPS}, {len(CHANNELS)}), got {self.sensors.shape}"
            )


@dataclass(frozen=True)
class Verdict:
    """Advisory output: anomaly flag, calibrated confidence, channels ranked by relevance."""

    flag: bool
    confidence: float
    ranked_channels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        unknown = [c for c in self.ranked_channels if c not in CHANNELS]
        if unknown:
            raise ValueError(f"unknown channels: {unknown}")
        object.__setattr__(self, "ranked_channels", tuple(self.ranked_channels))

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "Verdict":
        return cls(**json.loads(text))
