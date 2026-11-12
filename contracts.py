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

FAULT_KINDS = ("bearing_wear",)


@dataclass(frozen=True)
class FaultLabel:
    """Ground truth for an injected fault. Read by evaluation only, never by the model."""

    kind: str
    onset_s: float
    rate: float  # increase in friction (mu) per second, relative to healthy
    channels: tuple[str, ...]  # channels the fault acts on directly

    def __post_init__(self) -> None:
        object.__setattr__(self, "channels", tuple(self.channels))
        if self.kind not in FAULT_KINDS:
            raise ValueError(f"unknown fault kind {self.kind!r}, expected one of {FAULT_KINDS}")
        if not 0.0 <= self.onset_s <= RUN_S:
            raise ValueError(f"onset_s must be in [0, {RUN_S}], got {self.onset_s}")
        if not self.rate > 0:  # written this way so NaN fails too
            raise ValueError(f"rate must be positive, got {self.rate}")
        if not self.channels:
            raise ValueError("channels must name at least one channel")
        unknown = [c for c in self.channels if c not in CHANNELS]
        if unknown:
            raise ValueError(f"unknown channels: {unknown}")


@dataclass(frozen=True, eq=False)  # eq=False: == on arrays is elementwise, so compare fields explicitly
class Run:
    """One engine run. sensors[:, i] is CHANNELS[i]; throttle is kept separate."""

    throttle: np.ndarray  # (N_STEPS,)
    sensors: np.ndarray  # (N_STEPS, len(CHANNELS))
    seed: int
    fault: FaultLabel | None = None  # None means healthy

    def __post_init__(self) -> None:
        object.__setattr__(self, "throttle", np.array(self.throttle, dtype=float))  # copies
        object.__setattr__(self, "sensors", np.array(self.sensors, dtype=float))
        self.throttle.setflags(write=False)  # frozen stops reassignment, this stops in-place edits
        self.sensors.setflags(write=False)
        if self.throttle.shape != (N_STEPS,):
            raise ValueError(f"throttle must have shape ({N_STEPS},), got {self.throttle.shape}")
        if self.sensors.shape != (N_STEPS, len(CHANNELS)):
            raise ValueError(
                f"sensors must have shape ({N_STEPS}, {len(CHANNELS)}), got {self.sensors.shape}"
            )
        if self.fault is not None and not isinstance(self.fault, FaultLabel):
            raise ValueError(f"fault must be a FaultLabel or None, got {type(self.fault).__name__}")


@dataclass(frozen=True)
class Verdict:
    """Advisory output: anomaly flag, calibrated confidence, channels ranked by relevance."""

    flag: bool
    confidence: float
    ranked_channels: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "ranked_channels", tuple(self.ranked_channels))
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        unknown = [c for c in self.ranked_channels if c not in CHANNELS]
        if unknown:
            raise ValueError(f"unknown channels: {unknown}")

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, text: str) -> "Verdict":
        return cls(**json.loads(text))
