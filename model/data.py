"""Normalization from healthy train runs, and forecasting windows cut from a run."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from contracts import CHANNELS, WINDOW, Run
from generator.engine import SENSOR_NOISE

# Samples ahead. The D5 starting value, shared by training and scoring
HORIZON = 10


@dataclass(frozen=True)
class Normalizer:
    """Per-channel mean and std from healthy runs, plus each channel's noise floor."""

    # (len(CHANNELS),)
    mean: np.ndarray
    # (len(CHANNELS),)
    std: np.ndarray
    # Sensor noise variance in normalized units
    floor: np.ndarray

    def __post_init__(self) -> None:
        # Frozen stops reassigning a field but not editing an array in place. These numbers must stay
        # identical between training and scoring, so each array is copied and locked
        for name in ("mean", "std", "floor"):
            arr = np.array(getattr(self, name), dtype=float)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)

    @classmethod
    def fit(cls, runs: Iterable[Run]) -> "Normalizer":
        n = 0
        total = np.zeros(len(CHANNELS))
        total_sq = np.zeros(len(CHANNELS))

        for run in runs:
            if run.fault is not None:
                raise ValueError(f"fit on healthy runs only, but seed {run.seed} has a fault")

            n += len(run.sensors)
            total += run.sensors.sum(axis=0)
            total_sq += (run.sensors**2).sum(axis=0)

        if n == 0:
            raise ValueError("fit needs at least one run")
        mean = total / n
        # Rounding can dip just below 0
        std = np.sqrt(np.maximum(total_sq / n - mean**2, 0.0))
        # A flat channel can leave a rounding residue near 1e-8 instead of exactly 0, which would make
        # its floor huge and let it dominate the loss. So anything this small relative to the level counts as flat
        tiny = std <= 1e-6 * np.maximum(1.0, np.abs(mean))
        if tiny.any():
            raise ValueError(f"zero spread in {[c for c, t in zip(CHANNELS, tiny) if t]}")

        noise = np.array([SENSOR_NOISE[name] for name in CHANNELS])

        return cls(mean=mean, std=std, floor=noise**2 / std**2)

    def to_dict(self) -> dict[str, list[float]]:
        """Plain lists, so the stats can live in a run's config.json."""
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "floor": self.floor.tolist()}

    @classmethod
    def from_dict(cls, stats: dict[str, list[float]]) -> "Normalizer":
        # __post_init__ turns the lists back into locked arrays
        return cls(mean=stats["mean"], std=stats["std"], floor=stats["floor"])

    def transform(self, sensors: np.ndarray) -> np.ndarray:
        # A new array, so read only Run arrays are never touched
        return (sensors - self.mean) / self.std


def make_windows(
    sensors: np.ndarray, throttle: np.ndarray, horizon: int = HORIZON, stride: int = 1
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sensor windows (n, 8, WINDOW), throttle to the target time (n, WINDOW + horizon), targets (n, 8). All views."""
    # Horizon 0 would make each target the window's own last sample, so the model would learn to report the present
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, not {horizon}")
    if stride < 1:
        raise ValueError(f"stride must be at least 1, not {stride}")
    # Windows whose target is still inside the run
    n = len(sensors) - WINDOW - horizon + 1
    windows = sliding_window_view(sensors, WINDOW, axis=0)[:n:stride]
    throttle_hist = sliding_window_view(throttle, WINDOW + horizon)[:n:stride]
    targets = sensors[WINDOW - 1 + horizon :][::stride]
    return windows, throttle_hist, targets
