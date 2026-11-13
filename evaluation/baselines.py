"""Baselines the orchestrator must beat: redline, per-channel z-score, persistence forecast."""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from contracts import CHANNELS, Run
from generator.engine import REDLINES

REDLINE_THRESHOLD = 1.0  # fixed, never tuned: a channel exactly at its redline scores 1
_LIMITS = np.array([REDLINES[name] for name in CHANNELS])

def redline_scores(run: Run) -> np.ndarray:
    """Per sample, the highest channel reading as a fraction of its redline. Shape (N_STEPS,)."""
    return (run.sensors / _LIMITS).max(axis=1)

@dataclass(frozen=True)
class ZScoreBaseline:
    """Per-channel mean and std from healthy runs. Scores how far any channel strays from them."""

    mean: np.ndarray  # (len(CHANNELS),)
    std: np.ndarray  # (len(CHANNELS),)

    @classmethod
    def fit(cls, runs: Iterable[Run]) -> "ZScoreBaseline":
        n = 0
        total = np.zeros(len(CHANNELS))
        total_sq = np.zeros(len(CHANNELS))
        for run in runs:  # one run at a time, so a generator keeps memory flat
            if run.fault is not None:
                raise ValueError(f"fit on healthy runs only, got a fault in seed {run.seed}")
            n += len(run.sensors)
            total += run.sensors.sum(axis=0)
            total_sq += (run.sensors**2).sum(axis=0)

        mean = total / n
        std = np.sqrt(total_sq / n - mean**2)
        return cls(mean=mean, std=std)

    def scores(self, run: Run) -> np.ndarray:
        """Per sample, the largest absolute z-score across channels. Shape (N_STEPS,)."""
        return (np.abs(run.sensors - self.mean) / self.std).max(axis=1)

def persistence_forecast(sensors: np.ndarray, horizon: int) -> np.ndarray:
    """Forecast each sample as the value `horizon` steps earlier. The first rows repeat row 0."""
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    forecast = np.empty_like(sensors)
    forecast[horizon:] = sensors[:-horizon]
    forecast[:horizon] = sensors[0]
    return forecast