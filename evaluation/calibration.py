"""A run's confidence: the share of healthy calibrate runs it's more unusual than. Not the probability of a fault.

Decision log, 2026-11-19 (calibration).
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PercentileConfidence:
    """One detector's healthy calibrate peaks, sorted, so any peak score can be turned into a confidence."""

    # Sorted ascending, so a binary search finds how many sit below a given peak
    healthy_peaks: np.ndarray

    def __post_init__(self) -> None:
        # Copied and locked, like Run, Normalizer and ProfileReference, so scoring can't change it
        arr = np.sort(np.array(self.healthy_peaks, dtype=float))
        arr.setflags(write=False)
        object.__setattr__(self, "healthy_peaks", arr)

    @classmethod
    def fit(cls, peaks: np.ndarray) -> "PercentileConfidence":
        peaks = np.asarray(peaks, dtype=float)
        if peaks.size == 0:
            raise ValueError("fit needs at least one healthy peak")
        if not np.isfinite(peaks).all():
            raise ValueError("healthy peaks must all be finite")
        return cls(healthy_peaks=peaks)

    def confidence(self, peak: float | np.ndarray) -> float | np.ndarray:
        """Share of healthy peaks strictly below each peak, from 0 to 1."""
        # searchsorted puts NaN last, so a broken score would get 1.0. Refuse it rather than claim certainty
        if np.isnan(peak).any():
            raise ValueError("a NaN peak has no confidence")
        # side="left" counts the peaks strictly below, so a run flags at a budget exactly when this is at least 1 minus the budget
        return np.searchsorted(self.healthy_peaks, peak, side="left") / len(self.healthy_peaks)


def share_at_or_above(confidences: np.ndarray, q: float) -> float:
    """The exit test's count: the share of runs whose confidence is at least q."""
    return float(np.mean(np.asarray(confidences) >= q))
