"""The new score: the masked residual's level, then its lagged difference, each standardized against healthy runs.

Decision log, 2026-11-19 (next scoring fix). The residual is the mean aggregate, as pre-registered on D6.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from contracts import WINDOW, Run
from evaluation.profile import MIN_SAMPLES, ProfileReference
from generator.splits import SPLITS
from model.data import HORIZON

# The baseline window: from 2,000 samples (20 s) back, up to but not including 1,500 samples (15 s) back.
# The 15 s gap matches the catch window, so a fault can't leak into its own baseline while a catch still counts
LAG_FAR = 2000
LAG_NEAR = 1500
# The model's first forecast. Every row before it is blind, and must stay NaN until the very end
FIRST_FORECAST = WINDOW - 1 + HORIZON
# The first sample whose baseline window has a forecast throughout: the first forecast (109) plus 2,000
FIRST_DIFF = FIRST_FORECAST + LAG_FAR
# Healthy train runs never trained on (training used the first 500), and outside calibrate and val
REFERENCE_SEEDS = SPLITS["train"][500:1000]


def mean_residual(r: np.ndarray) -> np.ndarray:
    """(N_STEPS, 8) residuals to (N_STEPS,) by the mean, keeping the blind rows NaN.

    This is the D6 mean aggregate without aggregate's fill of 0, which this score must not see.
    """
    return r.mean(axis=1)


def check_blind_rows(s: np.ndarray) -> None:
    """Refuse a residual whose blind rows were filled in, for example by aggregate's 0."""
    if np.isfinite(s[:FIRST_FORECAST]).any():
        raise ValueError("the blind rows must be NaN. Use mean_residual, not aggregate, which fills them with 0")


def lagged_difference(s: np.ndarray) -> np.ndarray:
    """s[t] minus the mean of s over the half-open window [t - 2000, t - 1500). NaN where that window isn't complete."""
    n = len(s)
    finite = np.isfinite(s)
    # Cumulative sums with a leading 0, so the sum over [a, b) is total[b] - total[a]. NaN can't go into the sum,
    # since one NaN would make every later sum NaN, so NaNs are counted separately and their windows set to NaN
    total = np.concatenate([[0.0], np.cumsum(np.where(finite, s, 0.0))])
    missing = np.concatenate([[0], np.cumsum(~finite)])
    t = np.arange(LAG_FAR, n)
    start, stop = t - LAG_FAR, t - LAG_NEAR
    window_mean = (total[stop] - total[start]) / (LAG_FAR - LAG_NEAR)
    complete = (missing[stop] - missing[start]) == 0
    d = np.full(n, np.nan)
    d[t] = np.where(complete, s[t] - window_mean, np.nan)
    return d


@dataclass(frozen=True)
class StandardizedScore:
    """Two healthy references from the same residuals: one on the level, one on the lagged difference."""

    level: ProfileReference
    difference: ProfileReference

    @classmethod
    def fit(
        cls, runs: Iterable[Run], residual_fn: Callable[[Run], np.ndarray], min_samples: int = MIN_SAMPLES
    ) -> "StandardizedScore":
        runs = list(runs)
        # Masked scoring is the slow part, so each run is scored once and both references reuse it
        levels = {id(run): np.asarray(residual_fn(run), dtype=float) for run in runs}
        for s in levels.values():
            check_blind_rows(s)
        level = ProfileReference.fit(runs, lambda run: levels[id(run)], min_samples)
        difference = ProfileReference.fit(runs, lambda run: lagged_difference(levels[id(run)]), min_samples)
        return cls(level=level, difference=difference)

    def scores(self, s: np.ndarray, throttle: np.ndarray) -> np.ndarray:
        """(N_STEPS,): the level z before FIRST_DIFF, the difference z from it on, -inf for the blind rows."""
        check_blind_rows(s)
        level_z = self.level.standardize(s, throttle)
        diff_z = self.difference.standardize(lagged_difference(s), throttle)
        combined = level_z.copy()
        combined[FIRST_DIFF:] = diff_z[FIRST_DIFF:]
        # NaNs are kept until here, then filled with a value that can never alarm
        return np.where(np.isnan(combined), -np.inf, combined)
