"""Which channels a reading points at after onset, scored against the fault's ground truth channels.

Decision log, 2026-11-20 (attribution). Reported, not judged. There is no pass bar.
"""

import numpy as np

from contracts import CHANNELS, DT_S
from evaluation.budget import CATCH_WINDOW_S
from generator.splits import fault_for

# Every fault acts on 3 channels directly, so the top 3 are compared. A random pick averages 37.5%
TOP_K = 3
# The bootstrap: resamples, a fixed seed so every interval reproduces, and a 95% level
RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
LEVEL = 0.95


def rise(values: np.ndarray, onset_s: float) -> np.ndarray:
    """(N_STEPS, 8) to (8,): each channel's mean from onset to onset + 15 s, minus its own mean before onset.

    Blind rows are NaN and skipped, so the baseline starts at the first forecast. It includes startup, as the
    gate1 bias check does. The healthy control uses the same rule.
    """
    onset = round(onset_s / DT_S)
    end = round((onset_s + CATCH_WINDOW_S) / DT_S)
    return np.nanmean(values[onset:end], axis=0) - np.nanmean(values[:onset], axis=0)


def top_k(scores: np.ndarray, k: int = TOP_K) -> tuple[str, ...]:
    """The k channels with the largest scores, largest first."""
    scores = np.asarray(scores, dtype=float)
    # argsort would put a NaN channel last without a word, hiding a broken reading
    if np.isnan(scores).any():
        raise ValueError("scores contain NaN")
    # Stable, so a tie keeps channel order and the ranking reproduces
    order = np.argsort(-scores, kind="stable")[:k]
    return tuple(CHANNELS[i] for i in order)


def overlap(top: tuple[str, ...], truth: tuple[str, ...]) -> float:
    """The share of the top channels that are ground truth channels."""
    return len(set(top) & set(truth)) / len(top)


def bootstrap_interval(values: np.ndarray, resamples: int = RESAMPLES, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    """95% interval for the mean over runs, from resampling the runs with replacement."""
    values = np.asarray(values, dtype=float)
    # An empty subset is reported as n = 0, not as an interval
    if values.size == 0:
        raise ValueError("no runs to resample")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(resamples, values.size))].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
    return float(lo), float(hi)


def control_onset(split: str, seed: int) -> float:
    """For a healthy seed, the onset of the faulted seed after it. Odd seeds are faulted, so it always exists."""
    if fault_for(split, seed) is not None:
        raise ValueError(f"seed {seed} is faulted, and the control is read on healthy runs only")
    neighbor = fault_for(split, seed + 1)
    if neighbor is None:
        raise ValueError(f"seed {seed + 1} is healthy, so there's no onset to borrow")
    return neighbor.onset_s
