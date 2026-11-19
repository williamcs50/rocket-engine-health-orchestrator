"""Where each sample sits in the throttle profile, so healthy references compare like with like.

Shared by the fair z score and the standardized model score (decision log, 2026-11-19).
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from contracts import DT_S, Run

# Bins of progress through the startup ramp
RAMP_BINS = 10
# Width of each time-since-change bin, in seconds
TIME_BIN_S = 0.5
# From this many seconds after the last change, every sample shares one steady bin
STEADY_S = 10.0
# Throttle level edges. Below 0.6 is the ramp and shutdown, then four bins of 0.1
LEVEL_EDGES = (0.6, 0.7, 0.8, 0.9)
# Fewest healthy samples a bin needs before its own mean and std are trusted
MIN_SAMPLES = 1000
# No std falls below this fraction of the pooled std, so nothing divides by nearly 0
STD_FLOOR = 0.01

# Samples per time bin: 50 at 10 ms. Binning in whole samples means no rounding can cross an edge
SAMPLES_PER_TIME_BIN = round(TIME_BIN_S / DT_S)
# 20 time bins of 0.5 s up to 10 s, plus the steady bin
N_TIME_BINS = round(STEADY_S / TIME_BIN_S) + 1
# Phases: the ramp bins first, then the time bins, so a later phase is always later in time
N_PHASES = RAMP_BINS + N_TIME_BINS
# Up or down
N_DIRECTIONS = 2
N_LEVELS = len(LEVEL_EDGES) + 1
# Every combination of phase, direction and level
N_BINS = N_PHASES * N_DIRECTIONS * N_LEVELS
SHAPE = (N_PHASES, N_DIRECTIONS, N_LEVELS)


def bin_index(throttle: np.ndarray) -> np.ndarray:
    """(N_STEPS,) throttle to (N_STEPS,) bin numbers, one per sample."""
    i = np.arange(len(throttle))
    # Each sample's change from the one before. Sample 0 has nothing before it, so its change is 0
    step = np.diff(throttle, prepend=throttle[0])
    changed = step != 0

    # The startup ramp is the first stretch of changing samples. ramp_end is the first sample that holds still
    ramp_end = 1 + int(np.argmax(~changed[1:]))
    in_ramp = i < ramp_end

    # Progress through the ramp in RAMP_BINS bins. Whole-number division, so no float sits between
    ramp_bin = np.minimum(i * RAMP_BINS // ramp_end, RAMP_BINS - 1)

    # The most recent sample that changed. The ramp's last sample counts as a change
    last = np.maximum.accumulate(np.where(changed, i, 0))
    # Samples since the last change, kept as a whole number
    since = i - last
    # 0.5 s bins, with everything from 10 s onwards in the last, steady bin
    time_bin = np.minimum(since // SAMPLES_PER_TIME_BIN, N_TIME_BINS - 1)

    phase = np.where(in_ramp, ramp_bin, RAMP_BINS + time_bin)
    # Up if the last change went up. The startup ramp is always up
    direction = np.where(in_ramp, 1, (step[last] > 0).astype(int))
    # 0 below 0.6, then 1 to 4 for each 0.1 up to 1.0
    level = np.digitize(throttle, LEVEL_EDGES)

    return np.ravel_multi_index((phase, direction, level), SHAPE)


def unpack(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin numbers back to (phase, direction, level)."""
    return np.unravel_index(keys, SHAPE)


def merge_along_time(counts: np.ndarray, min_samples: int) -> np.ndarray | None:
    """One column's sample counts, earliest phase first, to a group number per phase. None if too sparse."""
    # The whole column can't reach the minimum, so the caller falls back to pooled stats
    if counts.sum() < min_samples:
        return None
    groups = np.zeros(len(counts), dtype=int)
    group, filled = 0, 0
    for phase, count in enumerate(counts):
        groups[phase] = group
        filled += count
        # This group has enough, so the next phase starts a new one
        if filled >= min_samples:
            group, filled = group + 1, 0
    # Phases after the last full group never reached the minimum, so they merge backwards into it
    groups[groups == group] = group - 1
    return groups


@dataclass(frozen=True)
class ProfileReference:
    """Healthy mean and std of a per-sample quantity in every bin. Fit on healthy runs only."""

    # (N_BINS, k): one row per bin, one column per value (8 channels, or 1)
    mean: np.ndarray
    std: np.ndarray
    # Bins, and training samples in them, that use the pooled stats
    fallback_bins: int
    fallback_samples: int

    def __post_init__(self) -> None:
        # Frozen stops reassigning a field but not editing an array in place. The reference must stay
        # identical between fitting and scoring, so each array is copied and locked, as in Normalizer
        for name in ("mean", "std"):
            arr = np.array(getattr(self, name), dtype=float)
            arr.setflags(write=False)
            object.__setattr__(self, name, arr)

    @classmethod
    def fit(
        cls, runs: Iterable[Run], values: Callable[[Run], np.ndarray], min_samples: int = MIN_SAMPLES
    ) -> "ProfileReference":
        counts = np.zeros(N_BINS)
        sums = squares = None
        for run in runs:
            if run.fault is not None:
                raise ValueError(f"fit on healthy runs only, but seed {run.seed} has a fault")
            # Always 2-D inside, (N_STEPS, k), so 8 channels and a single value share the same code
            v = np.asarray(values(run), dtype=float).reshape(len(run.throttle), -1)
            keys = bin_index(run.throttle)
            # Rows with any NaN, such as the blind first 1.09 s, stay out of every statistic
            ok = np.isfinite(v).all(axis=1)
            keys, v = keys[ok], v[ok]
            if sums is None:
                sums = np.zeros((N_BINS, v.shape[1]))
                squares = np.zeros_like(sums)
            counts += np.bincount(keys, minlength=N_BINS)
            for c in range(v.shape[1]):
                sums[:, c] += np.bincount(keys, weights=v[:, c], minlength=N_BINS)
                squares[:, c] += np.bincount(keys, weights=v[:, c] ** 2, minlength=N_BINS)
        if sums is None:
            raise ValueError("fit needs at least one run")

        # Pooled over every sample: the fallback, and the scale for the std floor
        n = counts.sum()
        pooled_mean = sums.sum(axis=0) / n
        # Variance as E[x²] − mean². Rounding can push it just below 0 when the mean is large next to the
        # spread, so it's clipped at 0 before the square root, here and per group below
        pooled_std = np.sqrt(np.maximum(squares.sum(axis=0) / n - pooled_mean**2, 0.0))
        # With no spread at all, the floor would be 0 too and standardizing would divide by 0. Refuse instead
        if (pooled_std == 0).any():
            raise ValueError("a quantity with no spread can't be standardized")

        mean = np.empty_like(sums)
        std = np.empty_like(sums)
        fallback_bins, fallback_samples = 0, 0
        for d in range(N_DIRECTIONS):
            for lv in range(N_LEVELS):
                # The bin numbers of every phase in this direction and level, earliest first
                column = np.ravel_multi_index((np.arange(N_PHASES), d, lv), SHAPE)
                groups = merge_along_time(counts[column], min_samples)
                if groups is None:
                    mean[column], std[column] = pooled_mean, pooled_std
                    fallback_bins += len(column)
                    fallback_samples += int(counts[column].sum())
                    continue
                for g in np.unique(groups):
                    members = column[groups == g]
                    m = sums[members].sum(axis=0) / counts[members].sum()
                    s = np.sqrt(np.maximum(squares[members].sum(axis=0) / counts[members].sum() - m**2, 0.0))
                    mean[members], std[members] = m, s

        # A flat bin would divide by 0, so every std is at least 1% of the pooled std
        std = np.maximum(std, STD_FLOOR * pooled_std)
        print(f"profile reference: {fallback_bins} of {N_BINS} bins and {fallback_samples} of {int(n)} samples use pooled stats")
        return cls(mean=mean, std=std, fallback_bins=fallback_bins, fallback_samples=fallback_samples)

    def standardize(self, values: np.ndarray, throttle: np.ndarray) -> np.ndarray:
        """(values − bin mean) / bin std, sample by sample, in the shape it was given. NaN stays NaN."""
        keys = bin_index(throttle)
        v = np.asarray(values, dtype=float)
        z = (v.reshape(len(keys), -1) - self.mean[keys]) / self.std[keys]
        return z.reshape(v.shape)
