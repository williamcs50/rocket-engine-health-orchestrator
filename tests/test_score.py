import numpy as np

from contracts import CHANNELS, DT_S, N_STEPS, Run
from evaluation.score import FIRST_DIFF, REFERENCE_SEEDS, StandardizedScore, lagged_difference
from generator.splits import SPLITS
from model.train import N_TRAIN_RUNS

# The model's first forecast. Residuals are NaN before it
FIRST_FORECAST = 109


def throttle() -> np.ndarray:
    """A known throttle: ramp to 0.75 over 3 s, step up to 0.95 at 20 s, down to 0.65 at 40 s, shutdown at 55 s."""
    knots = [(0.0, 0.0), (3.0, 0.75), (20.0, 0.75), (20.2, 0.95), (40.0, 0.95), (40.2, 0.65), (55.0, 0.65), (57.5, 0.0), (60.0, 0.0)]
    t, v = zip(*knots)
    return np.interp(np.arange(N_STEPS) * DT_S, t, v)


def residuals(seed: int) -> np.ndarray:
    """A made-up mean residual that varies over time, NaN before the first forecast like real ones."""
    t = np.arange(N_STEPS)
    s = 5.0 + seed + np.sin(t / (40.0 + 7 * seed)) * (1.0 + 0.3 * seed)
    s[:FIRST_FORECAST] = np.nan
    return s


def fitted() -> tuple[StandardizedScore, list[Run]]:
    # Constant runs would make every difference exactly 0, so these vary over time
    runs = [Run(throttle=throttle(), sensors=np.zeros((N_STEPS, len(CHANNELS))), seed=s) for s in range(4)]
    return StandardizedScore.fit(runs, lambda run: residuals(run.seed), min_samples=1), runs


# The lagged difference

def test_lag_window_is_20_to_15_seconds_back() -> None:
    # For s = 0, 1, 2, ..., the half-open window [t - 2000, t - 1500) averages to t - 1750.5
    d = lagged_difference(np.arange(N_STEPS, dtype=float))
    np.testing.assert_allclose(d[FIRST_DIFF:], 1750.5)


def test_difference_starts_at_2109() -> None:
    # With the blind rows NaN, as in real residuals, the first complete window starts at sample 109
    s = np.arange(N_STEPS, dtype=float)
    s[:FIRST_FORECAST] = np.nan
    d = lagged_difference(s)
    assert np.isnan(d[FIRST_DIFF - 1])
    assert np.isfinite(d[FIRST_DIFF])
    assert FIRST_DIFF == 2109


def test_a_per_run_offset_disappears() -> None:
    # The reason the score exists: a constant calibration offset cancels out
    s = np.random.default_rng(0).normal(size=N_STEPS)
    # atol, since differences of random normals can sit near 0, where a relative tolerance alone is too strict
    np.testing.assert_allclose(lagged_difference(s + 37.0), lagged_difference(s), atol=1e-9)


def test_a_rise_survives() -> None:
    # Removing the offset must not remove the fault. A step of 10 at sample 3000 shows in full until the
    # baseline window reaches it, 1,500 samples later
    s = np.zeros(N_STEPS)
    s[3000:] = 10.0
    np.testing.assert_allclose(lagged_difference(s)[3000:4501], 10.0)


# The combined score

def test_level_before_2109_difference_after() -> None:
    score, runs = fitted()
    s, thr = residuals(0), runs[0].throttle
    level_z = score.level.standardize(s, thr)
    diff_z = score.difference.standardize(lagged_difference(s), thr)
    combined = score.scores(s, thr)
    assert combined[FIRST_DIFF - 1] == level_z[FIRST_DIFF - 1]
    assert combined[FIRST_DIFF] == diff_z[FIRST_DIFF]
    # Otherwise the test couldn't tell the two apart
    assert level_z[FIRST_DIFF] != diff_z[FIRST_DIFF]


def test_blind_rows_are_minus_inf_and_nothing_else() -> None:
    score, runs = fitted()
    combined = score.scores(residuals(1), runs[1].throttle)
    np.testing.assert_array_equal(np.isneginf(combined), np.arange(N_STEPS) < FIRST_FORECAST)
    assert np.isfinite(combined[FIRST_FORECAST:]).all()


def test_scores_have_one_value_per_sample() -> None:
    score, runs = fitted()
    assert score.scores(residuals(2), runs[2].throttle).shape == (N_STEPS,)


def test_fit_scores_each_run_once() -> None:
    # Masked scoring is the slow part, about 8 minutes for 500 runs and 3 models. Calling it twice per run
    # would double that, and nothing else would show it
    runs = [Run(throttle=throttle(), sensors=np.zeros((N_STEPS, len(CHANNELS))), seed=s) for s in range(3)]
    calls = []

    def residual_fn(run: Run) -> np.ndarray:
        calls.append(run.seed)
        return residuals(run.seed)

    StandardizedScore.fit(runs, residual_fn, min_samples=1)
    assert sorted(calls) == [0, 1, 2]


def test_reference_seeds_were_never_trained_on() -> None:
    reference = set(REFERENCE_SEEDS)
    assert len(reference) == 500
    assert not reference & set(SPLITS["train"][:N_TRAIN_RUNS])
    assert not reference & set(SPLITS["calibrate"])
    assert not reference & set(SPLITS["val"])
