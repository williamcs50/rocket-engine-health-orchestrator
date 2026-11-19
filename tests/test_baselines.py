import numpy as np
import pytest

from contracts import CHANNELS, N_STEPS, FaultLabel, Run
from evaluation.baselines import (
    REDLINE_THRESHOLD,
    ConditionedZScore,
    ZScoreBaseline,
    persistence_forecast,
    redline_scores,
)
from evaluation.profile import N_PHASES, RAMP_BINS, bin_index, unpack
from generator.engine import REDLINES, make_fault, make_throttle, simulate
from generator.splits import SPLITS, fault_for, make_run

def run_with(sensors: np.ndarray, fault: FaultLabel | None = None) -> Run:
    """A Run around hand-built sensors. Throttle doesn't matter to these baselines."""
    return Run(throttle=np.zeros(N_STEPS), sensors=sensors, seed=0, fault=fault)

def constant_run(value: float) -> Run:
    """Every channel at `value` for the whole run."""
    return run_with(np.full((N_STEPS, len(CHANNELS)), value))

def test_redline_threshold_is_fixed() -> None:
    assert REDLINE_THRESHOLD == 1.0

def test_redline_scores_cross_one_only_where_a_channel_crosses() -> None:
    limits = np.array([REDLINES[name] for name in CHANNELS])
    sensors = np.tile(0.5 * limits, (N_STEPS, 1))
    sensors[3000, CHANNELS.index("bearing_temp")] = 1.01 * REDLINES["bearing_temp"]
    scores = redline_scores(run_with(sensors))
    assert scores.shape == (N_STEPS,)
    assert scores[3000] > REDLINE_THRESHOLD
    assert np.delete(scores, 3000).max() < REDLINE_THRESHOLD

def test_redline_quiet_on_a_healthy_run() -> None:
    run = simulate(make_throttle(0), 0)
    assert redline_scores(run).max() < REDLINE_THRESHOLD

def test_zscore_learns_mean_and_std_from_fit_runs() -> None:
    baseline = ZScoreBaseline.fit([constant_run(1.0), constant_run(3.0)])
    np.testing.assert_allclose(baseline.mean, 2.0)
    np.testing.assert_allclose(baseline.std, 1.0)
    scores = baseline.scores(constant_run(4.0))
    assert scores.shape == (N_STEPS,)
    np.testing.assert_allclose(scores, 2.0)  # 4 is two sigma above a mean of 2
    np.testing.assert_allclose(baseline.scores(constant_run(0.0)), 2.0)  # below counts too

def test_zscore_refuses_to_fit_on_faulted_runs() -> None:
    faulty = simulate(make_throttle(1), 1, fault=make_fault(1))
    with pytest.raises(ValueError):
        ZScoreBaseline.fit([constant_run(1.0), faulty])

def test_zscore_fits_from_a_generator() -> None:
    runs = (constant_run(v) for v in (1.0, 3.0))  # one run at a time, never a list
    np.testing.assert_allclose(ZScoreBaseline.fit(runs).mean, 2.0)

def test_persistence_forecasts_the_value_from_horizon_steps_back() -> None:
    sensors = np.arange(N_STEPS * len(CHANNELS), dtype=float).reshape(N_STEPS, len(CHANNELS))
    horizon = 10
    forecast = persistence_forecast(sensors, horizon)
    assert forecast.shape == sensors.shape
    np.testing.assert_array_equal(forecast[horizon:], sensors[:-horizon])

# Conditioned z score: the fair baseline for the Nov 20 test table. D4's ZScoreBaseline stays unchanged

def test_conditioned_zscore_scores_one_value_per_sample() -> None:
    runs = [make_run("train", s) for s in SPLITS["train"][:2]]
    assert ConditionedZScore.fit(runs).scores(runs[0]).shape == (N_STEPS,)

def test_conditioned_zscore_combines_channels_like_d4() -> None:
    # Two runs at 1.0 and 3.0 give every channel a mean of 2 and a std of 1. A run sitting at +4 sigma on one
    # channel and -6 sigma on another must score 6 everywhere: the largest absolute z, as D4 does
    z = ConditionedZScore.fit([constant_run(1.0), constant_run(3.0)])
    sensors = np.full((N_STEPS, len(CHANNELS)), 2.0)
    sensors[:, 3], sensors[:, 5] = 6.0, -4.0
    np.testing.assert_allclose(z.scores(run_with(sensors)), 6.0)

def test_conditioning_removes_the_startup_excess() -> None:
    # D4 compares every sample with one whole-run mean, so the startup ramp sits many sigma away. Conditioned on
    # the throttle profile, the ramp should look like the rest of the run. Medians over thousands of samples
    # barely move between runs, so the test is stable. Fit on 30 runs for speed, scored on 3 held-out runs
    fit_runs = [make_run("train", s) for s in SPLITS["train"][:30]]
    d4, conditioned = ZScoreBaseline.fit(fit_runs), ConditionedZScore.fit(fit_runs)
    held_out = [make_run("train", s) for s in SPLITS["train"][100:103]]

    def ramp_to_steady(detector: ZScoreBaseline | ConditionedZScore) -> float:
        ramp, steady = [], []
        for run in held_out:
            phase, _, _ = unpack(bin_index(run.throttle))
            scores = detector.scores(run)
            ramp.append(scores[phase < RAMP_BINS])
            steady.append(scores[phase == N_PHASES - 1])
        return float(np.median(np.concatenate(ramp)) / np.median(np.concatenate(steady)))

    assert ramp_to_steady(d4) > 2
    assert 0.5 < ramp_to_steady(conditioned) < 2

def test_conditioned_zscore_rejects_faulted_runs() -> None:
    faulty_seed = next(s for s in SPLITS["val"] if fault_for("val", s) is not None)
    with pytest.raises(ValueError):
        ConditionedZScore.fit([constant_run(1.0), make_run("val", faulty_seed)])
