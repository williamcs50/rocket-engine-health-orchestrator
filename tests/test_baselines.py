import numpy as np
import pytest

from contracts import CHANNELS, N_STEPS, FaultLabel, Run
from evaluation.baselines import REDLINE_THRESHOLD, ZScoreBaseline, persistence_forecast, redline_scores
from generator.engine import REDLINES, make_fault, make_throttle, simulate

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
