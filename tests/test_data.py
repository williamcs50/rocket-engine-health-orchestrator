import numpy as np
import pytest

from contracts import CHANNELS, N_STEPS, WINDOW, Run
from generator.engine import SENSOR_NOISE
from generator.splits import SPLITS, fault_for, make_run
from model.data import Normalizer, make_windows

def constant_run(value: float) -> Run:
    return Run(throttle=np.zeros(N_STEPS), sensors=np.full((N_STEPS, len(CHANNELS)), value), seed=0)

def ramps() -> tuple[np.ndarray, np.ndarray]:
    """Sensors and throttle where every value is its own index, so any misalignment shows."""
    sensors = np.arange(N_STEPS * len(CHANNELS), dtype=float).reshape(N_STEPS, len(CHANNELS))
    return sensors, np.arange(N_STEPS, dtype=float)

def test_fit_computes_pooled_stats() -> None:
    norm = Normalizer.fit([constant_run(1.0), constant_run(3.0)])
    np.testing.assert_allclose(norm.mean, 2.0)
    np.testing.assert_allclose(norm.std, 1.0)
    noise = np.array([SENSOR_NOISE[name] for name in CHANNELS])
    # Same formula as evaluation/checks.py
    np.testing.assert_allclose(norm.floor, noise**2 / norm.std**2)

def test_fit_rejects_faulted_runs() -> None:
    faulty_seed = next(s for s in SPLITS["val"] if fault_for("val", s) is not None)
    with pytest.raises(ValueError):
        Normalizer.fit([constant_run(1.0), make_run("val", faulty_seed)])

@pytest.mark.parametrize("level", [2.0, 0.1, 1.7])
def test_fit_rejects_zero_spread(level: float) -> None:
    # Std 0 would turn normalization into a silent divide by zero. At 0.1 and 1.7, rounding leaves
    # a tiny nonzero std instead of exactly 0, which must still count as flat
    with pytest.raises(ValueError):
        Normalizer.fit([constant_run(level), constant_run(level)])

def test_fit_rejects_no_runs() -> None:
    with pytest.raises(ValueError):
        Normalizer.fit([])

def test_normalizer_arrays_are_read_only() -> None:
    norm = Normalizer.fit([constant_run(1.0), constant_run(3.0)])
    for arr in (norm.mean, norm.std, norm.floor):
        with pytest.raises(ValueError):
            arr[0] = 0.0

def test_transform_uses_the_fitted_stats() -> None:
    norm = Normalizer.fit([constant_run(1.0), constant_run(3.0)])
    np.testing.assert_allclose(norm.transform(np.full((N_STEPS, len(CHANNELS)), 4.0)), 2.0)

@pytest.mark.parametrize("horizon", [1, 10, 20])
def test_windows_line_up_with_their_targets(horizon: int) -> None:
    sensors, throttle = ramps()
    windows, throttle_hist, targets = make_windows(sensors, throttle, horizon)
    # Every window that has a target inside the run
    n = N_STEPS - WINDOW - horizon + 1
    assert windows.shape == (n, len(CHANNELS), WINDOW)
    assert throttle_hist.shape == (n, WINDOW + horizon)
    assert targets.shape == (n, len(CHANNELS))
    # Window i covers samples i to i + WINDOW - 1. Its target is horizon samples after the last one.
    np.testing.assert_array_equal(windows[0], sensors[:WINDOW].T)
    np.testing.assert_array_equal(targets[0], sensors[WINDOW - 1 + horizon])
    # The last window's target is the run's last sample
    np.testing.assert_array_equal(targets[-1], sensors[-1])
    # Commands are known up to the target time, so a step inside the forecast gap isn't a surprise
    np.testing.assert_array_equal(throttle_hist[0], throttle[: WINDOW + horizon])
    np.testing.assert_array_equal(throttle_hist[-1], throttle[-(WINDOW + horizon):])

def test_windows_are_views_not_copies() -> None:
    sensors, throttle = ramps()
    windows, throttle_hist, _ = make_windows(sensors, throttle, 10)
    # 5,891 copied windows per run would be about 38 MB
    assert np.shares_memory(windows, sensors)
    assert np.shares_memory(throttle_hist, throttle)

def test_stride_keeps_every_nth_window() -> None:
    sensors, throttle = ramps()
    every, _, every_targets = make_windows(sensors, throttle, 10)
    strided, _, strided_targets = make_windows(sensors, throttle, 10, stride=10)
    np.testing.assert_array_equal(strided, every[::10])
    np.testing.assert_array_equal(strided_targets, every_targets[::10])

def test_fit_accepts_real_train_runs() -> None:
    # The flat-channel check must not be so loose that it rejects real data. Mixture ratio is the
    # flattest real channel, with a std about 0.5% of its level, still thousands of times above the 1e-6 tolerance
    norm = Normalizer.fit(make_run("train", seed) for seed in SPLITS["train"][:3])
    assert np.all(norm.std > 1e-3 * np.abs(norm.mean))
    assert np.all(np.isfinite(norm.floor))

@pytest.mark.parametrize("horizon, stride", [(0, 1), (-1, 1), (10, 0), (10, -1)])
def test_make_windows_rejects_bad_horizon_or_stride(horizon: int, stride: int) -> None:
    sensors, throttle = ramps()
    with pytest.raises(ValueError):
        make_windows(sensors, throttle, horizon, stride=stride)
