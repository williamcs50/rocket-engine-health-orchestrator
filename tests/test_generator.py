import numpy as np
import pytest

from contracts import CHANNELS, N_STEPS, DT_S, Run
from generator.engine import LEVEL_MAX, REDLINES, make_throttle, simulate

SEEDS = range(5)

def sec(t: float) -> int:
    """Seconds to a sample index."""
    return round(t / DT_S)

def col(run: Run, name: str) -> np.ndarray:
    """One sensor channel by name."""
    return run.sensors[:, CHANNELS.index(name)]

def step_throttle(before: float = 0.6, after: float = 0.9, at: float = 30.0) -> np.ndarray:
    """Hand-built throttle: hold `before`, step to `after` at `at` seconds, hold to the end."""
    thr = np.full(N_STEPS, before)
    thr[sec(at):] = after
    return thr

@pytest.mark.parametrize("seed", SEEDS)
def test_throttle_profile_is_valid(seed: int) -> None:
    thr = make_throttle(seed)
    assert thr.shape == (N_STEPS,)
    assert thr[0] == 0.0 and thr[-1] == 0.0
    assert thr.min() >= 0.0 and thr.max() <= LEVEL_MAX

def test_run_has_contract_shape() -> None:
    thr = make_throttle(seed=0)
    run = simulate(thr, seed=0)
    assert isinstance(run, Run)
    np.testing.assert_array_equal(run.throttle, thr)
    assert run.seed == 0
    assert np.isfinite(run.sensors).all()

def test_same_seed_reproduces_run() -> None:
    np.testing.assert_array_equal(make_throttle(seed=1), make_throttle(seed=1))
    thr = make_throttle(seed=1)
    a, b = simulate(thr, seed=1), simulate(thr, seed=1)
    np.testing.assert_array_equal(a.sensors, b.sensors)

def test_different_seeds_differ() -> None:
    thr = make_throttle(seed=1)
    a, b = simulate(thr, seed=1, noise=False), simulate(thr, seed=2, noise=False)
    assert not np.array_equal(a.sensors, b.sensors)        # same command, different engine
    assert not np.array_equal(make_throttle(seed=1), make_throttle(seed=2))  # duty cycle varies too

def test_throttle_up_raises_pressure_speed_and_nozzle_temp() -> None:
    run = simulate(step_throttle(), seed=0, noise=False)
    before, settled = sec(30.0) - 1, sec(40.0)
    for name in ("chamber_pressure", "turbopump_speed", "nozzle_temp"):
        assert col(run, name)[settled] > col(run, name)[before], f"{name} did not rise"

@pytest.mark.parametrize("seed", SEEDS)
def test_coupled_channels_correlate(seed: int) -> None:
    run = simulate(make_throttle(seed=seed), seed=seed)
    pc, speed = col(run, "chamber_pressure"), col(run, "turbopump_speed")
    pairs = {
        "chamber_pressure vs turbopump_speed": (pc, speed),
        "throttle vs chamber_pressure": (run.throttle, pc),
        "throttle vs turbopump_speed": (run.throttle, speed),
    }
    for label, (x, y) in pairs.items():
        r = np.corrcoef(x, y)[0, 1]
        assert r > 0.9, f"{label}: r = {r:.3f}"

# A first-order lag covers 1 - exp(-t/tau) of a step after time t. "Under 50% after 0.05 s"
# needs tau > ~0.07 s, and "over 95% after 5 s" needs tau < ~1.7 s, so pick tau_N in 0.1-1.5 s.
def test_pump_speed_lags_throttle_step() -> None:
    speed = col(simulate(step_throttle(), seed=0, noise=False), "turbopump_speed")
    start, end = speed[sec(30.0) - 1], speed[-1]

    def covered(t: float) -> float:
        """Fraction of the step's eventual change in speed reached by time t."""
        return float((speed[sec(t)] - start) / (end - start))

    assert covered(30.05) < 0.5, "speed jumped: no lag"
    assert covered(35.0) > 0.95, "speed has not settled 5 s after the step"

@pytest.mark.parametrize("seed", SEEDS)
def test_nominal_run_stays_under_redlines(seed: int) -> None:
    assert set(REDLINES) == set(CHANNELS), f"REDLINES keys {sorted(REDLINES)} must match CHANNELS exactly"
    run = simulate(make_throttle(seed=seed), seed=seed)
    for name, limit in REDLINES.items():
        peak = col(run, name).max()
        assert peak < limit, f"{name} peaked at {peak:.3g}, redline {limit:.3g}"
  