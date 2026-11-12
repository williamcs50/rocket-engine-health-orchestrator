import numpy as np
import pytest

from contracts import CHANNELS, DT_S, FaultLabel, Run
from generator.engine import (
    ONSET_WINDOW, REDLINES, SENSOR_NOISE, SUB_THRESHOLD_S,
    make_fault, make_throttle, simulate,
)

# Visibility floors, in sensor-noise sigmas per sample. Bearing temp is the fault's primary
# channel and must be clear on its own. Vibration is a supporting channel, subtle alone, because
# its redline margin (about 0.14 g) is under 5 sigma.
VISIBLE_SIGMAS = {
    "bearing_temp": 5.0,
    "turbopump_vibration": 2.0,
}
SUB_THRESHOLD_SEEDS = range(50)
MIN_QUIET_FRACTION = 0.8

def sec(t: float) -> int:
    """Seconds to a sample index."""
    return round(t / DT_S)

def col(run: Run, name: str) -> np.ndarray:
    """One sensor channel by name."""
    return run.sensors[:, CHANNELS.index(name)]

def faulty_and_twin(seed: int) -> tuple[Run, Run, FaultLabel]:
    """A faulted run and its healthy twin: same seed, same throttle, noise off."""
    thr, fault = make_throttle(seed), make_fault(seed)
    return simulate(thr, seed, noise=False, fault=fault), simulate(thr, seed, noise=False), fault

def test_healthy_run_has_no_fault_label() -> None:
    run = simulate(make_throttle(seed=0), seed=0, noise=False)
    assert run.fault is None

@pytest.mark.parametrize("seed", range(5))
def test_fault_changes_nothing_before_onset(seed: int) -> None:
    faulty, twin, fault = faulty_and_twin(seed)
    assert faulty.fault == fault  # the label travels with the run
    assert isinstance(faulty.fault, FaultLabel)
    before = sec(fault.onset_s)
    np.testing.assert_array_equal(faulty.sensors[:before], twin.sensors[:before])

@pytest.mark.parametrize("seed", range(5))
def test_bearing_wear_signature(seed: int) -> None:
    faulty, twin, fault = faulty_and_twin(seed)
    at = sec(fault.onset_s + SUB_THRESHOLD_S)

    def delta(name: str) -> float:
        """Faulty minus healthy twin for one channel, at the check time."""
        return float(col(faulty, name)[at] - col(twin, name)[at])

    assert delta("bearing_temp") > 0, "bearing did not warm"
    assert delta("turbopump_vibration") > 0, "vibration did not rise"
    assert delta("chamber_pressure") < 0, "chamber pressure did not ease down"

@pytest.mark.parametrize("seed", range(5))
def test_fault_is_visible_against_its_twin(seed: int) -> None:
    faulty, twin, fault = faulty_and_twin(seed)
    at = sec(fault.onset_s + SUB_THRESHOLD_S)

    for name, sigmas in VISIBLE_SIGMAS.items():
        delta = col(faulty, name)[at] - col(twin, name)[at]
        floor = sigmas * SENSOR_NOISE[name]
        assert delta > floor, f"{name} moved {delta:.3g}, needs > {floor:.3g} ({sigmas} sigma)"

def test_fault_stays_sub_threshold() -> None:
    quiet = 0
    for seed in SUB_THRESHOLD_SEEDS:
        fault = make_fault(seed)
        run = simulate(make_throttle(seed), seed, noise=True, fault=fault)
        window = run.sensors[: sec(fault.onset_s + SUB_THRESHOLD_S)]
        crossed = any(window[:, i].max() >= REDLINES[name] for i, name in enumerate(CHANNELS))
        quiet += not crossed
    fraction = quiet / len(SUB_THRESHOLD_SEEDS)
    assert fraction >= MIN_QUIET_FRACTION, f"only {fraction:.0%} of fault runs stayed under redlines"

def test_fault_seeds_reproduce() -> None:
    assert make_fault(3) == make_fault(3)
    assert make_fault(3) != make_fault(4)
    for seed in range(20):
        fault = make_fault(seed)
        assert ONSET_WINDOW[0] <= fault.onset_s <= ONSET_WINDOW[1]
        assert fault.rate > 0
        assert fault.kind == "bearing_wear"
