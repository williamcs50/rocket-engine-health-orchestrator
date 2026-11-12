import numpy as np
import pytest

from contracts import CHANNELS, DT_S, N_STEPS, FaultLabel, Run
from generator.engine import ONSET_WINDOW, REDLINES, SUB_THRESHOLD_S, make_fault, make_throttle, simulate

def sec(t: float) -> int:
    """Seconds to a sample index."""
    return round(t / DT_S)

def col(run: Run, name: str) -> np.ndarray:
    """One sensor channel by name."""
    return run.sensors[:, CHANNELS.index(name)]

def test_healthy_run_has_no_fault_label() -> None:
    run = simulate(make_throttle(seed=0), seed=0, noise=False)
    assert run.fault is None

def test_fault_changes_nothing_before_onset() -> None:
    seed = 0
    thr = make_throttle(seed)
    fault = make_fault(seed)
    faulty = simulate(thr, seed, noise=False, fault=fault)
    twin = simulate(thr, seed, noise=False)
    onset = int(fault.onset_s / DT_S)
    np.testing.assert_array_equal(faulty.sensors[:onset], twin.sensors[:onset])

def test_bearing_wear_signature() -> None:
    pytest.fail("not written yet")

def test_fault_is_visible_against_its_twin() -> None:
    pytest.fail("not written yet")

def test_fault_stays_sub_threshold() -> None:
    pytest.fail("not written yet")

def test_fault_seeds_reproduce() -> None:
    pytest.fail("not written yet")
