import numpy as np
import pytest

from contracts import N_STEPS
from evaluation.diagnose import (
    BAR_SPEARMAN, SWING_END, SWING_START, alarm_engines, diagnostic_verdict, rms_offset, spearman, swing,
)


# The window and bar, as pre-registered on 2026-11-24

def test_swing_window_is_21_09_to_50_seconds() -> None:
    assert (SWING_START, SWING_END) == (2109, 5000)
    assert BAR_SPEARMAN == 0.3


def test_swing_is_the_std_inside_the_window_only() -> None:
    s = np.zeros(N_STEPS)
    s[:109] = -np.inf
    # Outside the window, so it must not count
    s[1000] = 100.0
    s[5500] = 100.0
    s[SWING_START:SWING_END:2] = 1.0
    assert swing(s) == pytest.approx(0.5)


# The offset of one engine

def test_rms_offset() -> None:
    assert rms_offset(np.full(8, 3.0)) == pytest.approx(3.0)
    assert rms_offset(np.zeros(8)) == 0.0
    # Sign doesn't matter
    assert rms_offset(np.array([3.0, -3.0] * 4)) == pytest.approx(3.0)


# Spearman, from ranks, with no scipy

def test_spearman_is_one_for_any_increasing_relation() -> None:
    x = np.arange(10.0)
    assert spearman(x, x**3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)


def test_spearman_matches_a_hand_worked_example_with_a_tie() -> None:
    # Ranks of y are 1, 2, 3.5, 5, 3.5, so the Pearson correlation of the ranks is 8 / sqrt(10 * 9.5)
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([5.0, 6.0, 7.0, 8.0, 7.0])
    assert spearman(x, y) == pytest.approx(8 / np.sqrt(95))


def test_spearman_ignores_extreme_values() -> None:
    # The reason it was chosen: one huge value counts only as the highest rank
    x = np.arange(20.0)
    y = x.copy()
    y[-1] = 1e9
    assert spearman(x, y) == pytest.approx(1.0)


def test_spearman_refuses_mismatched_or_tiny_input() -> None:
    with pytest.raises(ValueError):
        spearman(np.arange(5.0), np.arange(4.0))
    with pytest.raises(ValueError):
        spearman(np.arange(2.0), np.arange(2.0))


# Engines that false-alarm on any seed, each counted once

def test_an_engine_alarming_on_any_seed_counts_once() -> None:
    alarms = {"m0": {10: True, 12: False, 14: False}, "m1": {10: True, 12: True, 14: False}, "m2": {10: False, 12: False, 14: False}}
    assert alarm_engines(alarms) == {10, 12}


# The verdict: both conditions must hold

def test_both_conditions_pass() -> None:
    passed, conditions = diagnostic_verdict([0.35, 0.4, 0.31], np.array([2.0, 2.1]), np.array([1.6, 1.7, 1.8]))
    assert passed
    assert list(conditions.values()) == [True, True]


def test_one_seed_under_the_bar_fails() -> None:
    passed, conditions = diagnostic_verdict([0.35, 0.29, 0.4], np.array([2.0]), np.array([1.6]))
    assert not passed
    assert list(conditions.values()) == [False, True]


def test_exactly_the_bar_passes() -> None:
    passed, _ = diagnostic_verdict([0.3, 0.3, 0.3], np.array([2.0]), np.array([1.6]))
    assert passed


def test_equal_medians_fail() -> None:
    # The false alarm engines must have a higher median, not an equal one
    passed, conditions = diagnostic_verdict([0.5, 0.5, 0.5], np.array([1.7, 1.9]), np.array([1.7, 1.9]))
    assert not passed
    assert list(conditions.values()) == [True, False]


def test_no_false_alarm_engines_fails_the_second_condition() -> None:
    passed, conditions = diagnostic_verdict([0.5, 0.5, 0.5], np.array([]), np.array([1.7]))
    assert not passed
    assert list(conditions.values()) == [True, False]
