from collections.abc import Sequence

import numpy as np
import pytest

from contracts import DT_S, N_STEPS
from evaluation.ablation import (
    BAR_RECALL_DROP, BAR_SHARE_MARGIN, BAR_STEP_DROP, NOCMD_MODELS, ablation_verdict, pooled_share, share,
    step_ratios, step_verdict, step_windows, transient_counts,
)
from evaluation.budget import TRANSIENT_S
from evaluation.gate1 import MODELS
from evaluation.harness import RunResult
from generator.engine import RAMP_UP, SHUTDOWN_WINDOW, make_throttle


def result(false_alarm: bool = False, transient: bool = False, caught: bool = False) -> RunResult:
    return RunResult(false_alarm, False, transient, caught, 1.0 if caught else None, int(false_alarm))


# The bar, as pre-registered on 2026-11-25

def test_bar_is_as_pre_registered() -> None:
    assert BAR_SHARE_MARGIN == 0.20
    assert BAR_RECALL_DROP == 0.03


def test_ablation_models_match_d7_seed_for_seed() -> None:
    assert NOCMD_MODELS == tuple(m.replace("_cmd_", "_nocmd_") for m in MODELS)


# Counting and shares

def test_transient_counts_only_healthy_false_alarms() -> None:
    rows = [
        (True, result(false_alarm=True, transient=True)),
        (True, result(false_alarm=True)),
        (True, result()),
        # A faulted run never counts toward false alarms, even if it carries the transient flag
        (False, result(transient=True, caught=True)),
    ]
    assert transient_counts(rows) == (1, 2)


def test_share_and_pooled_share() -> None:
    assert share(3, 4) == 0.75
    assert share(0, 0) is None
    # Every seed's transient runs over every seed's false alarm runs, not the mean of the shares
    assert pooled_share([(1, 1), (1, 3)]) == pytest.approx(0.5)
    assert pooled_share([(0, 0), (0, 0)]) is None


# The verdict, at its edges

def verdict(
    cmd: Sequence[tuple[int, int]],
    nocmd: Sequence[tuple[int, int]],
    cmd_recall: Sequence[float] = (0.5, 0.5, 0.5),
    nocmd_recall: Sequence[float] = (0.5, 0.5, 0.5),
) -> bool:
    return ablation_verdict(list(cmd), list(nocmd), list(cmd_recall), list(nocmd_recall))[0]


def test_holds_with_a_clear_drop() -> None:
    assert verdict([(5, 10)] * 3, [(10, 10)] * 3)


def test_a_drop_of_exactly_20_points_holds() -> None:
    # Pooled 24 of 30 is 80%, against 100%
    assert verdict([(8, 10)] * 3, [(10, 10)] * 3)


def test_a_drop_just_under_20_points_fails() -> None:
    assert not verdict([(9, 10), (8, 10), (8, 10)], [(10, 10)] * 3)


def test_one_seed_not_lower_fails() -> None:
    # Pooled the drop is large, but seed 2 is level with its ablation
    assert not verdict([(0, 10), (0, 10), (10, 10)], [(10, 10)] * 3)


def test_a_model_with_no_false_alarms_fails() -> None:
    assert not verdict([(0, 0), (2, 10), (2, 10)], [(10, 10)] * 3)
    assert not verdict([(2, 10)] * 3, [(0, 0), (10, 10), (10, 10)])


def test_recall_exactly_3_points_lower_holds() -> None:
    assert verdict([(5, 10)] * 3, [(10, 10)] * 3, cmd_recall=(0.47,) * 3, nocmd_recall=(0.50,) * 3)


def test_recall_more_than_3_points_lower_fails() -> None:
    assert not verdict([(5, 10)] * 3, [(10, 10)] * 3, cmd_recall=(0.46,) * 3, nocmd_recall=(0.50,) * 3)


def test_conditions_read_as_text() -> None:
    _, conditions = ablation_verdict([(5, 10)] * 3, [(10, 10)] * 3, [0.5] * 3, [0.5] * 3)
    assert len(conditions) == 2
    assert all(isinstance(text, str) and ok for text, ok in conditions.items())


# The mid-run step test, amended on 2026-11-25 before the full run

def i(t: float) -> int:
    return round(t / DT_S)


def throttle_with_one_step() -> np.ndarray:
    """Ramp to 0.8 by 3 s, a step to 0.6 at 20 s over 0.2 s, shutdown from 55 s to 58 s."""
    t = np.arange(N_STEPS) * DT_S
    return np.interp(t, [0.0, 3.0, 20.0, 20.2, 55.0, 58.0, 60.0], [0.0, 0.8, 0.8, 0.6, 0.6, 0.0, 0.0])


def test_step_bar_is_as_pre_registered() -> None:
    assert BAR_STEP_DROP == 0.10


def test_step_windows_skip_startup_and_shutdown() -> None:
    windows = step_windows(throttle_with_one_step())
    assert len(windows) == 1
    # From the step's first changing sample to 5 s after its last, the same rule transient_mask uses
    assert windows[0].start == i(20.0) + 1
    assert windows[0].stop == i(20.2) + i(TRANSIENT_S)


@pytest.mark.parametrize("seed", range(20))
def test_every_generated_run_has_one_or_two_steps_found(seed: int) -> None:
    throttle = make_throttle(seed)
    windows = step_windows(throttle)
    assert len(windows) in (1, 2)
    for w in windows:
        assert w.start * DT_S > RAMP_UP[1]
        # The only changes inside a window are the step's own slew, so neither startup nor shutdown gets in
        moving = np.diff(throttle[w.start - 1:w.stop]) != 0
        assert moving[0]
        assert not moving[np.argmin(moving):].any()
        assert w.start * DT_S < SHUTDOWN_WINDOW[0]


def test_a_late_step_window_stops_where_shutdown_starts() -> None:
    # A step at 50 s, then shutdown from 54 s, inside the step's 5 s window
    t = np.arange(N_STEPS) * DT_S
    throttle = np.interp(t, [0.0, 3.0, 50.0, 50.2, 54.0, 57.0, 60.0], [0.0, 0.8, 0.8, 0.6, 0.6, 0.0, 0.0])
    (w,) = step_windows(throttle)
    assert w.stop == i(54.0) + 1


def test_step_ratios_are_the_peak_inside_each_window_over_the_threshold() -> None:
    throttle = throttle_with_one_step()
    s = np.zeros(N_STEPS)
    s[i(22.0)] = 5.0
    # Outside the step window, so these must not count
    s[i(2.0)] = 100.0
    s[i(40.0)] = 100.0
    assert step_ratios(s, throttle, threshold=10.0) == [pytest.approx(0.5)]


def test_step_ratios_refuse_a_threshold_that_isnt_positive() -> None:
    with pytest.raises(ValueError):
        step_ratios(np.zeros(N_STEPS), throttle_with_one_step(), threshold=0.0)


def test_step_verdict_edges() -> None:
    # Exactly 10% lower on every seed holds
    assert step_verdict([0.45] * 3, [0.50] * 3)[0]
    # Just under 10% lower fails
    assert not step_verdict([0.46, 0.45, 0.45], [0.50] * 3)[0]
    # One seed not lower fails
    assert not step_verdict([0.30, 0.30, 0.50], [0.50] * 3)[0]
    # An ablation median at or below 0 has no "10% lower", so it fails
    assert not step_verdict([-0.5] * 3, [0.0] * 3)[0]
    # No steps on a model fails
    assert not step_verdict([None, 0.3, 0.3], [0.5] * 3)[0]
