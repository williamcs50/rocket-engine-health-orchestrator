import numpy as np
import pytest

from contracts import DT_S, N_STEPS, FaultLabel
from evaluation.baselines import REDLINE_THRESHOLD, ZScoreBaseline, redline_scores
from evaluation.budget import CATCH_WINDOW_S, FA_BUDGET, REPORT_SPLIT, THRESHOLD_SPLIT, TRANSIENT_S
from evaluation.harness import (
    alarm_events, evaluate, first_alarm_s, healthy_peaks, score_run, threshold_at_budget,
)
from generator.engine import WEAR_CHANNELS
from generator.splits import SPLITS, make_run

THRESHOLD = 1.0
ONSET = 20.0
FAULT = FaultLabel(kind="bearing_wear", onset_s=ONSET, rate=0.05, channels=WEAR_CHANNELS)
FLAT = np.full(N_STEPS, 0.8)  # throttle held constant, so there are no commanded transients

def sec(t: float) -> int:
    """Seconds to a sample index."""
    return round(t / DT_S)

def alarms_at(*times: float, width: float = 0.5) -> np.ndarray:
    """Per-sample scores: quiet everywhere, above THRESHOLD for `width` seconds at each time."""
    scores = np.zeros(N_STEPS)
    for t in times:
        scores[sec(t):sec(t + width)] = 2.0
    return scores

@pytest.mark.parametrize("budget", [0.01, 0.02, 0.05])
def test_threshold_meets_budget_on_healthy_peaks(budget: float) -> None:
    peaks = np.random.default_rng(0).normal(size=500)  # one peak score per healthy run
    threshold = threshold_at_budget(peaks, budget)
    rate = np.mean(peaks > threshold)
    assert rate <= budget, f"{rate:.1%} of healthy runs alarm, budget {budget:.0%}"
    assert rate >= budget - 1 / len(peaks), f"threshold too strict: only {rate:.1%} alarm"

def test_first_alarm_time() -> None:
    assert first_alarm_s(alarms_at(23.0), THRESHOLD) == pytest.approx(23.0)
    assert first_alarm_s(np.zeros(N_STEPS), THRESHOLD) is None

def test_alarm_events_count_separate_episodes() -> None:
    assert alarm_events(alarms_at(10.0, 20.0, 30.0), THRESHOLD) == 3
    assert alarm_events(alarms_at(10.0, width=5.0), THRESHOLD) == 1
    assert alarm_events(np.zeros(N_STEPS), THRESHOLD) == 0

def test_healthy_run_with_any_alarm_is_a_false_alarm() -> None:
    assert score_run(alarms_at(45.0), FLAT, THRESHOLD, fault=None).false_alarm
    assert not score_run(np.zeros(N_STEPS), FLAT, THRESHOLD, fault=None).false_alarm

def test_alarm_within_window_is_a_catch_with_delay() -> None:
    result = score_run(alarms_at(ONSET + 3.0), FLAT, THRESHOLD, fault=FAULT)
    assert result.caught
    assert result.delay_s == pytest.approx(3.0)
    assert not result.early_alarm and not result.false_alarm

def test_alarm_after_window_is_a_miss() -> None:
    result = score_run(alarms_at(ONSET + CATCH_WINDOW_S + 1.0), FLAT, THRESHOLD, fault=FAULT)
    assert not result.caught
    assert result.delay_s is None

def test_early_alarm_disqualifies_catch_but_stays_out_of_healthy_rate() -> None:
    result = score_run(alarms_at(ONSET - 10.0, ONSET + 3.0), FLAT, THRESHOLD, fault=FAULT)
    assert result.early_alarm
    assert not result.caught  # alarming before the fault exists earns no credit
    assert not result.false_alarm  # the 2% rate counts healthy runs only

def test_false_alarm_during_throttle_step_is_flagged_as_transient() -> None:
    step = np.where(np.arange(N_STEPS) < sec(30.0), 0.6, 0.9)
    during = score_run(alarms_at(30.0 + TRANSIENT_S / 2), step, THRESHOLD, fault=None)
    away = score_run(alarms_at(10.0), step, THRESHOLD, fault=None)
    assert during.false_alarm and during.transient_false_alarm
    assert away.false_alarm and not away.transient_false_alarm

@pytest.mark.slow
def test_exit_harness_reports_both_baselines_on_val() -> None:
    # D4 exit test: all three metrics for both baselines on val, and redlines miss most faults early.
    zscore = ZScoreBaseline.fit(make_run("train", seed) for seed in SPLITS["train"])
    zscore_threshold = threshold_at_budget(healthy_peaks(zscore.scores, THRESHOLD_SPLIT), FA_BUDGET)
    reports = evaluate(
        {"redline": (redline_scores, REDLINE_THRESHOLD), "zscore": (zscore.scores, zscore_threshold)},
        REPORT_SPLIT,
    )

    for name, report in reports.items():
        print(f"\n{name}: {report}")  # shown with pytest -s, so the numbers land in the build log
        assert 0.0 <= report.recall <= 1.0
        assert 0.0 <= report.false_alarm_rate <= 1.0
        assert 0.0 <= report.transient_false_alarm_rate <= report.false_alarm_rate
        assert report.mean_delay_s is None or 0.0 <= report.mean_delay_s <= CATCH_WINDOW_S

    redline = reports["redline"]
    assert redline.recall < 0.5, f"redlines caught {redline.recall:.0%} of faults early: faults too obvious, retune"
