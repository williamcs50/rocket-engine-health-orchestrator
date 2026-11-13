"""Scores a detector's per-sample scores against the evaluation rules in evaluation/budget.py."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from contracts import DT_S, FaultLabel, Run
from evaluation.budget import CATCH_WINDOW_S, TRANSIENT_S
from generator.splits import SPLITS, fault_for, make_run


@dataclass(frozen=True)
class RunResult:
    """How one run scored at one threshold."""

    false_alarm: bool  # healthy runs only. Counts toward the budget
    early_alarm: bool  # faulted run alarmed before onset
    transient_false_alarm: bool
    caught: bool  # within the catch window, with no early alarm
    delay_s: float | None  # only when caught
    events: int


def threshold_at_budget(peaks: Sequence[float], budget: float) -> float:
    """Lowest threshold at which at most `budget` of healthy runs alarm (alarm = score above it)."""
    peaks = np.sort(np.asarray(peaks, dtype=float))
    allowed = int(budget * len(peaks) + 1e-9)  # runs allowed to alarm. 1e-9 guards float rounding
    return float(peaks[len(peaks) - allowed - 1])


def first_alarm_s(scores: np.ndarray, threshold: float) -> float | None:
    """Time of the first sample above threshold, or None if the run never alarms."""
    hits = np.flatnonzero(scores > threshold)
    return float(hits[0] * DT_S) if hits.size else None


def alarm_events(scores: np.ndarray, threshold: float) -> int:
    """Number of separate alarm episodes: each quiet-to-alarm switch starts one."""
    alarm = scores > threshold
    return int(alarm[0]) + int((alarm[1:] & ~alarm[:-1]).sum())


def transient_mask(throttle: np.ndarray) -> np.ndarray:
    """True for samples within TRANSIENT_S after any commanded throttle change."""
    n = len(throttle)
    changed = np.zeros(n, dtype=bool)
    changed[1:] = np.diff(throttle) != 0
    last_change = np.maximum.accumulate(np.where(changed, np.arange(n), -n))
    return np.arange(n) - last_change < round(TRANSIENT_S / DT_S)


def score_run(scores: np.ndarray, throttle: np.ndarray, threshold: float, fault: FaultLabel | None) -> RunResult:
    """Apply the logged rules to one run's per-sample scores."""
    alarm = scores > threshold
    t = np.arange(len(alarm)) * DT_S
    events = alarm_events(scores, threshold)
    false = alarm if fault is None else alarm & (t < fault.onset_s)
    transient = bool((false & transient_mask(throttle)).any())

    if fault is None:
        return RunResult(bool(alarm.any()), False, transient, False, None, events)

    early = bool(false.any())
    after = np.flatnonzero(alarm & (t >= fault.onset_s))
    delay = float(t[after[0]] - fault.onset_s) if after.size else None
    caught = not early and delay is not None and delay <= CATCH_WINDOW_S
    return RunResult(False, early, transient, caught, delay if caught else None, events)


@dataclass(frozen=True)
class Report:
    """One detector's results on one split, at one threshold."""

    recall: float  # share of faulted runs caught
    mean_delay_s: float | None  # over caught runs. None when nothing was caught
    false_alarm_rate: float  # share of healthy runs with any alarm
    transient_false_alarm_rate: float
    early_alarm_rate: float  # share of faulted runs
    events_per_run: float  # per healthy run


def healthy_peaks(score_fn: Callable[[Run], np.ndarray], split: str) -> np.ndarray:
    """Each healthy run's highest score in a split. Thresholds come from these."""
    seeds = [seed for seed in SPLITS[split] if fault_for(split, seed) is None]
    return np.array([score_fn(make_run(split, seed)).max() for seed in seeds])


def evaluate(detectors: dict[str, tuple[Callable[[Run], np.ndarray], float]], split: str) -> dict[str, Report]:
    """Score every detector on every run of a split. Each run is simulated once and shared."""
    results: dict[str, list[tuple[bool, RunResult]]] = {name: [] for name in detectors}
    for seed in SPLITS[split]:
        run = make_run(split, seed)
        for name, (score_fn, threshold) in detectors.items():
            outcome = score_run(score_fn(run), run.throttle, threshold, run.fault)
            results[name].append((run.fault is None, outcome))
    return {name: _report(rows) for name, rows in results.items()}


def _report(rows: list[tuple[bool, RunResult]]) -> Report:
    """Summarize one detector's per-run results."""
    healthy = [r for is_healthy, r in rows if is_healthy]
    faulted = [r for is_healthy, r in rows if not is_healthy]
    delays = [r.delay_s for r in faulted if r.caught]
    return Report(
        recall=float(np.mean([r.caught for r in faulted])),
        mean_delay_s=float(np.mean(delays)) if delays else None,
        false_alarm_rate=float(np.mean([r.false_alarm for r in healthy])),
        transient_false_alarm_rate=float(np.mean([r.transient_false_alarm for r in healthy])),
        early_alarm_rate=float(np.mean([r.early_alarm for r in faulted])),
        events_per_run=float(np.mean([r.events for r in healthy])),
    )
