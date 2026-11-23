import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest

from contracts import CHANNELS, DT_S, N_STEPS
from evaluation.baselines import ZScoreBaseline, redline_scores
from evaluation.budget import REPORT_SPLIT
from evaluation.final import attribution_row, check_thresholds, claim_test_run, gate1_verdict, lines, plan
from evaluation.harness import Report
from evaluation.gate1 import detector_scores
from evaluation.score import StandardizedScore, mean_residual
from generator.engine import WEAR_CHANNELS
from generator.splits import SPLITS, fault_for, make_run
from model.data import Normalizer
from model.forecaster import Forecaster, aggregate, residual_by_channel, to_verdict
from model.train import RUNS_DIR, TrainResult

# The model's first forecast. Rows before it have no forecast
FIRST_FORECAST = 109


def test_claim_writes_the_started_marker(tmp_path: Path) -> None:
    out = tmp_path / "test"
    claim_test_run(out)
    assert (out / "started").exists()


def test_a_second_claim_is_refused(tmp_path: Path) -> None:
    # The test split is opened once. A crash leaves the marker in place, so a rerun needs a logged, manual delete
    out = tmp_path / "test"
    claim_test_run(out)
    with pytest.raises(FileExistsError):
        claim_test_run(out)


def test_a_claim_is_refused_if_results_exist_without_the_marker(tmp_path: Path) -> None:
    # Results saved by any other route still count as the test split having been opened
    out = tmp_path / "test"
    out.mkdir()
    (out / "metrics.json").write_text("{}\n")
    with pytest.raises(FileExistsError):
        claim_test_run(out)


# Which split and folder a run uses

def test_a_dry_run_uses_val_and_saves_nothing() -> None:
    # A dry run must never claim runs/test or ask for the test split
    assert plan(dry_run=20) == ("val", None)


def test_the_real_run_uses_test_and_saves_to_runs_test() -> None:
    assert plan(dry_run=None) == ("test", RUNS_DIR / "test")


def test_test2_saves_to_runs_test2() -> None:
    assert plan(dry_run=None, split="test2") == ("test2", RUNS_DIR / "test2")


def test_a_test2_dry_run_still_uses_val_and_saves_nothing() -> None:
    assert plan(dry_run=20, split="test2") == ("val", None)


def test_an_unlocked_split_is_refused() -> None:
    # Only the two held-out splits get a real run
    with pytest.raises(ValueError):
        plan(dry_run=None, split="val")


# Which line is judged, per the decision log: mean on test (2026-11-19), std on test2 (2026-11-23)

def test_test_judges_the_mean_line() -> None:
    assert lines("test") == ("mean", "std")


def test_test2_judges_the_std_line() -> None:
    assert lines("test2") == ("std", "mean")


# The Gate 1 verdict on the judged line, unchanged since D6

def report(recall: float, false_alarm: float) -> Report:
    return Report(recall=recall, mean_delay_s=None, false_alarm_rate=false_alarm, transient_false_alarm_rate=0.0,
                  early_alarm_rate=0.0, events_per_run=0.0)


def reports(line: str, recalls: list[float], false_alarms: list[float]) -> dict:
    return {(f"m{i}/{line}", 0.02): report(r, f) for i, (r, f) in enumerate(zip(recalls, false_alarms))}


MODELS3 = ["m0", "m1", "m2"]


def test_test_numbers_for_the_std_line_pass() -> None:
    # The std line on test: 52.0%, 64.0%, 58.6% at 2.4%, 2.8%, 3.0%
    passed, _ = gate1_verdict(reports("std", [0.52, 0.64, 0.586], [0.024, 0.028, 0.030]), "std", MODELS3)
    assert passed


def test_one_seed_over_3_percent_fails() -> None:
    # The mean line on test: seed 1 at 3.2%
    passed, conditions = gate1_verdict(reports("mean", [0.176, 0.016, 0.028], [0.026, 0.032, 0.022]), "mean", MODELS3)
    assert not passed
    assert [ok for ok in conditions.values()] == [True, True, False]


def test_exactly_3_percent_passes_and_exactly_1_2_percent_recall_fails() -> None:
    # At most 3% false alarms, edge included. Every seed's recall strictly above 1.2%
    passed, _ = gate1_verdict(reports("std", [0.5, 0.5, 0.5], [0.03, 0.03, 0.03]), "std", MODELS3)
    assert passed
    passed, _ = gate1_verdict(reports("std", [0.5, 0.5, 0.012], [0.02, 0.02, 0.02]), "std", MODELS3)
    assert not passed


def test_a_mean_recall_of_exactly_5_percent_passes() -> None:
    # Floating point can put a mean of three shares a hair under 5%
    passed, _ = gate1_verdict(reports("std", [0.05, 0.05, 0.05], [0.02, 0.02, 0.02]), "std", MODELS3)
    assert passed


# The thresholds must match D8's before test is opened

def saved(tmp_path: Path, thresholds: dict) -> Path:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"thresholds": thresholds}))
    return path


def test_matching_thresholds_pass(tmp_path: Path) -> None:
    path = saved(tmp_path, {"a/mean": {"2%": 145.02, "1%": 157.9, "5%": 121.7}})
    check_thresholds({"a/mean": {0.02: 145.02, 0.01: 157.9, 0.05: 121.7}}, path)


def test_a_changed_threshold_is_refused(tmp_path: Path) -> None:
    # Everything is seeded, so a different threshold means something drifted since D8
    path = saved(tmp_path, {"a/mean": {"2%": 145.02, "1%": 157.9, "5%": 121.7}})
    with pytest.raises(ValueError):
        check_thresholds({"a/mean": {0.02: 145.5, 0.01: 157.9, 0.05: 121.7}}, path)


def test_a_missing_or_extra_detector_is_refused(tmp_path: Path) -> None:
    path = saved(tmp_path, {"a/mean": {"2%": 1.0}, "b/mean": {"2%": 2.0}})
    with pytest.raises(ValueError):
        check_thresholds({"a/mean": {0.02: 1.0}}, path)
    with pytest.raises(ValueError):
        check_thresholds({"a/mean": {0.02: 1.0}, "b/mean": {0.02: 2.0}, "c/mean": {0.02: 3.0}}, path)


def test_float_noise_is_not_a_change(tmp_path: Path) -> None:
    # Recomputing on another day can differ in the last digits. That isn't drift
    path = saved(tmp_path, {"a/mean": {"2%": 145.0208732496981}})
    check_thresholds({"a/mean": {0.02: 145.0208732496981 * (1 + 1e-12)}}, path)


# One attribution row per run and model

def wear_rise(onset_s: float, channels: tuple[str, ...]) -> np.ndarray:
    """A made-up (N_STEPS, 8) reading that rises on the given channels at onset, NaN before the first forecast."""
    v = np.zeros((N_STEPS, len(CHANNELS)))
    v[:FIRST_FORECAST] = np.nan
    for rank, name in enumerate(channels):
        v[round(onset_s / DT_S):, CHANNELS.index(name)] += 3.0 - rank
    return v


def test_a_faulted_row_scores_against_its_own_fault() -> None:
    seed = next(s for s in SPLITS["val"] if fault_for("val", s) is not None)
    onset = fault_for("val", seed).onset_s
    residual = wear_rise(onset, WEAR_CHANNELS)
    attention = wear_rise(onset, ("chamber_pressure", "nozzle_temp", "bearing_temp"))
    row = attribution_row("val", seed, "m", residual, attention, caught=True)
    assert row["seed"] == seed and row["model"] == "m"
    assert row["faulted"] is True and row["caught"] is True
    assert row["onset_s"] == onset
    assert row["residual_top"] == list(WEAR_CHANNELS)
    assert row["attention_top"] == ["chamber_pressure", "nozzle_temp", "bearing_temp"]
    assert row["residual_overlap"] == 1.0
    assert row["attention_overlap"] == pytest.approx(1 / 3)
    assert row["agreement"] == pytest.approx(1 / 3)


def test_a_healthy_row_borrows_the_next_onset_and_has_no_catch() -> None:
    seed = next(s for s in SPLITS["val"] if fault_for("val", s) is None)
    onset = fault_for("val", seed + 1).onset_s
    reading = wear_rise(onset, WEAR_CHANNELS)
    row = attribution_row("val", seed, "m", reading, reading, caught=None)
    assert row["faulted"] is False and row["caught"] is None
    assert row["onset_s"] == onset
    assert row["residual_overlap"] == 1.0


def test_rows_save_as_json() -> None:
    seed = next(s for s in SPLITS["val"] if fault_for("val", s) is not None)
    reading = wear_rise(fault_for("val", seed).onset_s, WEAR_CHANNELS)
    json.dumps(attribution_row("val", seed, "m", reading, reading, caught=False))


# The scoring shared with gate1

def test_detector_scores_matches_scoring_each_detector_directly() -> None:
    # gate1 and the test run both score through this, so test is scored exactly as val was
    runs = [make_run(REPORT_SPLIT, s) for s in SPLITS[REPORT_SPLIT][:2]]
    healthy = [r for r in runs if r.fault is None]
    normalizer = Normalizer.fit(healthy)
    trained = TrainResult(model=Forecaster().eval(), normalizer=normalizer, losses=[], config={"mask_prob": 0.25})
    std = StandardizedScore.fit(
        healthy, lambda run: mean_residual(residual_by_channel(trained.model, normalizer, run)), min_samples=1
    )
    zscore = ZScoreBaseline.fit(healthy)
    run = runs[1]

    scores, residuals = detector_scores(run, {"m": trained}, {"zscore": zscore}, {"m": std})

    r = residual_by_channel(trained.model, normalizer, run, masked=True)
    assert set(scores) == {"redline", "zscore", "m/mean", "m/max", "m/std"}
    np.testing.assert_array_equal(residuals["m"], r)
    np.testing.assert_array_equal(scores["redline"], redline_scores(run))
    np.testing.assert_array_equal(scores["zscore"], zscore.scores(run))
    np.testing.assert_array_equal(scores["m/mean"], aggregate(r, "mean"))
    np.testing.assert_array_equal(scores["m/max"], aggregate(r, "max"))
    np.testing.assert_array_equal(scores["m/std"], std.scores(mean_residual(r), run.throttle))


# main, with the slow parts stubbed out, so only the order of the guards is tested

def stub_slow_parts(monkeypatch: pytest.MonkeyPatch, calls: list[str]) -> None:
    """Replace model loading, fitting and scoring in evaluation.final, and record every guard that runs."""
    import evaluation.final as final

    monkeypatch.setattr(final, "load_run", lambda folder: None)
    monkeypatch.setattr(final, "fit_detectors", lambda models, limit: ({}, {}))
    monkeypatch.setattr(final, "calibrate_thresholds", lambda *args: ({}, {}))
    monkeypatch.setattr(final, "provenance", lambda: {})
    monkeypatch.setattr(final, "evaluate", lambda *args: ({}, {}, []))
    monkeypatch.setattr(final, "calibration_exit_test", lambda *args: {})
    monkeypatch.setattr(final, "summarize_attribution", lambda *args: {})
    monkeypatch.setattr(final, "print_results", lambda *args: None)
    monkeypatch.setattr(final, "gate1_verdict", lambda *args: (False, {}))
    monkeypatch.setattr(final, "claim_test_run", lambda out: calls.append("claim"))

    def opened(split: str = "test") -> object:
        calls.append(f"open {split}")
        return nullcontext()

    monkeypatch.setattr(final, "open_test_split", opened)


def test_a_dry_run_never_claims_or_opens_test(monkeypatch: pytest.MonkeyPatch) -> None:
    import evaluation.final as final

    calls: list[str] = []
    stub_slow_parts(monkeypatch, calls)
    monkeypatch.setattr(final, "check_thresholds", lambda *args: calls.append("check"))
    final.main(dry_run=2)
    assert calls == []


def test_a_threshold_mismatch_stops_before_the_claim(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The one run must not be used up by a run that was never going to be valid
    import evaluation.final as final

    calls: list[str] = []
    stub_slow_parts(monkeypatch, calls)
    monkeypatch.setattr(final, "plan", lambda dry_run, split="test": ("test", tmp_path / "test"))

    def mismatch(*args) -> None:
        raise ValueError("drifted")

    monkeypatch.setattr(final, "check_thresholds", mismatch)
    with pytest.raises(ValueError):
        final.main()
    assert calls == []


def test_the_real_run_checks_then_claims_then_opens_test(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import evaluation.final as final

    calls: list[str] = []
    stub_slow_parts(monkeypatch, calls)
    monkeypatch.setattr(final, "plan", lambda dry_run, split="test": ("test", tmp_path / "test"))
    monkeypatch.setattr(final, "check_thresholds", lambda *args: calls.append("check"))
    # Saving needs the folder the claim would have made
    (tmp_path / "test").mkdir()
    final.main()
    assert calls == ["check", "claim", "open test"]


def test_the_test2_run_checks_then_claims_then_opens_test2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import evaluation.final as final

    calls: list[str] = []
    stub_slow_parts(monkeypatch, calls)
    monkeypatch.setattr(final, "plan", lambda dry_run, split="test": (split, tmp_path / split))
    monkeypatch.setattr(final, "check_thresholds", lambda *args: calls.append("check"))
    monkeypatch.setattr(final, "gate1_verdict", lambda *args: (False, {}))
    (tmp_path / "test2").mkdir()
    final.main(split="test2")
    assert calls == ["check", "claim", "open test2"]


def test_the_verdict_flags_exactly_as_test2_scored() -> None:
    # The std score through to_verdict must be the same per-sample score detector_scores gave test2
    runs = [make_run(REPORT_SPLIT, s) for s in SPLITS[REPORT_SPLIT][:2]]
    healthy = [r for r in runs if r.fault is None]
    normalizer = Normalizer.fit(healthy)
    model = Forecaster().eval()
    std = StandardizedScore.fit(healthy, lambda run: mean_residual(residual_by_channel(model, normalizer, run)), min_samples=1)
    run = runs[1]
    peak = std.scores(mean_residual(residual_by_channel(model, normalizer, run)), run.throttle).max()
    np.testing.assert_array_equal(
        std.verdict_score(residual_by_channel(model, normalizer, run), run),
        std.scores(mean_residual(residual_by_channel(model, normalizer, run)), run.throttle),
    )
    assert to_verdict(model, normalizer, run, peak * 0.99, lambda p: 0.5, score=std.verdict_score).flag is True
    assert to_verdict(model, normalizer, run, peak * 1.01, lambda p: 0.5, score=std.verdict_score).flag is False
