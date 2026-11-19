import numpy as np
import pytest

from evaluation.calibration import PercentileConfidence, share_at_or_above, within_tolerance
from evaluation.harness import threshold_at_budget

# Small slack for float comparisons like 490 / 500 against 0.98
EPS = 1e-12


def healthy_peaks(n: int = 500, seed: int = 0) -> np.ndarray:
    """Distinct made-up peak scores, standing in for calibrate's 500 healthy runs."""
    return np.random.default_rng(seed).gamma(4.0, 50.0, size=n)


@pytest.mark.parametrize("budget", [0.01, 0.02, 0.05])
def test_flag_matches_confidence_at_every_budget(budget: float) -> None:
    # A run alarms when its peak is above the threshold. Its confidence must then be at least 1 minus the budget,
    # and below it otherwise. If these ever disagree, an operator sees "alarm, confidence 0.97"
    peaks = healthy_peaks()
    calibration = PercentileConfidence.fit(peaks)
    threshold = threshold_at_budget(peaks, budget)
    ordered = np.sort(peaks)
    # Every healthy peak, every midpoint between neighbours, and values beyond both ends
    candidates = np.concatenate([ordered, (ordered[:-1] + ordered[1:]) / 2, [ordered[0] - 1, ordered[-1] + 1]])
    for peak in candidates:
        flagged = peak > threshold
        confident = calibration.confidence(peak) >= 1 - budget - EPS
        assert flagged == confident, f"peak {peak:.3f}: flag {flagged}, confidence {calibration.confidence(peak):.3f}"


def test_confidence_is_the_share_strictly_below() -> None:
    calibration = PercentileConfidence.fit(np.array([10.0, 20.0, 30.0, 40.0]))
    # None below 5, none strictly below 10, one below 15, all four below 50
    np.testing.assert_allclose(calibration.confidence(np.array([5.0, 10.0, 15.0, 50.0])), [0.0, 0.0, 0.25, 1.0])


def test_confidence_rises_with_the_score() -> None:
    calibration = PercentileConfidence.fit(healthy_peaks())
    scores = np.linspace(0.0, 2000.0, 1000)
    confidence = calibration.confidence(scores)
    assert np.all(np.diff(confidence) >= 0)
    assert confidence.min() >= 0.0 and confidence.max() <= 1.0


@pytest.mark.parametrize("q", [0.8, 0.9, 0.95])
def test_exit_test_share_equals_the_false_alarm_rate(q: float) -> None:
    # The exit test counts healthy val runs whose confidence is at least q. That must equal the share of val
    # runs above calibrate's threshold at a budget of 1 minus q, the number Gate 1 reports
    calibrate, val = healthy_peaks(seed=0), healthy_peaks(seed=1)
    share = share_at_or_above(PercentileConfidence.fit(calibrate).confidence(val), q)
    false_alarm_rate = np.mean(val > threshold_at_budget(calibrate, 1 - q))
    assert share == pytest.approx(false_alarm_rate)


@pytest.mark.parametrize("share, target, ok", [(0.08, 0.05, True), (0.02, 0.05, True), (0.082, 0.05, False), (0.018, 0.05, False)])
def test_a_share_exactly_on_the_edge_passes(share: float, target: float, ok: bool) -> None:
    # Shares come in steps of 1 / 500, so one can sit exactly 3 points off. In floating point 0.08 - 0.05 is
    # 0.030000000000000006, which a plain <= 0.03 would wrongly fail. One step further must still fail
    assert within_tolerance(share, target, 0.03) is ok


def test_a_nan_peak_is_refused() -> None:
    # searchsorted sorts NaN to the end, so a broken score would get confidence 1.0, certainty from a bug.
    # The verdict should refuse instead
    calibration = PercentileConfidence.fit(healthy_peaks())
    with pytest.raises(ValueError):
        calibration.confidence(np.nan)
    with pytest.raises(ValueError):
        calibration.confidence(np.array([100.0, np.nan]))


def test_fit_needs_peaks() -> None:
    with pytest.raises(ValueError):
        PercentileConfidence.fit(np.array([]))


def test_fit_rejects_nan_peaks() -> None:
    with pytest.raises(ValueError):
        PercentileConfidence.fit(np.array([1.0, np.nan, 3.0]))
