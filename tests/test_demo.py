from pathlib import Path

import numpy as np

from contracts import CHANNELS, N_STEPS, FaultLabel, Run, Verdict
from evaluation.calibration import PercentileConfidence
from evaluation.demo import (
    EXPORT_STRIDE, SCENARIOS, SMOOTH, SUBSYSTEMS, causal_ranking, plot_demo, redline_fraction, smooth, to_frames,
)
from generator.engine import REDLINES


def test_subsystems_cover_every_channel_once() -> None:
    grouped = [c for channels in SUBSYSTEMS.values() for c in channels]
    assert sorted(grouped) == sorted(CHANNELS)
    assert len(grouped) == len(set(grouped))


def test_redline_fraction_is_each_reading_over_its_limit() -> None:
    sensors = np.array([[REDLINES[c] / 2 for c in CHANNELS], [REDLINES[c] for c in CHANNELS]])
    np.testing.assert_allclose(redline_fraction(sensors), [[0.5] * len(CHANNELS), [1.0] * len(CHANNELS)])


def test_plot_demo_writes_a_figure(tmp_path: Path) -> None:
    # A made-up run, only to check the figure draws end to end
    sensors = np.array([[REDLINES[c] * 0.7 for c in CHANNELS]] * N_STEPS)
    fault = FaultLabel(kind="bearing_wear", onset_s=20.0, rate=0.05, channels=("bearing_temp",))
    run = Run(throttle=np.full(N_STEPS, 0.8), sensors=sensors, seed=1, fault=fault)
    score = np.linspace(0.0, 10.0, N_STEPS)
    score[:109] = -np.inf
    verdict = Verdict(flag=True, confidence=0.99, ranked_channels=tuple(CHANNELS))
    path = tmp_path / "demo.png"
    plot_demo(run, run, score, threshold=8.0, verdict=verdict, alarm_s=48.0, path=path, label="test")
    assert path.stat().st_size > 0


# The replay export

def test_smooth_is_a_trailing_mean_that_skips_nan_rows() -> None:
    r = np.full((300, 2), 2.0)
    r[:50] = np.nan
    r[200:, 0] = 4.0
    out = smooth(r)
    assert np.isnan(out[49]).all()
    np.testing.assert_allclose(out[50], [2.0, 2.0])
    # A full window after the step is all 4s
    assert out[200 + SMOOTH - 1, 0] == 4.0
    # Halfway through the step the window holds half of each
    assert out[200 + SMOOTH // 2 - 1, 0] == 3.0


def test_causal_ranking_ignores_rows_after_upto() -> None:
    r = np.zeros((10, len(CHANNELS)))
    r[:5, 2] = 1.0
    # A later rise on another channel must not change the ranking at row 4
    r[5:, 6] = 100.0
    assert causal_ranking(r, 4)[0] == CHANNELS[2]
    assert causal_ranking(r, 9)[0] == CHANNELS[6]


def test_to_frames_latches_the_alarm_and_only_uses_the_past() -> None:
    sensors = np.array([[REDLINES[c] * 0.5 for c in CHANNELS]] * N_STEPS)
    run = Run(throttle=np.full(N_STEPS, 0.8), sensors=sensors, seed=7)
    r = np.ones((N_STEPS, len(CHANNELS)))
    score = np.zeros(N_STEPS)
    score[:109] = -np.inf
    # One spike at 30 s, then quiet again
    score[3000] = 20.0
    confidence = PercentileConfidence.fit(np.arange(10.0)).confidence
    f = to_frames(run, r, score, threshold=8.0, confidence=confidence, scale=np.ones(len(CHANNELS)))
    assert len(f["t"]) == N_STEPS // EXPORT_STRIDE
    assert f["alarm_s"] == 30.0
    # Latched: still alarmed after the spike passes, and not before it
    assert f["alarm"][2990 // EXPORT_STRIDE] is False
    assert f["alarm"][-1] is True
    assert f["confidence"][-1] == 1.0
    assert f["score"][0] is None
    assert f["pct"][0][0] == 50.0
    assert f["onset_s"] is None and f["fault_channels"] == []


def test_four_scenarios_are_exported() -> None:
    assert set(SCENARIOS) == {"quiet", "caught", "missed", "false_alarm"}
