"""The static demo, cut list item 4: one real caught fault, with the v1 verdict traced back to the engine's hardware.

Also exports four real runs for the demo page to replay, docs/demo/runs.js. Each run is chosen by a fixed rule, not by
eye: the first val2 run in seed order that model seed 0 catches at its 2% threshold while every channel is still under
its redline, the first healthy run with no alarm, the first faulted run missed with no early alarm, and the first
healthy run that false-alarms. Test and test2 are locked, so val2 supplies them. The figure shows the caught run.

Run with: python -m evaluation.demo
The figure goes to docs/demo.png, the replay data to docs/demo/runs.js.
"""

import json
from collections.abc import Callable
from pathlib import Path

import matplotlib

# Write image files, never open a window
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from contracts import CHANNELS, DT_S, N_STEPS, Run, Verdict
from evaluation.baselines import redline_scores
from evaluation.budget import CATCH_WINDOW_S, FA_BUDGET, THRESHOLD_SPLIT
from evaluation.calibration import PercentileConfidence
from evaluation.final import D8_METRICS
from evaluation.gate1 import MODELS
from evaluation.harness import first_alarm_s, score_run, threshold_at_budget
from evaluation.score import REFERENCE_SEEDS, StandardizedScore, mean_residual
from generator.engine import REDLINES, make_throttle, simulate
from generator.splits import SPLITS, fault_for, make_run
from model.forecaster import residual_by_channel, to_verdict
from model.train import RUNS_DIR, load_run

SPLIT = "val2"
MODEL = MODELS[0]
OUT = Path("docs/demo.png")
# Each channel under the hardware it measures, so the verdict reads back onto the engine
SUBSYSTEMS: dict[str, tuple[str, ...]] = {
    "TURBOPUMP": ("turbopump_speed", "turbopump_vibration", "bearing_temp"),
    "COMBUSTION": ("chamber_pressure", "nozzle_temp", "mixture_ratio"),
    "FEED": ("feed_pressure", "coolant_flow"),
}
# How many of the verdict's ranked channels are highlighted
TOP_K = 3
OUT_JS = Path("docs/demo/runs.js")
# One replay frame per 10 samples, so 10 frames per second of run
EXPORT_STRIDE = 10
# A channel's departure is its 1 s mean miss over this percentile of healthy calibrate samples
SMOOTH = 100
DEPARTURE_Q = 99
# The four replayed runs, each the first match in seed order for its rule
SCENARIOS = {
    "quiet": "Healthy run, throttle steps",
    "caught": "Bearing wear, caught",
    "missed": "Bearing wear, flagged too late",
    "false_alarm": "Healthy run, false alarm",
}


def redline_fraction(sensors: np.ndarray) -> np.ndarray:
    """(n, 8) readings as fractions of each channel's redline."""
    return sensors / np.array([REDLINES[c] for c in CHANNELS])


def plot_demo(
    run: Run, twin: Run, score: np.ndarray, threshold: float, verdict: Verdict, alarm_s: float, path: Path, label: str
) -> None:
    """Throttle, every channel as a share of its redline grouped by subsystem, the score, and the verdict."""
    t = np.arange(N_STEPS) * DT_S
    frac, twin_frac = redline_fraction(run.sensors) * 100, redline_fraction(twin.sensors) * 100
    top = verdict.ranked_channels[:TOP_K]
    order = [c for channels in SUBSYSTEMS.values() for c in channels]
    group = {c: name for name, channels in SUBSYSTEMS.items() for c in channels}

    fig, axes = plt.subplots(len(order) + 2, 1, sharex=True, figsize=(11, 15))
    axes[0].plot(t, run.throttle, color="black", lw=1)
    axes[0].set_ylabel("throttle\ncommand", rotation=0, ha="right", fontsize=8)

    for ax, name in zip(axes[1:-1], order):
        i = CHANNELS.index(name)
        hot = name in top
        ax.plot(t, twin_frac[:, i], color="lightgray", lw=0.8, label="same engine, no fault")
        ax.plot(t, frac[:, i], color="tab:red" if hot else "tab:blue", lw=0.7 if not hot else 1.0, label="this run")
        ax.axhline(100, color="red", ls="--", lw=0.8, label="redline")
        ax.set_ylim(min(frac[:, i].min(), twin_frac[:, i].min()) - 5, 105)
        rank = f"  #{top.index(name) + 1}" if hot else ""
        ax.set_ylabel(f"{group[name]}\n{name}{rank}\n% of redline", rotation=0, ha="right", fontsize=8,
                      color="tab:red" if hot else "black")
    axes[1].legend(loc="lower right", fontsize=7)

    finite = np.where(np.isfinite(score), score, np.nan)
    axes[-1].plot(t, finite, color="tab:purple", lw=0.8)
    axes[-1].axhline(threshold, color="tab:purple", ls="--", lw=0.8, label=f"alarm threshold, {FA_BUDGET:.0%} false alarm budget")
    axes[-1].set_ylabel("orchestrator\nstd score", rotation=0, ha="right", fontsize=8)
    axes[-1].legend(loc="upper left", fontsize=7)
    axes[-1].set_xlabel("time (s)")

    onset = run.fault.onset_s if run.fault is not None else None
    for ax in axes:
        if onset is not None:
            ax.axvline(onset, color="orange", lw=1)
            ax.axvspan(onset, onset + CATCH_WINDOW_S, color="orange", alpha=0.08)
        ax.axvline(alarm_s, color="tab:purple", lw=1.2)

    peak_redline = float(redline_scores(run)[: round(alarm_s / DT_S) + 1].max())
    lines = [
        "ADVISORY VERDICT. Redline protection keeps all authority",
        f"Flag: {'ANOMALY' if verdict.flag else 'none'} at {alarm_s:.1f} s"
        + (f", {alarm_s - onset:.1f} s after the injected onset at {onset:.1f} s" if onset is not None else ""),
        f"Confidence: {verdict.confidence:.2f}, more unusual than {verdict.confidence:.0%} of healthy calibrate runs",
        "Where to look: " + ", ".join(f"#{k + 1} {c}" for k, c in enumerate(top)),
        "Ranked by forecast residual: relevance, not proven cause",
        f"Redlines at the alarm: none crossed. Highest channel at {peak_redline:.0%} of its limit",
    ]
    if run.fault is not None:
        lines.append("True fault channels (ground truth, never seen by the model): " + ", ".join(run.fault.channels))
    fig.text(0.5, 0.995, "\n".join(lines), ha="center", va="top", fontsize=9, family="monospace",
             bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "tab:purple"})
    fig.suptitle(label, y=0.915, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def smooth(r: np.ndarray) -> np.ndarray:
    """(n, k) moving mean over the last SMOOTH samples, skipping NaN rows. NaN where a window has no data."""
    finite = np.isfinite(r)
    total = np.cumsum(np.where(finite, r, 0.0), axis=0)
    count = np.cumsum(finite, axis=0)
    total[SMOOTH:] -= total[:-SMOOTH].copy()
    count[SMOOTH:] -= count[:-SMOOTH].copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def causal_ranking(r: np.ndarray, upto: int) -> tuple[str, ...]:
    """The verdict's rule, each channel's average miss, using only rows up to and including upto."""
    per_channel = np.nanmean(r[: upto + 1], axis=0)
    return tuple(CHANNELS[i] for i in np.argsort(per_channel)[::-1])


def to_frames(
    run: Run,
    r: np.ndarray,
    score: np.ndarray,
    threshold: float,
    confidence: Callable[[np.ndarray], np.ndarray],
    scale: np.ndarray,
) -> dict:
    """One run as replay frames at 10 Hz, for the demo page. Everything a frame shows is known by its time."""
    running_peak = np.maximum.accumulate(score)
    alarm = running_peak > threshold
    first = int(np.argmax(alarm)) if alarm.any() else None
    departure = np.nan_to_num(smooth(r) / scale, nan=0.0)
    idx = np.arange(0, N_STEPS, EXPORT_STRIDE)
    return {
        "seed": int(run.seed),
        "onset_s": None if run.fault is None else round(run.fault.onset_s, 2),
        "fault_channels": [] if run.fault is None else list(run.fault.channels),
        "threshold": round(float(threshold), 3),
        "alarm_s": None if first is None else round(first * DT_S, 2),
        "ranked_at_alarm": [] if first is None else list(causal_ranking(r, first)),
        "t": [round(i * DT_S, 2) for i in idx],
        "throttle": [round(float(x), 3) for x in run.throttle[idx]],
        "pct": [[round(float(x), 1) for x in row] for row in redline_fraction(run.sensors[idx]) * 100],
        "departure": [[round(float(x), 2) for x in row] for row in departure[idx]],
        "score": [None if not np.isfinite(x) else round(float(x), 2) for x in score[idx]],
        "alarm": [bool(x) for x in alarm[idx]],
        "confidence": [round(float(x), 3) for x in confidence(running_peak[idx])],
    }


def main() -> None:
    trained = load_run(RUNS_DIR / MODEL)

    def residuals(run: Run) -> np.ndarray:
        return residual_by_channel(trained.model, trained.normalizer, run, masked=True)

    # The v1 score and calibration for this model, exactly as on D8
    std = StandardizedScore.fit((make_run("train", s) for s in REFERENCE_SEEDS), lambda run: mean_residual(residuals(run)))
    peaks, samples = [], []
    for s in SPLITS[THRESHOLD_SPLIT]:
        if fault_for(THRESHOLD_SPLIT, s) is not None:
            continue
        run = make_run(THRESHOLD_SPLIT, s)
        r = residuals(run)
        peaks.append(float(std.scores(mean_residual(r), run.throttle).max()))
        # Every 10th sample is plenty for a percentile, and keeps memory small
        samples.append(smooth(r)[::10])
    threshold = threshold_at_budget(peaks, FA_BUDGET)
    saved = json.loads(D8_METRICS.read_text())["thresholds"][f"{MODEL}/std"]["2%"]
    if not np.isclose(threshold, saved, rtol=1e-6, atol=0):
        raise ValueError(f"threshold {threshold} differs from D8's {saved}")
    calibration = PercentileConfidence.fit(peaks)
    # Each channel's healthy 99th percentile miss, so a departure of 1 means beyond 99% of healthy calibrate samples
    scale = np.nanpercentile(np.concatenate(samples), DEPARTURE_Q, axis=0)

    # Fixed rules, first match in seed order for each, so nothing is picked by eye
    found: dict[str, tuple] = {}
    for seed in SPLITS[SPLIT]:
        if len(found) == len(SCENARIOS):
            break
        run = make_run(SPLIT, seed)
        r = residuals(run)
        score = std.verdict_score(r, run)
        result = score_run(score, run.throttle, threshold, run.fault)
        if run.fault is None:
            kind = "false_alarm" if result.false_alarm else "quiet"
        elif result.caught:
            alarm_s = first_alarm_s(score, threshold)
            # Caught while every redline is still clear at the alarm
            kind = "caught" if redline_scores(run)[: round(alarm_s / DT_S) + 1].max() < 1.0 else None
        else:
            kind = None if result.early_alarm else "missed"
        if kind is not None and kind not in found:
            found[kind] = (run, r, score)
    missing = set(SCENARIOS) - set(found)
    if missing:
        raise RuntimeError(f"no {SPLIT} run found for {sorted(missing)}")

    export = {}
    for kind in SCENARIOS:
        run, r, score = found[kind]
        frames = to_frames(run, r, score, threshold, calibration.confidence, scale)
        # The replay's flag and confidence must be the v1 verdict's own
        verdict = to_verdict(trained.model, trained.normalizer, run, threshold, calibration.confidence, score=std.verdict_score)
        if verdict.flag != (frames["alarm_s"] is not None) or abs(verdict.confidence - float(calibration.confidence(score.max()))) > 1e-12:
            raise ValueError(f"{kind}: the replay disagrees with to_verdict")
        frames["verdict_confidence"] = round(float(verdict.confidence), 3)
        export[kind] = {"label": SCENARIOS[kind], **frames}
        print(f"{kind}: seed {run.seed}, alarm {frames['alarm_s']}, onset {frames['onset_s']}, confidence {verdict.confidence:.3f}")

    OUT_JS.parent.mkdir(parents=True, exist_ok=True)
    header = f"// Generated by python -m evaluation.demo from {SPLIT}, model {MODEL}. Real model outputs, not scripted\n"
    OUT_JS.write_text(header + "window.RUNS = " + json.dumps(export, separators=(",", ":")) + ";\n")
    print(f"saved {OUT_JS}")

    run, _, score = found["caught"]
    verdict = to_verdict(trained.model, trained.normalizer, run, threshold, calibration.confidence, score=std.verdict_score)
    alarm_s = first_alarm_s(score, threshold)
    twin = simulate(make_throttle(run.seed), run.seed)
    label = f"{SPLIT} seed {run.seed}, model {MODEL}. Bearing wear, a real run chosen by a fixed rule"
    plot_demo(run, twin, score, threshold, verdict, alarm_s, OUT, label)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
