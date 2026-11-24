"""Do calibration offsets drive the std score's false alarms? Decision log, 2026-11-24 (v1.1, false alarms near the cap).

Run with: python -m evaluation.diagnose
Quick check of the wiring: python -m evaluation.diagnose --limit 20
"""

import argparse
import json

import numpy as np

from contracts import DT_S
from evaluation.budget import FA_BUDGET
from evaluation.final import D8_METRICS, provenance
from evaluation.gate1 import MODELS, calibrate_thresholds, fit_detectors
from evaluation.score import FIRST_DIFF, mean_residual
from generator.engine import calibration_offsets
from generator.splits import SPLITS, fault_for, make_run
from model.forecaster import residual_by_channel
from model.train import RUNS_DIR, load_run

# The development split this diagnostic reads
SPLIT = "val2"
# The swing window: from the first lagged difference (21.09 s) up to 50 s, before any shutdown starts
SWING_START = FIRST_DIFF
SWING_END = round(50.0 / DT_S)
# Pre-registered bar for the first condition
BAR_SPEARMAN = 0.3
OUT = RUNS_DIR / "diagnose_offsets"


def swing(s: np.ndarray) -> float:
    """The standard deviation of a run's std score inside the swing window."""
    return float(np.std(s[SWING_START:SWING_END]))


def rms_offset(offsets: np.ndarray) -> float:
    """One engine's offset: the root mean square of its 8 channel offsets, in noise sigmas."""
    return float(np.sqrt(np.mean(np.asarray(offsets) ** 2)))


def _ranks(x: np.ndarray) -> np.ndarray:
    """Ranks from 1, with tied values sharing the average of their ranks."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    # Each tied group gets the mean of the ranks it covers
    _, group = np.unique(x, return_inverse=True)
    return (np.bincount(group, ranks) / np.bincount(group))[group]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman's correlation: the Pearson correlation of the two variables' ranks."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, not {len(x)} and {len(y)}")
    if len(x) < 3:
        raise ValueError("Spearman needs at least 3 pairs")
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def alarm_engines(alarms: dict[str, dict[int, bool]]) -> set[int]:
    """Healthy engines that false-alarm on any model seed, each counted once."""
    return {seed for by_seed in alarms.values() for seed, alarmed in by_seed.items() if alarmed}


def diagnostic_verdict(
    spearmans: list[float], alarm_offsets: np.ndarray, quiet_offsets: np.ndarray
) -> tuple[bool, dict[str, bool]]:
    """Both pre-registered conditions, each as readable text."""
    alarm_median = float(np.median(alarm_offsets)) if len(alarm_offsets) else None
    quiet_median = float(np.median(quiet_offsets))
    seeds = ", ".join(f"{r:.3f}" for r in spearmans)
    conditions = {
        # A tiny slack, so a correlation of exactly 0.3 passes as the bar says
        f"Spearman at least {BAR_SPEARMAN} on every seed ({seeds})": all(r >= BAR_SPEARMAN - 1e-9 for r in spearmans),
        f"false alarm engines' median offset ({'none' if alarm_median is None else f'{alarm_median:.3f}'}) "
        f"above the others' ({quiet_median:.3f})": alarm_median is not None and alarm_median > quiet_median,
    }
    return all(conditions.values()), conditions


def main(limit: int | None = None) -> None:
    models = {name: load_run(RUNS_DIR / name) for name in MODELS}
    _, std_scores = fit_detectors(models, limit)
    thresholds, _ = calibrate_thresholds(models, {}, std_scores, limit)
    if limit is None:
        # The same std thresholds as D8, test and test2, or this isn't v1's std score. Only the model lines are
        # computed here, since the baselines aren't needed, so the full check_thresholds doesn't apply
        saved = json.loads(D8_METRICS.read_text())["thresholds"]
        for name in MODELS:
            if not np.isclose(thresholds[f"{name}/std"][FA_BUDGET], saved[f"{name}/std"]["2%"], rtol=1e-6, atol=0):
                raise ValueError(f"{name}/std threshold differs from D8's")
        print(f"Std thresholds match {D8_METRICS}")

    rows = []
    healthy = [s for s in SPLITS[SPLIT] if fault_for(SPLIT, s) is None][:limit]
    for seed in healthy:
        run = make_run(SPLIT, seed)
        offset = rms_offset(calibration_offsets(seed))
        for name, trained in models.items():
            masked = trained.config["mask_prob"] > 0
            r = residual_by_channel(trained.model, trained.normalizer, run, masked=masked)
            # The std line exactly as detector_scores scored test2
            s = std_scores[name].scores(mean_residual(r), run.throttle)
            peak = float(s.max())
            rows.append({
                "seed": int(seed), "model": name, "offset": offset, "swing": swing(s), "peak": peak,
                "alarm": bool(peak > thresholds[f"{name}/std"][FA_BUDGET]),
            })

    spearmans = [spearman([r["offset"] for r in rows if r["model"] == m], [r["swing"] for r in rows if r["model"] == m]) for m in MODELS]
    alarmed = alarm_engines({m: {r["seed"]: r["alarm"] for r in rows if r["model"] == m} for m in MODELS})
    offsets = {r["seed"]: r["offset"] for r in rows}
    alarm_offsets = np.array([o for s, o in offsets.items() if s in alarmed])
    quiet_offsets = np.array([o for s, o in offsets.items() if s not in alarmed])
    passed, conditions = diagnostic_verdict(spearmans, alarm_offsets, quiet_offsets)

    print(f"\nOffset diagnostic on {len(healthy)} healthy {SPLIT} engines")
    for m, r in zip(MODELS, spearmans):
        n_alarm = sum(row["alarm"] for row in rows if row["model"] == m)
        print(f"  {m}: Spearman {r:.3f}, false alarms {n_alarm} of {len(healthy)} ({n_alarm / len(healthy):.1%})")
    print(f"  Engines that false-alarm on any seed: {len(alarmed)} of {len(healthy)}")
    print(f"\nHypothesis: {'HOLDS' if passed else 'DOES NOT HOLD'}")
    for text, ok in conditions.items():
        print(f"  {'yes' if ok else 'no '}  {text}")

    if limit is None:
        OUT.mkdir(parents=True, exist_ok=True)
        metrics = {
            "split": SPLIT,
            "provenance": provenance(),
            "spearman": dict(zip(MODELS, spearmans)),
            "alarm_engines": sorted(alarmed),
            "median_offset": {"alarm": float(np.median(alarm_offsets)) if len(alarm_offsets) else None,
                              "quiet": float(np.median(quiet_offsets))},
            "holds": passed,
            "conditions": conditions,
            "rows": rows,
        }
        (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"\nsaved {OUT / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do calibration offsets drive the std score's false alarms?")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N seeds per split (quick check)")
    args = parser.parse_args()
    main(args.limit)
