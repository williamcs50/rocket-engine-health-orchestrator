"""Gate 1: the trained models against the redline and z-score on val, at the pre-registered budgets.

Run with: python -m evaluation.gate1
"""

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from contracts import CHANNELS, DT_S
from evaluation.baselines import REDLINE_THRESHOLD, ConditionedZScore, ZScoreBaseline, redline_scores
from evaluation.budget import CATCH_WINDOW_S, FA_BUDGET, FA_SECONDARY, REPORT_SPLIT, THRESHOLD_SPLIT
from evaluation.calibration import PercentileConfidence, share_at_or_above, within_tolerance
from evaluation.harness import score_run, summarize, threshold_at_budget, transient_mask
from evaluation.score import REFERENCE_SEEDS, StandardizedScore, mean_residual
from generator.splits import SPLITS, fault_for, make_run
from model.forecaster import aggregate, residual_by_channel
from model.train import RUNS_DIR, load_run

# D7's command-aware seeds. D6's models (h10_mask0.25_seed*, h10_mask0_seed0) were D7's comparison, with results in
# runs/gate1. Reproduce them from commit c2cf7fe. The current load_run refuses them, since they predate command_aware.
# The matched ablation is D12's _nocmd models, in evaluation/ablation.py
MODELS = ("h10_mask0.25_cmd_seed0", "h10_mask0.25_cmd_seed1", "h10_mask0.25_cmd_seed2")
# Where results are saved. Each day's run gets its own folder, so earlier committed results stay as they were:
# D6 in runs/gate1, D7 in runs/gate1_cmd
OUT_NAME = "gate1_d8"
AGGREGATES = ("mean", "max")
# 2% primary, then 1% and 5%
BUDGETS = (FA_BUDGET, *FA_SECONDARY)
# Calibration exit test: the share of healthy val runs with confidence at least q must be within 3 points of 1 minus q
CALIBRATION_QS = (0.8, 0.9, 0.95)
CALIBRATION_TOLERANCE = 0.03
# The Gate 1 bar, unchanged since D6: mean recall over seeds, every seed's recall, every seed's false alarm rate
BAR_MEAN_RECALL = 0.05
BAR_SEED_RECALL = 0.012
BAR_FALSE_ALARM = 0.03
# Lines summarized per seed: D6's two aggregates, plus the new standardized, differenced score
SUMMARY = ("mean", "max", "std")


def time_one_run() -> None:
    """How long one model takes to score one run. Multiply by 1,500 runs x 3 models for the full gate."""
    trained = load_run(RUNS_DIR / MODELS[0])
    run = make_run(REPORT_SPLIT, SPLITS[REPORT_SPLIT][0])
    began = time.perf_counter()
    residual_by_channel(trained.model, trained.normalizer, run)
    print(f"one masked scoring: {time.perf_counter() - began:.2f} s")


def healthy_seeds(split: str, limit: int | None = None) -> list[int]:
    seeds = [s for s in SPLITS[split] if fault_for(split, s) is None]
    return seeds[:limit]


def calibrate_thresholds(models: dict, baselines: dict, std_scores: dict, limit: int | None = None) -> tuple[dict, dict]:
    """Pass 1: every detector's peak score on calibrate's healthy runs, then its thresholds and its calibration.

    Both come from the same peaks, built here in one place, so a detector's flag and confidence always agree.
    """
    # Detector name to one peak per healthy run
    peaks: dict[str, list[float]] = {}
    for seed in healthy_seeds(THRESHOLD_SPLIT, limit):
        # Simulated once, shared by every detector
        run = make_run(THRESHOLD_SPLIT, seed)
        for name, trained in models.items():
            # Once per model. The no-masking comparison model is scored with each channel's own history
            masked = trained.config["mask_prob"] > 0
            r = residual_by_channel(trained.model, trained.normalizer, run, masked=masked)
            for how in AGGREGATES:
                # Both aggregates come from the same residuals, so masked scoring runs once
                peaks.setdefault(f"{name}/{how}", []).append(float(aggregate(r, how).max()))
            # The new score from the same residuals, through mean_residual so the blind rows stay NaN
            peaks.setdefault(f"{name}/std", []).append(float(std_scores[name].scores(mean_residual(r), run.throttle).max()))
        # The baselines need no model, so each is scored once per run
        for name, baseline in baselines.items():
            peaks.setdefault(name, []).append(float(baseline.scores(run).max()))

    # One threshold per detector and budget
    thresholds = {name: {b: threshold_at_budget(p, b) for b in BUDGETS} for name, p in peaks.items()}
    # The redline is fixed and never tuned, so it keeps 1.0 at every budget
    thresholds["redline"] = {b: REDLINE_THRESHOLD for b in BUDGETS}
    # Each detector's confidence, from the same healthy peaks as its thresholds
    calibrations = {name: PercentileConfidence.fit(p) for name, p in peaks.items()}
    return thresholds, calibrations


def peak_times(trained, n_runs: int = 10) -> None:
    """For a few healthy calibrate runs: when each run's score peaks, and which channel drives it."""
    print(f"\nWhere healthy {THRESHOLD_SPLIT} runs peak (mean aggregate)")
    for seed in healthy_seeds(THRESHOLD_SPLIT, n_runs):
        run = make_run(THRESHOLD_SPLIT, seed)
        r = residual_by_channel(trained.model, trained.normalizer, run)
        row = int(np.argmax(aggregate(r, "mean")))
        # Within TRANSIENT_S of a commanded throttle change, which includes the startup ramp
        near_change = bool(transient_mask(run.throttle)[row])
        print(f"  seed {seed}: {row * DT_S:5.1f} s, driven by {CHANNELS[int(r[row].argmax())]}, near a throttle change: {near_change}")


def fit_detectors(models: dict, limit: int | None = None) -> tuple[dict, dict]:
    """The baselines on train, and each model's std score references on train runs it never saw. As on D8."""
    # Train only, as in D4. A quick check fits on fewer runs, since only the wiring is being tested
    baselines = {
        "zscore": ZScoreBaseline.fit(make_run("train", s) for s in SPLITS["train"][:limit]),
        # The fair baseline for the Nov 20 test table. No training, so all 2,000 train runs. The Gate 1 bar
        # still uses D4's fixed 1.2%
        "zscore_fair": ConditionedZScore.fit(make_run("train", s) for s in SPLITS["train"][:limit]),
    }
    # The new score's healthy references, per model, on 500 train runs never trained on. Each run is scored once
    std_scores = {
        name: StandardizedScore.fit(
            (make_run("train", s) for s in REFERENCE_SEEDS[:limit]),
            lambda run, t=trained: mean_residual(residual_by_channel(t.model, t.normalizer, run, masked=True)),
        )
        for name, trained in models.items()
    }
    return baselines, std_scores


def detector_scores(run, models: dict, baselines: dict, std_scores: dict) -> tuple[dict, dict]:
    """Every detector's per-sample scores on one run, and each model's residuals. Shared with the test run script.

    Masked scoring is the slow part, so each model scores the run once and all three of its lines reuse it.
    """
    scores = {"redline": redline_scores(run)}
    for name, baseline in baselines.items():
        scores[name] = baseline.scores(run)
    residuals = {}
    for name, trained in models.items():
        masked = trained.config["mask_prob"] > 0
        r = residual_by_channel(trained.model, trained.normalizer, run, masked=masked)
        residuals[name] = r
        for how in AGGREGATES:
            scores[f"{name}/{how}"] = aggregate(r, how)
        # Through mean_residual, so the blind rows stay NaN
        scores[f"{name}/std"] = std_scores[name].scores(mean_residual(r), run.throttle)
    return scores, residuals


def evaluate_val(
    models: dict, baselines: dict, std_scores: dict, thresholds: dict, limit: int | None = None
) -> tuple[dict, dict]:
    """Pass 2: every detector at every budget on val, plus each detector's healthy val peaks for calibration."""
    # (detector, budget) to [(is healthy, RunResult), ...]
    rows: dict[tuple[str, float], list] = {}
    # Detector to one peak per healthy val run, for the calibration exit test
    healthy_val_peaks: dict[str, list[float]] = {}
    for seed in SPLITS[REPORT_SPLIT][:limit]:
        run = make_run(REPORT_SPLIT, seed)
        scores, _ = detector_scores(run, models, baselines, std_scores)
        for detector, s in scores.items():
            for b in BUDGETS:
                result = score_run(s, run.throttle, thresholds[detector][b], run.fault)
                rows.setdefault((detector, b), []).append((run.fault is None, result))
            if run.fault is None:
                healthy_val_peaks.setdefault(detector, []).append(float(s.max()))
    return {key: summarize(results) for key, results in rows.items()}, healthy_val_peaks


def calibration_exit_test(calibrations: dict, healthy_val_peaks: dict, detectors: list[str]) -> dict:
    """Per detector and q: the share of healthy val runs with confidence at least q, and whether it's within tolerance."""
    results = {}
    for detector in detectors:
        confidence = calibrations[detector].confidence(np.array(healthy_val_peaks[detector]))
        for q in CALIBRATION_QS:
            share = share_at_or_above(confidence, q)
            passes = within_tolerance(share, 1 - q, CALIBRATION_TOLERANCE)
            results[f"{detector}@{q}"] = {"share": share, "target": 1 - q, "passes": passes}
    return results


def bias_check(trained, limit: int | None = None) -> dict:
    """Bearing temp: masked residual spread across healthy runs vs the fault's rise within its own run."""
    i = CHANNELS.index("bearing_temp")
    healthy, rises = [], []
    for seed in SPLITS[REPORT_SPLIT][:limit]:
        run = make_run(REPORT_SPLIT, seed)
        r = residual_by_channel(trained.model, trained.normalizer, run)[:, i]
        if run.fault is None:
            # This run's typical level. Its spread across runs is the per-run bias
            healthy.append(float(np.nanmean(r)))
            continue
        onset = round(run.fault.onset_s / DT_S)
        end = round((run.fault.onset_s + CATCH_WINDOW_S) / DT_S)
        # How far the fault lifts the residual above this same run's level before onset
        rises.append(float(np.nanmean(r[onset:end]) - np.nanmean(r[:onset])))
    return {
        "healthy_std": float(np.std(healthy)),
        "healthy_p5": float(np.percentile(healthy, 5)),
        "healthy_p95": float(np.percentile(healthy, 95)),
        "mean_rise": float(np.mean(rises)),
        "median_rise": float(np.median(rises)),
    }


def main(limit: int | None = None) -> None:
    models = {name: load_run(RUNS_DIR / name) for name in MODELS}
    baselines, std_scores = fit_detectors(models, limit)

    thresholds, calibrations = calibrate_thresholds(models, baselines, std_scores, limit)
    for detector, by_budget in thresholds.items():
        print(detector, {f"{b:.0%}": round(t, 3) for b, t in by_budget.items()})

    peak_times(models[MODELS[0]])
    reports, healthy_val_peaks = evaluate_val(models, baselines, std_scores, thresholds, limit)

    n_faulted = sum(fault_for(REPORT_SPLIT, s) is not None for s in SPLITS[REPORT_SPLIT][:limit])
    for b in BUDGETS:
        print(f"\nVal at the {b:.0%} budget, {n_faulted} faulted runs")
        print(f"{'detector':26s} {'recall':>7s} {'false alarm':>12s} {'transient':>10s} {'early':>7s} {'delay':>8s}")
        for detector in thresholds:
            rep = reports[(detector, b)]
            delay = f"{rep.mean_delay_s:.1f} s" if rep.mean_delay_s is not None else "n/a"
            print(
                f"{detector:26s} {rep.recall:7.1%} {rep.false_alarm_rate:12.1%} "
                f"{rep.transient_false_alarm_rate:10.1%} {rep.early_alarm_rate:7.1%} {delay:>8s}"
            )

    masked_runs = [m for m in MODELS if models[m].config["mask_prob"] > 0]
    print(f"\nMasked seeds at {FA_BUDGET:.0%}, mean ± spread over {len(masked_runs)} seeds")
    for how in SUMMARY:
        recall = [reports[(f"{m}/{how}", FA_BUDGET)].recall for m in masked_runs]
        fa = [reports[(f"{m}/{how}", FA_BUDGET)].false_alarm_rate for m in masked_runs]
        transient = [reports[(f"{m}/{how}", FA_BUDGET)].transient_false_alarm_rate for m in masked_runs]
        print(
            f"  {how:4s} recall {np.mean(recall):.1%} ± {np.std(recall, ddof=1):.1%}, lowest seed {min(recall):.1%}, "
            f"highest false alarm {max(fa):.1%}, mean transient false alarm {np.mean(transient):.1%}"
        )

    # Adoption: the new score is adopted only if its line passes all three Gate 1 conditions, per the D8 rule
    std_recall = [reports[(f"{m}/std", FA_BUDGET)].recall for m in MODELS]
    std_fa = [reports[(f"{m}/std", FA_BUDGET)].false_alarm_rate for m in MODELS]
    conditions = {
        # A tiny slack, since a mean of three shares that is exactly 5% can come out a hair under it in floating point
        f"mean recall {np.mean(std_recall):.1%} at least {BAR_MEAN_RECALL:.0%}": np.mean(std_recall) >= BAR_MEAN_RECALL - 1e-9,
        f"lowest seed {min(std_recall):.1%} above {BAR_SEED_RECALL:.1%}": min(std_recall) > BAR_SEED_RECALL,
        f"highest false alarm {max(std_fa):.1%} at most {BAR_FALSE_ALARM:.0%}": max(std_fa) <= BAR_FALSE_ALARM,
    }
    adopted = all(conditions.values())
    print(f"\nGate 1 on the new score (std line): {'PASS, adopted' if adopted else 'FAIL, not adopted'}")
    for text, ok in conditions.items():
        print(f"  {'yes' if ok else 'no '}  {text}")

    # Judged per seed, on the scoring that goes to test: the new score if adopted, otherwise the mean aggregate
    calibration = calibration_exit_test(calibrations, healthy_val_peaks, [f"{m}/{how}" for m in MODELS for how in ("mean", "std")])
    print(f"\nCalibration exit test: share of healthy {REPORT_SPLIT} runs with confidence at least q, target 1 minus q, within 3 points")
    for how in ("mean", "std"):
        for m in MODELS:
            detector = f"{m}/{how}"
            cells = [
                f"q {q}: {calibration[f'{detector}@{q}']['share']:.1%} ({'pass' if calibration[f'{detector}@{q}']['passes'] else 'FAIL'})"
                for q in CALIBRATION_QS
            ]
            print(f"  {detector:26s} " + "   ".join(cells))
    judged = "std" if adopted else "mean"
    passed = all(v["passes"] for k, v in calibration.items() if k.split("@")[0].endswith(f"/{judged}"))
    print(f"  Exit test, judged on the {judged} line: {'PASS' if passed else 'FAIL'}, per seed")

    bias = bias_check(models[MODELS[0]], limit)
    print(
        f"\nBias check, bearing temp ({MODELS[0]}): healthy level std {bias['healthy_std']:.2f}, "
        f"5th to 95th {bias['healthy_p5']:.2f} to {bias['healthy_p95']:.2f}. "
        f"Fault rise mean {bias['mean_rise']:.2f}, median {bias['median_rise']:.2f}"
    )

    # Quick checks aren't saved, so a 20 run result can never be committed as the Gate 1 numbers
    if limit is None:
        out = RUNS_DIR / OUT_NAME
        out.mkdir(parents=True, exist_ok=True)
        metrics = {
            "thresholds": {d: {f"{b:.0%}": t for b, t in tb.items()} for d, tb in thresholds.items()},
            # JSON keys must be strings, so (detector, budget) becomes "detector@2%"
            "reports": {f"{d}@{b:.0%}": asdict(rep) for (d, b), rep in reports.items()},
            "bias": bias,
            "calibration_exit_test": calibration,
            "new_score": {"adopted": adopted, "conditions": {text: bool(ok) for text, ok in conditions.items()}},
        }
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"\nsaved {out / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate 1 on val.")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N seeds per split (quick check)")
    parser.add_argument("--time", action="store_true", help="only time one masked scoring")
    args = parser.parse_args()
    if args.time:
        time_one_run()
    else:
        main(args.limit)