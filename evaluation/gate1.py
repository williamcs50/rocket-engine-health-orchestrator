"""Gate 1: the trained models against the redline and z-score on val, at the pre-registered budgets.

Run with: python -m evaluation.gate1
"""

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from contracts import CHANNELS, DT_S
from evaluation.baselines import REDLINE_THRESHOLD, ZScoreBaseline, redline_scores
from evaluation.budget import CATCH_WINDOW_S, FA_BUDGET, FA_SECONDARY, REPORT_SPLIT, THRESHOLD_SPLIT
from evaluation.harness import score_run, summarize, threshold_at_budget, transient_mask
from generator.splits import SPLITS, fault_for, make_run
from model.forecaster import aggregate, residual_by_channel
from model.train import RUNS_DIR, load_run

# D7's command-aware seeds. D6's models (h10_mask0.25_seed*, h10_mask0_seed0) are the ablation, with results in
# runs/gate1. Reproduce them from commit c2cf7fe. The current load_run refuses them, since they predate command_aware
MODELS = ("h10_mask0.25_cmd_seed0", "h10_mask0.25_cmd_seed1", "h10_mask0.25_cmd_seed2")
# Where results are saved. Its own folder, so D6's runs/gate1/metrics.json stays as the ablation's record
OUT_NAME = "gate1_cmd"
AGGREGATES = ("mean", "max")
# 2% primary, then 1% and 5%
BUDGETS = (FA_BUDGET, *FA_SECONDARY)


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


def calibrate_thresholds(models: dict, zscore, limit: int | None = None) -> dict:
    """Pass 1: every detector's peak score on calibrate's healthy runs, then one threshold per budget."""
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
        # The z score needs no model, so it's scored once per run
        peaks.setdefault("zscore", []).append(float(zscore.scores(run).max()))

    # One threshold per detector and budget
    thresholds = {name: {b: threshold_at_budget(p, b) for b in BUDGETS} for name, p in peaks.items()}
    # The redline is fixed and never tuned, so it keeps 1.0 at every budget
    thresholds["redline"] = {b: REDLINE_THRESHOLD for b in BUDGETS}
    return thresholds


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


def evaluate_val(models: dict, zscore, thresholds: dict, limit: int | None = None) -> dict:
    """Pass 2: every detector at every budget on val. Each run simulated once, each model scored once."""
    # (detector, budget) to [(is healthy, RunResult), ...]
    rows: dict[tuple[str, float], list] = {}
    for seed in SPLITS[REPORT_SPLIT][:limit]:
        run = make_run(REPORT_SPLIT, seed)
        scores = {"redline": redline_scores(run), "zscore": zscore.scores(run)}
        for name, trained in models.items():
            masked = trained.config["mask_prob"] > 0
            r = residual_by_channel(trained.model, trained.normalizer, run, masked=masked)
            for how in AGGREGATES:
                scores[f"{name}/{how}"] = aggregate(r, how)
        for detector, s in scores.items():
            for b in BUDGETS:
                result = score_run(s, run.throttle, thresholds[detector][b], run.fault)
                rows.setdefault((detector, b), []).append((run.fault is None, result))
    return {key: summarize(results) for key, results in rows.items()}


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
    # Train only, as in D4. A quick check fits on fewer runs, since only the wiring is being tested
    zscore = ZScoreBaseline.fit(make_run("train", s) for s in SPLITS["train"][:limit])

    thresholds = calibrate_thresholds(models, zscore, limit)
    for detector, by_budget in thresholds.items():
        print(detector, {f"{b:.0%}": round(t, 3) for b, t in by_budget.items()})

    peak_times(models[MODELS[0]])
    reports = evaluate_val(models, zscore, thresholds, limit)

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
    for how in AGGREGATES:
        recall = [reports[(f"{m}/{how}", FA_BUDGET)].recall for m in masked_runs]
        fa = [reports[(f"{m}/{how}", FA_BUDGET)].false_alarm_rate for m in masked_runs]
        # The D7 exit test is judged on the mean aggregate's line only. Max is reported, not judged
        transient = [reports[(f"{m}/{how}", FA_BUDGET)].transient_false_alarm_rate for m in masked_runs]
        print(
            f"  {how:4s} recall {np.mean(recall):.1%} ± {np.std(recall, ddof=1):.1%}, lowest seed {min(recall):.1%}, "
            f"highest false alarm {max(fa):.1%}, mean transient false alarm {np.mean(transient):.1%}"
        )

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