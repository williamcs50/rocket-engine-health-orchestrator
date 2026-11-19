"""Follow-up checks, run by name.

python -m evaluation.checks              D4: z-score false-alarm consistency, peak timing, persistence error (about 90 s)
python -m evaluation.checks slow_state   D7 diagnostic: is startup slow state behind the false alarms? (about 2 minutes)
"""

import argparse
import json
from math import comb

import numpy as np

from contracts import CHANNELS, DT_S
from evaluation.baselines import ZScoreBaseline, persistence_forecast
from evaluation.budget import FA_BUDGET, REPORT_SPLIT, THRESHOLD_SPLIT
from evaluation.harness import first_alarm_s, healthy_peaks, score_run, threshold_at_budget, transient_mask
from generator.engine import SENSOR_NOISE
from generator.splits import SPLITS, fault_for, make_run
from model.forecaster import residual_by_channel, residual_scores
from model.train import RUNS_DIR, load_run

# Samples ahead, the D5 starting horizon
HORIZON = 10
# The D7 diagnostic's model, and the seconds counted as startup. Both fixed in the predictions before it ran
DIAG_RUN = "h10_mask0.25_cmd_seed0"
STARTUP_S = 8.0


def healthy_seeds(split: str) -> list[int]:
    return [seed for seed in SPLITS[split] if fault_for(split, seed) is None]


def main() -> None:
    zscore = ZScoreBaseline.fit(make_run("train", seed) for seed in SPLITS["train"])
    cal = healthy_peaks(zscore.scores, THRESHOLD_SPLIT)
    val = healthy_peaks(zscore.scores, REPORT_SPLIT)
    threshold = threshold_at_budget(cal, FA_BUDGET)
    print(f"z-score threshold: {threshold:.3f}")

    # The tutor's check: the threshold applied back to the split it was set on must give the budget
    print(f"{THRESHOLD_SPLIT} false alarms (peaks):     {np.sum(cal > threshold)}/{len(cal)} = {np.mean(cal > threshold):.1%}")
    flagged = [score_run(zscore.scores(run), run.throttle, threshold, None).false_alarm
               for run in (make_run(THRESHOLD_SPLIT, seed) for seed in healthy_seeds(THRESHOLD_SPLIT))]
    print(f"{THRESHOLD_SPLIT} false alarms (score_run): {np.mean(flagged):.1%}")
    print(f"{REPORT_SPLIT} false alarms:                 {np.sum(val > threshold)}/{len(val)} = {np.mean(val > threshold):.1%}")

    k, n = int(np.sum(val > threshold)), len(val)
    chance = sum(comb(n, i) * FA_BUDGET**i * (1 - FA_BUDGET) ** (n - i) for i in range(k + 1))
    print(f"chance of {k} or fewer out of {n} at a true {FA_BUDGET:.0%}: {chance:.4f}")

    print(f"{THRESHOLD_SPLIT} peak quantiles (50, 90, 98, 100): {np.round(np.quantile(cal, [0.5, 0.9, 0.98, 1.0]), 2)}")
    print(f"{REPORT_SPLIT} peak quantiles (50, 90, 98, 100):       {np.round(np.quantile(val, [0.5, 0.9, 0.98, 1.0]), 2)}")
    print(f"{REPORT_SPLIT}'s own threshold at the budget: {threshold_at_budget(val, FA_BUDGET):.3f}")

    times = [np.argmax(zscore.scores(make_run(THRESHOLD_SPLIT, seed))) * DT_S for seed in healthy_seeds(THRESHOLD_SPLIT)[:20]]
    print(f"peak times (s), first 20 healthy {THRESHOLD_SPLIT} runs: {np.round(times, 1)}")

    errors = []
    for seed in healthy_seeds(REPORT_SPLIT):
        # Train statistics only
        x = (make_run(REPORT_SPLIT, seed).sensors - zscore.mean) / zscore.std
        errors.append(np.mean((x[HORIZON:] - persistence_forecast(x, HORIZON)[HORIZON:]) ** 2, axis=0))
    errors = np.array(errors).mean(axis=0)
    print(f"persistence MSE overall (normalized, h={HORIZON}): {errors.mean():.4f}, "
          f"mixture_ratio share {errors[CHANNELS.index('mixture_ratio')] / errors.sum():.0%}")

    # Noise floor: sensor noise variance in train-normalized units. No forecaster can beat it.
    floor = np.array([SENSOR_NOISE[name] for name in CHANNELS]) ** 2 / zscore.std**2
    print(f"{'channel':20s} {'persistence':>11s} {'floor':>7s} {'ratio':>6s}")
    for name, err, flr in zip(CHANNELS, errors, floor):
        print(f"{name:20s} {err:11.4f} {flr:7.4f} {err / flr:6.1f}")


def slow_state() -> None:
    """D7 diagnostic. Both predictions are in the D7 build log, written before this ran."""
    trained = load_run(RUNS_DIR / DIAG_RUN)
    i = CHANNELS.index("nozzle_temp")

    # Prediction 1: unmasking shrinks nozzle temp's startup peak by at least half, and by more than its steady-state peak
    print(f"Nozzle temp peak residual, masked and unmasked ({DIAG_RUN}, first 10 healthy {THRESHOLD_SPLIT} runs)")
    print("Ratio is unmasked / masked, so a smaller ratio means unmasking helped more")
    print(f"{'seed':>7s} {'startup':>9s} {'unmasked':>9s} {'ratio':>6s} {'steady':>9s} {'unmasked':>9s} {'ratio':>6s}  both")
    met = 0
    for seed in healthy_seeds(THRESHOLD_SPLIT)[:10]:
        run = make_run(THRESHOLD_SPLIT, seed)
        masked = residual_by_channel(trained.model, trained.normalizer, run, masked=True)[:, i]
        unmasked = residual_by_channel(trained.model, trained.normalizer, run, masked=False)[:, i]
        t = np.arange(len(run.throttle)) * DT_S
        startup = t < STARTUP_S
        # After startup and away from every commanded change, so mid-run steps are excluded too
        steady = (t >= STARTUP_S) & ~transient_mask(run.throttle)
        # nanmax, since the first 1.09 s have no forecast yet
        start_m, start_u = np.nanmax(masked[startup]), np.nanmax(unmasked[startup])
        steady_m, steady_u = np.nanmax(masked[steady]), np.nanmax(unmasked[steady])
        start_ratio, steady_ratio = start_u / start_m, steady_u / steady_m
        # Both conditions from the prediction, in the same run
        both = start_ratio <= 0.5 and start_ratio < steady_ratio
        met += both
        print(
            f"{seed:>7d} {start_m:9.1f} {start_u:9.1f} {start_ratio:6.2f} "
            f"{steady_m:9.1f} {steady_u:9.1f} {steady_ratio:6.2f}  {'yes' if both else 'no'}"
        )
    print(f"Both conditions met in {met} of 10 runs. Predicted at least 6")

    # Prediction 2: most of this seed's val false alarms start during startup
    metrics = json.loads((RUNS_DIR / "gate1_cmd" / "metrics.json").read_text())
    # Read from the file, so it's exactly the threshold Gate 1 used, not a rounded copy
    threshold = metrics["thresholds"][f"{DIAG_RUN}/mean"]["2%"]
    firsts = []
    for seed in healthy_seeds(REPORT_SPLIT):
        run = make_run(REPORT_SPLIT, seed)
        # Masked scoring with the mean aggregate, the same score Gate 1 thresholded
        first = first_alarm_s(residual_scores(trained.model, trained.normalizer, run), threshold)
        if first is not None:
            firsts.append(first)
    early = sum(f < 10.0 for f in firsts)
    print(f"\nFirst false alarm times on healthy {REPORT_SPLIT} runs, threshold {threshold:.3f}")
    print(f"  {len(firsts)} runs alarmed (Gate 1 counted 17). {early} first alarmed before 10 s. Predicted at least 12")
    print(f"  times (s): {sorted(round(f, 1) for f in firsts)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Follow-up checks.")
    parser.add_argument("check", nargs="?", default="d4", choices=["d4", "slow_state"])
    args = parser.parse_args()
    if args.check == "slow_state":
        slow_state()
    else:
        main()
