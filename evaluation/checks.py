"""D4 follow-up checks: z-score false-alarm consistency, peak timing, persistence error.

Run with: python -m evaluation.checks  (about 90 s, mostly fitting the z-score on train)
"""

from math import comb

import numpy as np

from contracts import CHANNELS, DT_S
from evaluation.baselines import ZScoreBaseline, persistence_forecast
from evaluation.budget import FA_BUDGET, REPORT_SPLIT, THRESHOLD_SPLIT
from evaluation.harness import healthy_peaks, score_run, threshold_at_budget
from generator.engine import SENSOR_NOISE
from generator.splits import SPLITS, fault_for, make_run

HORIZON = 10  # samples ahead, the D5 starting horizon


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
        x = (make_run(REPORT_SPLIT, seed).sensors - zscore.mean) / zscore.std  # train statistics only
        errors.append(np.mean((x[HORIZON:] - persistence_forecast(x, HORIZON)[HORIZON:]) ** 2, axis=0))
    errors = np.array(errors).mean(axis=0)
    print(f"persistence MSE overall (normalized, h={HORIZON}): {errors.mean():.4f}, "
          f"mixture_ratio share {errors[CHANNELS.index('mixture_ratio')] / errors.sum():.0%}")

    # Noise floor: sensor noise variance in train-normalized units. No forecaster can beat it.
    floor = np.array([SENSOR_NOISE[name] for name in CHANNELS]) ** 2 / zscore.std**2
    print(f"{'channel':20s} {'persistence':>11s} {'floor':>7s} {'ratio':>6s}")
    for name, err, flr in zip(CHANNELS, errors, floor):
        print(f"{name:20s} {err:11.4f} {flr:7.4f} {err / flr:6.1f}")


if __name__ == "__main__":
    main()
