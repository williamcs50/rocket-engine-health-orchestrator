"""The one test run (D9). Decision log, 2026-11-19 (test run reporting) and 2026-11-20 (attribution).

Run with: python -m evaluation.final
Dry run on val, saving nothing: python -m evaluation.final --dry-run 20
"""

import argparse
import hashlib
import json
import subprocess
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np

from evaluation.attribution import bootstrap_interval, control_onset, overlap, rise, top_k
from evaluation.budget import FA_BUDGET
from evaluation.gate1 import (
    BUDGETS, CALIBRATION_QS, MODELS, OUT_NAME as D8_NAME, calibrate_thresholds, calibration_exit_test,
    detector_scores, fit_detectors,
)
from evaluation.harness import score_run, summarize
from generator.splits import SPLITS, fault_for, make_run, open_test_split
from model.forecaster import attention_by_channel
from model.train import RUNS_DIR, load_run

# Written before any test run is simulated, so a crash still counts as the test split having been opened
STARTED = "started"
METRICS = "metrics.json"
# The committed D8 results. The thresholds recomputed today must match them before test is opened
D8_METRICS = RUNS_DIR / D8_NAME / METRICS
# Float32 scoring can differ in the last digits between days. Any real drift is far larger
THRESHOLD_RTOL = 1e-6
# Judged line, and the secondary line with its label, per the decision log (test run reporting)
JUDGED = "mean"
SECONDARY = "std"
SECONDARY_LABEL = "failed Gate 1 on val by one false alarm, not adopted, shown for information"
# Attribution populations: every faulted run, the faulted runs the judged line caught at 2%, and the healthy control
POPULATIONS = {
    "faulted": lambda row: row["faulted"],
    "caught": lambda row: row["faulted"] and row["caught"],
    "healthy control": lambda row: not row["faulted"],
}
ATTRIBUTION_METRICS = ("residual_overlap", "attention_overlap", "agreement")


def plan(dry_run: int | None) -> tuple[str, Path | None]:
    """Which split to report on, and where to save. A dry run uses val and saves nothing."""
    if dry_run is not None:
        return "val", None
    return "test", RUNS_DIR / "test"


def ensure_unclaimed(out: Path) -> None:
    """Refuse if this folder already holds a started marker or results."""
    for name in (STARTED, METRICS):
        if (out / name).exists():
            raise FileExistsError(
                f"{out / name} exists, so the test split was already opened. "
                "A rerun needs the marker deleted by hand and the reason logged"
            )


def claim_test_run(out: Path) -> None:
    """Refuse if this folder already holds a started marker or results, otherwise write the marker."""
    ensure_unclaimed(out)
    out.mkdir(parents=True, exist_ok=True)
    # exclusive mode, so two runs started at the same moment can't both claim it
    with open(out / STARTED, "x") as f:
        f.write("claimed\n")


def check_thresholds(thresholds: dict, saved_path: Path, rtol: float = THRESHOLD_RTOL) -> None:
    """Refuse unless every detector's thresholds match the saved ones, budget by budget."""
    saved = json.loads(saved_path.read_text())["thresholds"]
    if set(saved) != set(thresholds):
        raise ValueError(f"detectors differ from {saved_path}: missing {set(saved) - set(thresholds)}, extra {set(thresholds) - set(saved)}")
    for detector, by_budget in thresholds.items():
        ours = {f"{b:.0%}": t for b, t in by_budget.items()}
        if set(ours) != set(saved[detector]):
            raise ValueError(f"{detector}: budgets differ from {saved_path}")
        for budget, t in ours.items():
            if not np.isclose(t, saved[detector][budget], rtol=rtol, atol=0):
                raise ValueError(f"{detector} at {budget}: {t} today against {saved[detector][budget]} saved")


def attribution_row(split: str, seed: int, model: str, residual: np.ndarray, attention: np.ndarray, caught: bool | None) -> dict:
    """One run and model: both rankings by rise, and their overlaps with the ground truth and with each other."""
    fault = fault_for(split, seed)
    if fault is None:
        # The healthy control borrows the next seed's onset, and that fault's channels as the truth
        onset = control_onset(split, seed)
        truth = fault_for(split, seed + 1).channels
    else:
        onset, truth = fault.onset_s, fault.channels
    residual_top = top_k(rise(residual, onset))
    attention_top = top_k(rise(attention, onset))
    return {
        "seed": int(seed),
        "model": model,
        "faulted": fault is not None,
        "caught": caught,
        "onset_s": float(onset),
        "residual_top": list(residual_top),
        "attention_top": list(attention_top),
        "residual_overlap": overlap(residual_top, truth),
        "attention_overlap": overlap(attention_top, truth),
        "agreement": overlap(attention_top, residual_top),
    }


def summarize_attribution(rows: list[dict], models: list[str]) -> dict:
    """Per population and metric: each seed's n, mean and 95% bootstrap interval, then mean and std over seeds."""
    out = {}
    for population, keep in POPULATIONS.items():
        for metric in ATTRIBUTION_METRICS:
            per_seed = {}
            for m in models:
                values = np.array([r[metric] for r in rows if r["model"] == m and keep(r)])
                per_seed[m] = {
                    "n": int(values.size),
                    "mean": float(values.mean()) if values.size else None,
                    "interval": bootstrap_interval(values) if values.size else None,
                }
            means = [s["mean"] for s in per_seed.values() if s["mean"] is not None]
            out[f"{population}/{metric}"] = {
                "per_seed": per_seed,
                "mean": float(np.mean(means)) if means else None,
                "std": float(np.std(means, ddof=1)) if len(means) > 1 else None,
            }
    return out


def evaluate(split: str, models: dict, baselines: dict, std_scores: dict, thresholds: dict, limit: int | None) -> tuple[dict, dict, list]:
    """One pass over the split: every detector at every budget, healthy peaks, and attribution per run and model."""
    rows: dict[tuple[str, float], list] = {}
    healthy_peaks: dict[str, list[float]] = {}
    attribution: list[dict] = []
    for seed in SPLITS[split][:limit]:
        run = make_run(split, seed)
        scores, residuals = detector_scores(run, models, baselines, std_scores)
        caught_at_budget = {}
        for detector, s in scores.items():
            for b in BUDGETS:
                result = score_run(s, run.throttle, thresholds[detector][b], run.fault)
                rows.setdefault((detector, b), []).append((run.fault is None, result))
                if b == FA_BUDGET:
                    caught_at_budget[detector] = result.caught
            if run.fault is None:
                healthy_peaks.setdefault(detector, []).append(float(s.max()))
        for name, trained in models.items():
            attention = attention_by_channel(trained.model, trained.normalizer, run)
            # Caught by the judged line at 2%, the runs where a verdict would be shown
            caught = bool(caught_at_budget[f"{name}/{JUDGED}"]) if run.fault is not None else None
            attribution.append(attribution_row(split, seed, name, residuals[name], attention, caught))
    return {key: summarize(results) for key, results in rows.items()}, healthy_peaks, attribution


def provenance() -> dict:
    """The commit, whether the working tree had changes, and each model's weights hash: the frozen model on record."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    weights = {m: hashlib.sha256((RUNS_DIR / m / "model.pt").read_bytes()).hexdigest() for m in MODELS}
    return {"commit": commit, "uncommitted_changes": dirty, "model_sha256": weights}


def print_results(split: str, reports: dict, thresholds: dict, calibration: dict, attribution: dict) -> None:
    for b in BUDGETS:
        print(f"\n{split} at the {b:.0%} budget")
        print(f"{'detector':26s} {'recall':>7s} {'false alarm':>12s} {'transient':>10s} {'early':>7s} {'delay':>8s}")
        for detector in thresholds:
            rep = reports[(detector, b)]
            delay = f"{rep.mean_delay_s:.1f} s" if rep.mean_delay_s is not None else "n/a"
            print(
                f"{detector:26s} {rep.recall:7.1%} {rep.false_alarm_rate:12.1%} "
                f"{rep.transient_false_alarm_rate:10.1%} {rep.early_alarm_rate:7.1%} {delay:>8s}"
            )

    print(f"\nSeeds at {FA_BUDGET:.0%}, mean and spread over {len(MODELS)} seeds")
    for how in (JUDGED, SECONDARY):
        recall = [reports[(f"{m}/{how}", FA_BUDGET)].recall for m in MODELS]
        fa = [reports[(f"{m}/{how}", FA_BUDGET)].false_alarm_rate for m in MODELS]
        label = "judged" if how == JUDGED else SECONDARY_LABEL
        print(
            f"  {how:4s} ({label}): recall {np.mean(recall):.1%} ± {np.std(recall, ddof=1):.1%}, "
            f"seeds {', '.join(f'{r:.1%}' for r in recall)}. False alarms {', '.join(f'{f:.1%}' for f in fa)}"
        )

    print(f"\nCalibration exit test on healthy {split} runs, judged on the {JUDGED} line, per seed")
    for how in (JUDGED, SECONDARY):
        for m in MODELS:
            cells = [f"q {q}: {calibration[f'{m}/{how}@{q}']['share']:.1%} ({'pass' if calibration[f'{m}/{how}@{q}']['passes'] else 'FAIL'})" for q in CALIBRATION_QS]
            print(f"  {m}/{how:4s} " + "   ".join(cells))
    passed = all(v["passes"] for k, v in calibration.items() if k.split("@")[0].endswith(f"/{JUDGED}"))
    print(f"  Exit test on the {JUDGED} line: {'PASS' if passed else 'FAIL'}")

    print("\nAttribution, top 3 overlap. Chance is 37.5%")
    for key, summary in attribution.items():
        seeds = "   ".join(
            f"n {s['n']}: {s['mean']:.1%} ({s['interval'][0]:.1%} to {s['interval'][1]:.1%})" if s["n"] else "n 0"
            for s in summary["per_seed"].values()
        )
        overall = f"{summary['mean']:.1%}" + (f" ± {summary['std']:.1%}" if summary["std"] is not None else "") if summary["mean"] is not None else "n/a"
        print(f"  {key:36s} {overall:>14s}   {seeds}")


def main(dry_run: int | None = None) -> None:
    split, out = plan(dry_run)
    # Fail before the slow work if the one run was already used
    if out is not None:
        ensure_unclaimed(out)

    models = {name: load_run(RUNS_DIR / name) for name in MODELS}
    baselines, std_scores = fit_detectors(models, dry_run)
    thresholds, calibrations = calibrate_thresholds(models, baselines, std_scores, dry_run)
    if out is not None:
        # Everything is seeded, so a mismatch means something drifted since D8. Stop before test is opened
        check_thresholds(thresholds, D8_METRICS)
        print(f"Thresholds match {D8_METRICS}")

    record = provenance()
    if out is not None:
        claim_test_run(out)
    with open_test_split() if split == "test" else nullcontext():
        reports, healthy_peaks, rows = evaluate(split, models, baselines, std_scores, thresholds, dry_run)

    detectors = [f"{m}/{how}" for m in MODELS for how in (JUDGED, SECONDARY)]
    calibration = calibration_exit_test(calibrations, healthy_peaks, detectors)
    attribution = summarize_attribution(rows, list(MODELS))
    print_results(split, reports, thresholds, calibration, attribution)

    if out is not None:
        metrics = {
            "split": split,
            "provenance": record,
            "thresholds": {d: {f"{b:.0%}": t for b, t in tb.items()} for d, tb in thresholds.items()},
            "reports": {f"{d}@{b:.0%}": asdict(rep) for (d, b), rep in reports.items()},
            "judged": JUDGED,
            "secondary": {"line": SECONDARY, "label": SECONDARY_LABEL},
            "calibration_exit_test": calibration,
            "attribution": attribution,
            "attribution_rows": rows,
        }
        (out / METRICS).write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"\nsaved {out / METRICS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="The one test run.")
    parser.add_argument("--dry-run", type=int, default=None, help="use val and the first N seeds per split, saving nothing")
    args = parser.parse_args()
    main(args.dry_run)
