"""The held-out runs: test (D9) and test2 (D10). Decision log, 2026-11-19 (test run reporting), 2026-11-20
(attribution) and 2026-11-23 (test2).

Run with: python -m evaluation.final --split test2
Dry run on val, saving nothing: python -m evaluation.final --split test2 --dry-run 20
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
    BAR_FALSE_ALARM, BAR_MEAN_RECALL, BAR_SEED_RECALL, BUDGETS, CALIBRATION_QS, MODELS, OUT_NAME as D8_NAME,
    calibrate_thresholds, calibration_exit_test, detector_scores, fit_detectors,
)
from evaluation.harness import score_run, summarize
from generator.splits import LOCKED_SPLITS, SPLITS, fault_for, make_run, open_test_split
from model.forecaster import attention_by_channel
from model.train import RUNS_DIR, load_run

# Written before any test run is simulated, so a crash still counts as the test split having been opened
STARTED = "started"
METRICS = "metrics.json"
# The committed D8 results. The thresholds recomputed today must match them before test is opened
D8_METRICS = RUNS_DIR / D8_NAME / METRICS
# Float32 scoring can differ in the last digits between days. Any real drift is far larger
THRESHOLD_RTOL = 1e-6
# Per held-out split, the judged line and the secondary line, per the decision log: test (2026-11-19), test2 (2026-11-23)
LINES = {"test": ("mean", "std"), "test2": ("std", "mean")}
# How the secondary line is labeled on each split
SECONDARY_LABELS = {
    "test": "failed Gate 1 on val by one false alarm, not adopted, shown for information",
    "test2": "judged on test, where it failed Gate 1 by one false alarm, shown for information",
}
# Attribution populations: every faulted run, the faulted runs the judged line caught at 2%, and the healthy control
POPULATIONS = {
    "faulted": lambda row: row["faulted"],
    "caught": lambda row: row["faulted"] and row["caught"],
    "healthy control": lambda row: not row["faulted"],
}
ATTRIBUTION_METRICS = ("residual_overlap", "attention_overlap", "agreement")


def plan(dry_run: int | None, split: str = "test") -> tuple[str, Path | None]:
    """Which split to report on, and where to save. A dry run uses val and saves nothing."""
    if split not in LOCKED_SPLITS:
        raise ValueError(f"a real run is only for {LOCKED_SPLITS}, not {split!r}")
    if dry_run is not None:
        return "val", None
    return split, RUNS_DIR / split


def lines(split: str) -> tuple[str, str]:
    """The judged line and the secondary line for a held-out split."""
    return LINES[split]


def gate1_verdict(reports: dict, line: str, models: list[str]) -> tuple[bool, dict[str, bool]]:
    """The Gate 1 bar, unchanged since D6, on one line at the 2% budget. Each condition as readable text."""
    recall = [reports[(f"{m}/{line}", FA_BUDGET)].recall for m in models]
    fa = [reports[(f"{m}/{line}", FA_BUDGET)].false_alarm_rate for m in models]
    conditions = {
        # A tiny slack, since a mean of three shares that is exactly 5% can come out a hair under it in floating point
        f"mean recall {np.mean(recall):.1%} at least {BAR_MEAN_RECALL:.0%}": bool(np.mean(recall) >= BAR_MEAN_RECALL - 1e-9),
        f"lowest seed {min(recall):.1%} above {BAR_SEED_RECALL:.1%}": bool(min(recall) > BAR_SEED_RECALL),
        # The same slack, so a rate of exactly 3% passes as the bar says
        f"highest false alarm {max(fa):.1%} at most {BAR_FALSE_ALARM:.0%}": bool(max(fa) <= BAR_FALSE_ALARM + 1e-9),
    }
    return all(conditions.values()), conditions


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


def evaluate(
    split: str, models: dict, baselines: dict, std_scores: dict, thresholds: dict, limit: int | None, judged: str
) -> tuple[dict, dict, list]:
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
            caught = bool(caught_at_budget[f"{name}/{judged}"]) if run.fault is not None else None
            attribution.append(attribution_row(split, seed, name, residuals[name], attention, caught))
    return {key: summarize(results) for key, results in rows.items()}, healthy_peaks, attribution


def provenance() -> dict:
    """The commit, whether the working tree had changes, and each model's weights hash: the frozen model on record."""
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    weights = {m: hashlib.sha256((RUNS_DIR / m / "model.pt").read_bytes()).hexdigest() for m in MODELS}
    return {"commit": commit, "uncommitted_changes": dirty, "model_sha256": weights}


def print_results(
    split: str, reports: dict, thresholds: dict, calibration: dict, attribution: dict, target: str, gate: tuple[bool, dict]
) -> None:
    judged, secondary = lines(target)
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
    for how in (judged, secondary):
        recall = [reports[(f"{m}/{how}", FA_BUDGET)].recall for m in MODELS]
        fa = [reports[(f"{m}/{how}", FA_BUDGET)].false_alarm_rate for m in MODELS]
        label = "judged" if how == judged else SECONDARY_LABELS[target]
        print(
            f"  {how:4s} ({label}): recall {np.mean(recall):.1%} ± {np.std(recall, ddof=1):.1%}, "
            f"seeds {', '.join(f'{r:.1%}' for r in recall)}. False alarms {', '.join(f'{f:.1%}' for f in fa)}"
        )

    passed, conditions = gate
    print(f"\nGate 1 on the judged {judged} line: {'PASS' if passed else 'FAIL'}")
    for text, ok in conditions.items():
        print(f"  {'yes' if ok else 'no '}  {text}")

    print(f"\nCalibration exit test on healthy {split} runs, judged on the {judged} line, per seed")
    for how in (judged, secondary):
        for m in MODELS:
            cells = [f"q {q}: {calibration[f'{m}/{how}@{q}']['share']:.1%} ({'pass' if calibration[f'{m}/{how}@{q}']['passes'] else 'FAIL'})" for q in CALIBRATION_QS]
            print(f"  {m}/{how:4s} " + "   ".join(cells))
    exit_passed = all(v["passes"] for k, v in calibration.items() if k.split("@")[0].endswith(f"/{judged}"))
    print(f"  Exit test on the {judged} line: {'PASS' if exit_passed else 'FAIL'}")

    print("\nAttribution, top 3 overlap. Chance is 37.5%")
    for key, summary in attribution.items():
        seeds = "   ".join(
            f"n {s['n']}: {s['mean']:.1%} ({s['interval'][0]:.1%} to {s['interval'][1]:.1%})" if s["n"] else "n 0"
            for s in summary["per_seed"].values()
        )
        overall = f"{summary['mean']:.1%}" + (f" ± {summary['std']:.1%}" if summary["std"] is not None else "") if summary["mean"] is not None else "n/a"
        print(f"  {key:36s} {overall:>14s}   {seeds}")


def main(dry_run: int | None = None, split: str = "test") -> None:
    # The held-out split this run is for. A dry run reads val in its place
    target = split
    split, out = plan(dry_run, target)
    judged, secondary = lines(target)
    # Fail before the slow work if the one run was already used
    if out is not None:
        ensure_unclaimed(out)

    models = {name: load_run(RUNS_DIR / name) for name in MODELS}
    baselines, std_scores = fit_detectors(models, dry_run)
    thresholds, calibrations = calibrate_thresholds(models, baselines, std_scores, dry_run)
    if out is not None:
        # Everything is seeded, so a mismatch means something drifted since D8. Stop before the split is opened
        check_thresholds(thresholds, D8_METRICS)
        print(f"Thresholds match {D8_METRICS}")

    record = provenance()
    if out is not None:
        claim_test_run(out)
    with open_test_split(split) if split in LOCKED_SPLITS else nullcontext():
        reports, healthy_peaks, rows = evaluate(split, models, baselines, std_scores, thresholds, dry_run, judged)

    gate = gate1_verdict(reports, judged, list(MODELS))
    detectors = [f"{m}/{how}" for m in MODELS for how in (judged, secondary)]
    calibration = calibration_exit_test(calibrations, healthy_peaks, detectors)
    attribution = summarize_attribution(rows, list(MODELS))
    print_results(split, reports, thresholds, calibration, attribution, target, gate)

    if out is not None:
        metrics = {
            "split": split,
            "provenance": record,
            "thresholds": {d: {f"{b:.0%}": t for b, t in tb.items()} for d, tb in thresholds.items()},
            "reports": {f"{d}@{b:.0%}": asdict(rep) for (d, b), rep in reports.items()},
            "judged": judged,
            "secondary": {"line": secondary, "label": SECONDARY_LABELS[target]},
            "gate1": {"line": judged, "passed": gate[0], "conditions": gate[1]},
            "calibration_exit_test": calibration,
            "attribution": attribution,
            "attribution_rows": rows,
        }
        (out / METRICS).write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"\nsaved {out / METRICS}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A held-out run, test or test2. Each is opened once.")
    parser.add_argument("--split", choices=LOCKED_SPLITS, default="test", help="the held-out split to run")
    parser.add_argument("--dry-run", type=int, default=None, help="use val and the first N seeds per split, saving nothing")
    args = parser.parse_args()
    main(args.dry_run, args.split)
