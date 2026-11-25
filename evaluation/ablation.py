"""Does command cross-attention move false alarms out of throttle transients? Decision log, 2026-11-25 (ablation).

Run with: python -m evaluation.ablation
Quick check of the wiring: python -m evaluation.ablation --limit 20
"""

import argparse
import hashlib
import json
import time

import numpy as np

from evaluation.budget import FA_BUDGET
from evaluation.final import D8_METRICS, provenance
from evaluation.gate1 import MODELS, calibrate_thresholds, fit_detectors
from evaluation.harness import RunResult, score_run, summarize
from evaluation.score import mean_residual
from generator.splits import SPLITS, make_run
from model.forecaster import residual_by_channel
from model.train import RUNS_DIR, load_run

# The development split this ablation reads
SPLIT = "val2"
# The D12 ablation models, matched seed for seed with D7's
NOCMD_MODELS = tuple(m.replace("_cmd_", "_nocmd_") for m in MODELS)
# Pre-registered bar: the command-aware pooled transient share is at least this much lower than the ablation's
BAR_SHARE_MARGIN = 0.20
# Pre-registered bar: the command-aware mean recall is at most this much lower than the ablation's
BAR_RECALL_DROP = 0.03
OUT = RUNS_DIR / "ablation"


def transient_counts(rows: list[tuple[bool, RunResult]]) -> tuple[int, int]:
    """(transient false alarm runs, false alarm runs) over the healthy rows. Each row is (run is healthy, result)."""
    healthy = [r for is_healthy, r in rows if is_healthy]
    return sum(r.transient_false_alarm for r in healthy), sum(r.false_alarm for r in healthy)


def share(transient: int, false_alarms: int) -> float | None:
    """The share of false alarm runs that are transient. None when there are no false alarms."""
    return transient / false_alarms if false_alarms else None


def pooled_share(counts: list[tuple[int, int]]) -> float | None:
    """Every seed's transient false alarm runs over every seed's false alarm runs."""
    return share(sum(t for t, _ in counts), sum(f for _, f in counts))


def ablation_verdict(
    cmd: list[tuple[int, int]], nocmd: list[tuple[int, int]], cmd_recall: list[float], nocmd_recall: list[float]
) -> tuple[bool, dict[str, bool]]:
    """Both pre-registered conditions, each as readable text. cmd and nocmd are matched seed for seed."""
    seed_shares = [(share(*c), share(*n)) for c, n in zip(cmd, nocmd)]
    pooled_cmd, pooled_nocmd = pooled_share(cmd), pooled_share(nocmd)
    # A model with no false alarms has no share, and the first condition then fails
    defined = pooled_cmd is not None and pooled_nocmd is not None and all(c is not None and n is not None for c, n in seed_shares)
    # A tiny slack, so a drop of exactly 20 points passes as the bar says
    dropped = defined and pooled_nocmd - pooled_cmd >= BAR_SHARE_MARGIN - 1e-9
    every_seed = defined and all(c < n for c, n in seed_shares)

    def pct(x: float | None) -> str:
        return "none" if x is None else f"{x:.1%}"

    seeds = ", ".join(f"{pct(c)} against {pct(n)}" for c, n in seed_shares)
    recall_c, recall_n = float(np.mean(cmd_recall)), float(np.mean(nocmd_recall))
    conditions = {
        f"pooled transient share {pct(pooled_cmd)} against {pct(pooled_nocmd)}, at least {BAR_SHARE_MARGIN:.0%} lower, "
        f"and lower on every seed ({seeds})": dropped and every_seed,
        f"mean recall {recall_c:.1%} against {recall_n:.1%}, at most {BAR_RECALL_DROP:.0%} lower":
            recall_c >= recall_n - BAR_RECALL_DROP - 1e-9,
    }
    return all(conditions.values()), conditions


def main(limit: int | None = None) -> None:
    models = {name: load_run(RUNS_DIR / name) for name in (*MODELS, *NOCMD_MODELS)}
    for name in NOCMD_MODELS:
        if models[name].model.command_input:
            raise ValueError(f"{name} reads the throttle, so it isn't the ablation")
    _, std_scores = fit_detectors(models, limit)
    thresholds, _ = calibrate_thresholds(models, {}, std_scores, limit)
    if limit is None:
        # The command-aware lines must be v1's std score, with D8's thresholds
        saved = json.loads(D8_METRICS.read_text())["thresholds"]
        for name in MODELS:
            if not np.isclose(thresholds[f"{name}/std"][FA_BUDGET], saved[f"{name}/std"]["2%"], rtol=1e-6, atol=0):
                raise ValueError(f"{name}/std threshold differs from D8's")
        print(f"Std thresholds match {D8_METRICS}")

    # Model to [(run is healthy, result), ...], and one compact row per run and model for the record
    rows: dict[str, list[tuple[bool, RunResult]]] = {name: [] for name in models}
    record = []
    seeds = SPLITS[SPLIT][:limit]
    began = time.perf_counter()
    for i, seed in enumerate(seeds):
        # Simulated once, shared by all six models
        run = make_run(SPLIT, seed)
        for name, trained in models.items():
            r = residual_by_channel(trained.model, trained.normalizer, run, masked=True)
            # The std line exactly as detector_scores scored test2
            s = std_scores[name].scores(mean_residual(r), run.throttle)
            result = score_run(s, run.throttle, thresholds[f"{name}/std"][FA_BUDGET], run.fault)
            rows[name].append((run.fault is None, result))
            record.append({
                "seed": int(seed), "model": name, "healthy": run.fault is None, "false_alarm": result.false_alarm,
                "transient": result.transient_false_alarm, "caught": result.caught,
            })
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} of {len(seeds)} runs, {time.perf_counter() - began:.0f} s")

    reports = {name: summarize(r) for name, r in rows.items()}
    counts = {name: transient_counts(r) for name, r in rows.items()}
    passed, conditions = ablation_verdict(
        [counts[m] for m in MODELS], [counts[m] for m in NOCMD_MODELS],
        [reports[m].recall for m in MODELS], [reports[m].recall for m in NOCMD_MODELS],
    )

    print(f"\nAblation on {SPLIT} at the {FA_BUDGET:.0%} budget, std score")
    print(f"{'model':28s} {'threshold':>9s} {'recall':>7s} {'false alarm':>12s} {'transient':>10s} {'share':>7s}")
    for name in models:
        rep, (t, f) = reports[name], counts[name]
        s = share(t, f)
        print(
            f"{name:28s} {thresholds[f'{name}/std'][FA_BUDGET]:9.3f} {rep.recall:7.1%} {rep.false_alarm_rate:12.1%} "
            f"{rep.transient_false_alarm_rate:10.1%} {'none' if s is None else f'{s:.1%}':>7s}"
        )
    print(f"\nHypothesis: {'HOLDS' if passed else 'DOES NOT HOLD'}")
    for text, ok in conditions.items():
        print(f"  {'yes' if ok else 'no '}  {text}")

    # Quick checks aren't saved, so a 20 run result can never be committed as the ablation's numbers
    if limit is None:
        OUT.mkdir(parents=True, exist_ok=True)
        metrics = {
            "split": SPLIT,
            "provenance": provenance(),
            # provenance hashes the D7 weights only
            "nocmd_sha256": {m: hashlib.sha256((RUNS_DIR / m / "model.pt").read_bytes()).hexdigest() for m in NOCMD_MODELS},
            "thresholds": {name: thresholds[f"{name}/std"][FA_BUDGET] for name in models},
            "reports": {name: {k: v for k, v in vars(rep).items()} for name, rep in reports.items()},
            "transient_counts": {name: {"transient": t, "false_alarms": f} for name, (t, f) in counts.items()},
            "holds": passed,
            "conditions": conditions,
            "rows": record,
        }
        (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(f"\nsaved {OUT / 'metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Does command cross-attention move false alarms out of transients?")
    parser.add_argument("--limit", type=int, default=None, help="use only the first N seeds per split (quick check)")
    args = parser.parse_args()
    main(args.limit)
