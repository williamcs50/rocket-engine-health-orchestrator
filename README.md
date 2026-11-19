# Rocket Engine Health Orchestrator

A research prototype that catches rocket engine faults hiding across many sensors at once, before any single sensor crosses its redline.

## The problem

Engine monitoring usually checks each sensor against its own redline. A degrading bearing shows up as vibration creeping up, a temperature trending warm and a pressure easing down, each still inside its limit. Redlines can't see that combined signature until the fault is well advanced.

## The approach

A reduced-order physics model generates engine telemetry and injects faults as physical changes, so their signatures emerge across channels on their own. Each of the 8 sensor channels is encoded separately, and an attention layer weighs the channels against each other and against the throttle command. The model is trained only on normal runs to forecast what each channel should do next, and when reality departs from the forecast, that gap becomes the anomaly score. The output is an advisory verdict: a flag, a calibrated confidence and the channels that mattered most. It only advises, and redline protection keeps all authority.

> **Status:** v1 in progress (Nov 10 to 23, 2026). Model trained and evaluated on validation. Gate 1 not yet passed: recall clears its bar, but false alarms on val exceed the 3% cap. Next: diagnosing startup transients. Daily log: [build log](docs/build-log.md) · [two-week plan](docs/two-week-plan.md)

## Installation

Requires Python 3.11+. Using conda:

```bash
git clone https://github.com/williamcs50/rocket-engine-health-orchestrator.git
cd rocket-engine-health-orchestrator
conda create -n reho python=3.11 -y
conda activate reho
pip install -r requirements.txt
```

Or with venv: `python -m venv .venv && source .venv/bin/activate`, then `pip install -r requirements.txt`.

Check the setup by running the fast tests (about 20 seconds, the same set CI runs):

```bash
pytest -m "not slow"
```

The slow exit tests train a model and take about 90 seconds: `pytest -m slow`.

## Quickstart

Interim, until v1. Train one model and evaluate it against the baselines:

```bash
# Train one model, about a minute. Saved to runs/h10_mask0.25_cmd_seed0/
python -m model.train --seed 0

# Evaluate on val against the redline and z score, about 25 minutes.
# It expects all three seeds (--seed 0, 1 and 2). Add --limit 20 for a quick check
python -m evaluation.gate1
```

## Results

Coming in v1.

## Docs

* [Case study](https://williamopyrchal.com/orchestrator-case-study)
* [Demo](https://williamopyrchal.com/orchestrator)
* [Design rationale](docs/design-doc.md)
* [Architecture](docs/architecture.md)
* [Two-week plan](docs/two-week-plan.md)
* [How It Works book](https://williamopyrchal.com/rocket-orchestrator-book)
