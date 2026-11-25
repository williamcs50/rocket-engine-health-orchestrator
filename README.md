# Rocket Engine Health Orchestrator

A research prototype that catches rocket engine faults hiding across many sensors at once, before any single sensor crosses its redline.

## The problem

Engine monitoring usually checks each sensor against its own redline. A degrading bearing shows up as vibration creeping up, a temperature trending warm and a pressure easing down, each still inside its limit. Redlines can't see that combined signature until the fault is well advanced.

## The approach

A reduced-order physics model generates engine telemetry and injects faults as physical changes, so their signatures emerge across channels on their own. Each of the 8 sensor channels is encoded separately, and an attention layer weighs the channels against each other and against the throttle command. The model is trained only on normal runs to forecast what each channel should do next, and when reality departs from the forecast, that gap is compared with the same run's recent past and with healthy runs at the same point in the throttle profile, and the result is the anomaly score. The output is an advisory verdict: a flag, a calibrated confidence and the channels that mattered most. It only advises, and redline protection keeps all authority.

> **Status:** v1 shipped Nov 23, 2026. The orchestrator passes its Gate 1 bar on a pre-registered held-out split: 58.0% of faults caught within 15 s of onset, with false alarms at most 2.6% per seed, against 35.0% for the best baseline and 0% for redlines. Daily log: [build log](docs/build-log.md) · [two-week plan](docs/two-week-plan.md)

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

```bash
# The three trained models are committed, so no training is needed.
# Reproduce the val results, about 35 minutes. It rewrites runs/gate1_d8/metrics.json,
# so `git diff` afterward shows whether every number matched
python -m evaluation.gate1
```

The test and test2 results can't be rerun. Each held-out split is locked in code and opens once. Their results are in `runs/test/metrics.json` and `runs/test2/metrics.json`. Retraining (`python -m model.train --seed 0`) gives close but not identical weights, and it overwrites the committed model in that folder.

## Results

At the 2% false alarm budget (at most 10 of 500 healthy runs alarm on calibrate). Recall is the share of faults caught within 15 s of onset, as the mean over 3 seeds. False alarms are per seed.

| Detector | Test2 recall | Test2 false alarms | Test recall | Val recall |
| --- | --- | --- | --- | --- |
| **Orchestrator, std score (v1)** | **58.0%** | 1.8%, 2.6%, 1.4% | 58.2% | 58.1% |
| Orchestrator, plain mean (D6 to D9) | 8.1% | 1.4%, 3.2%, 1.8% | 7.3% | 8.1% |
| Fair z score (conditioned on throttle) | 35.0% | 2.4% | 35.2% | 39.6% |
| D4 z score | 1.2% | 2.2% | 0.2% | 1.2% |
| Redline | 0.0% | 0.0% | 0.2% | 0.0% |

- **Gate 1** (mean recall at least 5%, every seed above 1.2%, every seed's false alarms at most 3%): the std score **passes on test2**. On test it cleared the bar but was reported for information only, because it had missed on val by one false alarm. The plain mean failed on val and on test.
- **Calibration:** the share of healthy runs with confidence at least q is within 3 points of 1 − q for q = 0.8, 0.9 and 0.95, on test and test2.
- **Attribution:** the channels with the largest residual rise overlap the fault's true channels at 62.2% on test2, against 42.9% on healthy runs and 37.5% by chance. Attention weights don't: 42.2%, against 43.9% on healthy runs.
- **False alarms (v1.1):** the std score's false alarms sit near the cap, at 1.4% to 3.2% per seed across splits. A pre-registered diagnostic on a fresh development split, val2, ruled out calibration offsets as the cause: offset and score swing have a Spearman correlation of about 0.06. On val2 the false alarms were 2.0%, 2.4% and 2.0%.
- **Command cross-attention (v1.2):** a pre-registered, matched ablation on val2 compared the same models trained on an all-zero throttle history. At mid-run throttle steps the command-aware models scored lower on every seed, but by 3.7% to 16.3%, against a 10% bar on every seed, so it isn't shown. False alarms during throttle transients fell from 86.2% to 78.1% of all false alarms, against a 20 point bar. Cross-attention lowers forecast error, but at the alarm level its effect is small.

**What was held out, and what wasn't:** val was used for three rounds of decisions, so val numbers are optimistic. Test was opened once, judged on the plain mean, with every decision committed beforehand. Test2 was pre-registered after test showed the std score clearing the bar: method, bar, seeds and a no-retry rule were committed before the split existed, and it was opened once.

**Limitations:** synthetic data from a reduced-order model, with one fault type (bearing wear). The verdict's ranked channels use the whole-run average residual, not the evaluated rise. The std score's false alarms sit close to the 3% cap, so whether a split passes is partly sampling. The cause isn't pinned down. Details are in the [decision log](docs/architecture.md#decision-log).

## Docs

* [Demo figure](docs/demo.png): one real caught fault, traced back to the engine's hardware
* [Demo site](https://williamopyrchal.com/orchestrator)
* [Design rationale](docs/design-doc.md)
* [Architecture](docs/architecture.md)
* [Two-week plan](docs/two-week-plan.md)
* [How It Works book](https://williamopyrchal.com/rocket-orchestrator-book)
