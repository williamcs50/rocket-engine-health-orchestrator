# Build Log

One entry per session, newest first. Full decision entries live in [architecture.md](architecture.md).

## D6 · Tue Nov 17, 2026 · P2 evaluate

**Exit test (Gate 1):** failed. Recall clears its bar, but false alarms on val exceed the 3% cap for both aggregations.
- Bar, pre-registered in architecture.md before any val evaluation: at the 2% budget on val, the 3 masked seeds' mean recall is at least 5%, every seed's recall is above the D4 z score's 1.2%, and every seed's false alarm rate is at most 3%.

| At the 2% budget | Mean | Max |
| --- | --- | --- |
| Recall, seeds 0, 1, 2 | 10.0%, 15.4%, 9.4% | 3.8%, 9.8%, 9.0% |
| Mean recall over 3 seeds (bar 5%) | 11.6% ± 3.3%, passes | 7.5% ± 3.3%, passes |
| Lowest seed (bar above 1.2%) | 9.4%, passes | 3.8%, passes |
| False alarms, seeds 0, 1, 2 (bar at most 3%) | 3.4%, 3.4%, 2.8%, fails | 3.2%, 2.0%, 3.6%, fails |
| Mean delay | 11.9 to 12.2 s | 11.7 to 12.7 s |

- Baselines at the 2% budget: redline 0.0% recall, z score 1.2% recall at 0.6% false alarms. Unmasked comparison: 2.0% recall with mean, 4.2% with max.
- Aggregation chosen: mean. Higher recall on every seed, and the same pass or fail on each condition.
- Reproduce with `python -m evaluation.gate1` (about 22 minutes). Every number above is in the committed `runs/gate1/metrics.json`.

### Built
- `model/train.py`: training with separate random streams for batches, masks and the probe, `random_mask` (never hides all eight, rejects rates outside 0 to 1), a probe loss curve on one fixed batch from before the first step, `TrainResult` carrying every setting used, `save_run` and `load_run` (normalizer and architecture in a committed config.json, weights local only, refuses a mismatched architecture), and a command line entry
- `Normalizer.to_dict` and `from_dict`, so the statistics live in each run's config
- `model/forecaster.py`: one `aggregate` function shared by `residual_scores` and `to_verdict`, a mean or max switch, a `masked` switch on `to_verdict`, and a plain `bool` flag
- `evaluation/gate1.py`: thresholds from calibrate's healthy runs, every detector on val at 2%, 1% and 5%, where healthy peaks happen, the bearing temp bias check, results saved to `runs/gate1/metrics.json`
- `tests/test_train.py`: 13 fast tests (mask rate and never all, bad rates, settings recorded, run names, save and load round trip, architecture mismatch refused)
- Trained 3 masked seeds and the unmasked comparison, 10,000 steps each, 43 to 49 s per model (`seconds` in each `runs/*/metrics.json`)
- CI runs the fast tests only. 153 fast tests pass

### Decided
- CI runs `pytest -m "not slow"`. The slow exit tests run locally and their numbers go in this log
- Training steps raised from 3,000 to 10,000, because at 3,000 feed pressure missed its half the gap target
- Gate 1 bar pre-registered (above), replacing "beats the redline", which the redline's 0% recall made passable by one lucky catch
- Aggregation: mean, chosen on val (11.6% against max's 7.5%). Its val recall is slightly optimistic, since it's the better of two
- Gate 1 failed, so per the plan D7 goes to fixing and the GRU experiment is dropped. Which fix comes first is open, to settle before D7 starts
- The training loss curve is measured on one fixed probe batch, starting before any training, so the exit test compares against the true starting loss
- Masks, batches and the probe each get their own random stream, so the masked and unmasked runs train on the same windows

### Training curves
| Run | Probe loss at start | Probe loss at end |
| --- | --- | --- |
| masked, seed 0 | 1008 | 8.4 |
| masked, seed 1 | 1199 | 9.0 |
| masked, seed 2 | 2825 | 7.9 |
| unmasked, seed 0 | 894 | 1.6 |

The masked seeds end at 8.5 ± 0.5 (mean and sample std over 3 seeds), so training is stable across seeds. The unmasked model ends at 1.6, close to 1.0, the noise floor itself. The masked loss is a blend: hidden channels miss by much more, since a hidden channel's per-run level is invisible to the others, and visible ones by much less. Each seed draws its own probe batch, so the end values compare only roughly across seeds.

Reproduce:
- Retrain: `python -m model.train --seed 0` (and 1, 2), and `python -m model.train --mask-prob 0 --seed 0`. Each prints its start and end probe loss
- Without retraining, from the committed `runs/*/metrics.json`:
  `python -c "import json,glob; [print(p, round(m['losses'][0],1), round(m['losses'][-1],1)) for p in sorted(glob.glob('runs/h10_*/metrics.json')) for m in [json.load(open(p))]]"`

### Expected vs actual
| Expected | Actual | Why |
| --- | --- | --- |
| 3,000 steps clear the D5 forecast bar | Feed pressure missed its half the gap target. 10,000 steps pass (`pytest -m slow tests/test_model.py -s`) | Undertrained. The 3,000 step run used a draft of train.py that was never committed, so its exact numbers can't be reproduced and aren't recorded |
| Per-run bias in the masked bearing temp residual is comparable to the fault's rise, so a differenced score is needed | Partly. Over 500 healthy val runs the level's std is 31.9 and its 5th to 95th percentile 4.9 to 105.2. The fault's rise is 117.5 on average, median 85.1. A 20 run quick check had suggested 10 times the spread, which didn't hold | The typical run is fine, but the 5% of healthy runs with the largest bias sit above the median fault's rise, so bias matters at the tail. The differenced score stays a candidate |
| Gate 1 passes, but barely. I think the masked mean recall lands around 7% at the 2% budget, lower than the quick check's 17% because 500 healthy runs push the thresholds higher. Mean beats max, since max picks up the full size of every throttle spike (written before the full run's output) | Failed. Recall was higher than predicted, 11.6%, and mean did beat max. But false alarms reached 3.4% on two seeds, over the 3% cap | The prediction didn't consider whether thresholds carry from calibrate to val. They're set by rare transient spikes, a heavy tail (seed 2's threshold is 256 at 2% but 344 at 1%), so 500 runs pin the 2% point only roughly. All three seeds were calibrated on the same 500 healthy runs, which would make them overshoot together. A likely explanation, not a reason to pass |

### Learned
- Throttle transients appear to set the model's thresholds. All 10 sampled healthy calibrate peaks were within 5 s of a throttle change, mostly on fast channels (`python -m evaluation.gate1`, which samples 10 runs, so this is still a small sample). The model doesn't read the throttle until D7, so every commanded step surprises it. Mean thresholds are 202 to 256 at 2%, while a bearing fault's rise is divided by 8 in the mean, which fits catches arriving late, about 12 s into the 15 s window.
- Masking pays off. Masked models catch 11.6% on average with mean, the unmasked model 2.0%.
- A 20 run check can mislead. It suggested bias was a tenth of the fault's rise, and the full run showed bias matters at the tail.
- A false alarm cap catches what a recall bar can't. Recall passed easily, and the thresholds still didn't carry from calibrate to val.
- A guard inside a function only protects its callers if every path calls it. `random_mask` rejected NaN, but `train` only called it when `mask_prob > 0`, which NaN fails, so a NaN rate would have trained an unmasked model silently.

### Ledger
- **Built by hand:** `random_mask`, the core of `train` (random streams, normalizing and cutting runs, the training loop), and in `gate1.py`: `time_one_run`, `healthy_seeds`, `calibrate_thresholds` and the start of `main`
- **Delegated, can explain:** _confirm: `forecast_run`, `residual_by_channel`, `residual_scores`, `to_verdict` (walked through today)_
- **Delegated, not yet understood:** `save_run` and `load_run` (`weights_only`, the architecture check), the probe curve, `Normalizer.to_dict` and `from_dict`, `aggregate` and its tests, `tests/test_train.py`, and in `gate1.py`: `peak_times`, `evaluate_val`, `bias_check` and the printing and saving in `main`
- **Moved:** none

### Moved to tomorrow
- Walk through everything in "not yet understood", starting with `load_run`
- Fair z score fix, still due before Nov 20

### Next
Wed 9:00, D7 goes to fixing, per the Gate 1 bar, and the GRU experiment is dropped:
- First, choose the fix with the tutor. Candidates: command awareness (cross attention to the throttle history, aimed at the transient peaks that set thresholds and overshoot on val) and the differenced residual (aimed at per-run bias at the tail)
- Rerun `python -m evaluation.gate1` against the same bar. The bar doesn't move

## D5 · Mon Nov 16, 2026 · P1 and P2 model

**Exit test:** passed on D6, once `model/train.py` existed. All four slow exit tests pass at 10,000 training steps (loss falls on nominal data, beats persistence per channel, fault residuals rise, end-to-end verdict).
- At 3,000 steps, feed pressure missed its half the gap target. That run used an uncommitted draft of train.py, so its numbers aren't recorded. At 10,000 steps every channel passes.
- Per channel on 100 healthy val runs, normalized squared error, model against persistence: chamber pressure 0.0009 against 0.0024, speed 0.0006 against 0.0018, vibration 0.0051 against 0.0083, bearing temp 0.0066 against 0.0096, nozzle temp 0.0014 against 0.0023, coolant flow 0.0009 against 0.0024, feed pressure 0.0012 against 0.0021.
- Reproduce with `pytest -m slow tests/test_model.py -s` (about 60 s).

### Built
- `model/data.py`: `Normalizer` (train stats and noise floors, rejects faulted runs, near-zero spread and empty input, read-only arrays) and `make_windows` (views, not copies, with stride and horizon guards)
- `model/encoder.py`: one MLP shared by all 8 channels, (batch, 8, 100) to (batch, 8, 16)
- `model/forecaster.py`: channel IDs, a learned missing vector for masking, transformer fusion (4 heads, feed-forward 16 to 64 to 16), a per-channel head, the noise floor loss, and `forecast_run` scoring each channel with only its own input masked, in chunks
- `residual_scores` and a placeholder `to_verdict`
- `evaluation/checks.py`: reproduces the z-score false-alarm check, peak timing, persistence error and noise floors
- `tests/test_data.py` and `tests/test_model.py`: 36 fast tests, plus 4 slow exit tests that need `train.py`
- A test that thresholds from `healthy_peaks` and false alarms from `score_run` flag the same runs
- `healthy_peaks` takes optional seeds, so that test stays fast

### Decided
- 2% stays the primary operating point. The redline comparison is reported at the 1% secondary point, not 0%
- Z-score baseline limitation logged: its threshold is set by startup. Fix before Nov 20
- Forecast bar per channel: beat persistence on the 7 dynamic channels, close half the gap to the noise floor on speed, chamber pressure, coolant flow and feed pressure. Mixture ratio reported against its floor
- Training loss divides each channel's error by its noise floor, so every channel counts equally
- Each window sees telemetry up to its end and the throttle up to the target time
- Channel masking: a masked channel's embedding becomes a learned missing vector, training masks each channel with probability 0.25 (never all), and scoring forecasts each channel with only its own input masked
- The persistence bar uses unmasked forecasts. Masked forecasts are for fault scoring
- Flat-channel check uses a relative tolerance of 1e-6, because rounding leaves flat channels a std near 1e-8

### Expected vs actual
| Expected | Actual | Why |
| --- | --- | --- |
| With channel masking, fault residuals rise above the healthy twin's in at least 80% of faulted runs (written before training) | Matched. The exit test passed, so at least 16 of 20 rose | Masked forecasts come from the other channels, so a bearing wearing out shows as disagreement |
| Without masking, fault residuals rise in fewer than 80% of faulted runs, because a forecaster reading a channel's own history continues slow drifts (written before training) | Not measured. The exit test ran on the masked model only | The unmasked comparison model exists now, so this can be checked |

### Learned
A forecaster that reads a channel's own recent history simply continues a slow drift, so its error barely moves when the bearing wears. Hiding each channel's own input forces the forecast to come from the other channels, and that's where a fault shows: one channel inconsistent with the rest.

### Weekly review (9:00 to 10:00, moved from Saturday)
- Ledger check: done. Nothing left in "not yet understood", every move recorded
- Reread the D1 to D4 build logs: _done or skipped_
- Redo one hand-built piece from memory and compare with the real code: _which piece, and how close, or skipped_
- Update the case study's status line: _done or skipped_
- Anything skipped carries to Saturday's review

### Ledger
- **Built by hand:** `model/encoder.py`, `model/data.py` (`fit`, `transform`, `make_windows`), `Forecaster` (`__init__` with the transformer fusion, `tokens`, `forward`), `floor_weighted_mse`
- **Delegated, can explain:** none
- **Delegated, not yet understood:** `forecast_run` (stacked masking), `residual_by_channel`, `residual_scores`, `to_verdict`, `evaluation/checks.py`, `tests/test_data.py`, `tests/test_model.py`, the threshold consistency test in `tests/test_harness.py`, the `Normalizer` fixes (read-only arrays, flat-channel tolerance) and the `make_windows` guards
- **Moved:** none

### Moved to tomorrow
- `model/train.py` and the four slow D5 exit tests: moved to D6, before Gate 1
- Walk through everything in "not yet understood", starting with `forecast_run`

### Next
Tue 9:00, before Gate 1:
- Masked-residual bias check on bearing temp: compare (a) the spread of masked residuals across healthy val runs with (b) the rise the fault causes within 15 s of onset. If (a) is comparable to (b), switch `residual_scores` to the change in the masked residual over the previous 15 to 20 s, and log it. Training doesn't change
- Then Gate 1: the model against the redline at the 2% budget on val, plus the redline comparison at the 1% point

## D4 · Fri Nov 13, 2026 · Baselines and eval harness

**Exit test:** passed. The harness reports recall, detection delay and transient false alarms for both baselines on val.
- Redline: recall 0.0%, false alarms 0.0%.
- Z-score at the 2% budget: recall 1.2%, mean delay 13.8 s, false alarms 0.6%.
- The redline missed every coupled fault within the 15 s window.
- 95 fast tests plus 1 slow test pass.
- Reproduce with `pytest -s -m slow` (about 80 s). The numbers print from the exit test.

### Built
- `evaluation/budget.py`: the pre-registered rules in one place (budget, secondary points, catch window, transient window, threshold and report splits)
- `evaluation/baselines.py`: redline baseline (fixed at 1.0), per-channel z-score fitted on train, persistence forecast
- `evaluation/harness.py`: thresholds from healthy peaks, per-run scoring (false alarm, early alarm, transient, catch, delay, events), `evaluate()` over a whole split
- `evaluation/plots.py`: a healthy run and a faulted run against redlines, with the healthy twin, onset and catch window
- `tests/test_budget.py`, `tests/test_baselines.py`, `tests/test_harness.py`, including the D4 exit test marked `slow`
- `slow` marker registered in `pyproject.toml`, matplotlib and its dependencies pinned in `requirements.txt`
- Fault magnitudes pinned by a new test in `tests/test_faults.py`
- D2 and D3 build log entries finished, architecture.md brought up to date, semicolons removed across the docs

### Decided
- Fault magnitudes locked: `RATE_RANGE` 0.04 to 0.06 per s, `K_VIB` 0.2, `K_LOAD` 0.02
- Val, calibrate and test grow to 1,000 runs each (500 healthy, 500 faulted)
- False-alarm budget: at most 2% of healthy runs, counted per run over the whole run. 1% and 5% pre-registered as secondary points
- Thresholds set on calibrate's healthy runs, results reported on val, test opened once on D9
- A catch is a first alarm within onset + 15 s. An alarm before onset disqualifies the catch but stays out of the 2% rate
- Headline comparison: orchestrator against redline at the redline's own false-alarm rate
- `TRANSIENT_S` = 5 s after any throttle change. Transient false alarms count toward the budget and are reported separately
- Rejected a 5% budget: it invites "of course you catch more" against a redline at 0%

### Expected vs actual
| Expected | Actual | Why |
| --- | --- | --- |
| Redline misses most coupled faults within 15 s of onset | Missed all 500 (recall 0.0%) | Matched. The faults were tuned on D3 to stay under every redline |
| A per-channel z-score catches more than the redline | Caught 6 of 500 (1.2%) at 0.6% false alarms | Barely. It compares each channel to one mean for the whole run, so normal throttle changes, startup and shutdown included, already sit many sigma away. The threshold has to sit above all of that, far above the fault's shift. All of its false alarms were transient |
| A 2% budget makes a fair comparison against the redline | It invites "of course you catch more" | The redline's false-alarm rate is 0%, so the headline comparison moved to the redline's own rate |
| 150 healthy calibrate runs can set a 2% threshold | Only about 3 runs would set it | Grew val, calibrate and test to 1,000 runs before any evaluation |
| A test skeleton with `pass` bodies would show as failing | All 6 baseline tests "passed" | A test with no assert always passes. Real bodies replaced them |
| matplotlib is installed in `reho` | `ModuleNotFoundError` on the first plot run | It was only in base Python. Installed in `reho` and pinned in requirements.txt |

### Learned
- A false-alarm budget is only a fair comparison if both detectors run at the same rate.
- A test with no assertion passes whether or not the code works.

### Ledger
- **Built by hand:** `evaluation/baselines.py`, `evaluation/harness.py` (scoring functions), `evaluation/plots.py`
- **Delegated, can explain:** `evaluation/budget.py`, `evaluate()` and `healthy_peaks()`, `tests/test_harness.py`, `tests/test_baselines.py`, `tests/test_budget.py`
- **Delegated, not yet understood:** none
- **Moved:** none

### Moved to tomorrow
- the site build log update
- the channel lag decision, before Monday's training

### Next
Sat 9:00:
- Weekly review, including the channel lag decision and the site build log update
- Write scaled dot-product attention by hand and check it matches `nn.MultiheadAttention` to 1e-5

## D3 · Thu Nov 12, 2026 · P0 faults and splits

**Exit test:** passed. At least 80% of 50 fault runs stay under every redline for 15 s after onset, and split seeds never overlap: 76/76 tests in `reho`.

### Built
- `simulate()` rewritten to the book's [chapter 6](https://williamopyrchal.com/rocket-orchestrator-book#ch-06): coolant flow from feed pressure, bearing and nozzle temps as heat balances, mixture ratio constant, calibration offsets up to 3 sigma
- Bearing friction `mu` coupled into bearing heating, vibration and pump speed
- `FaultLabel` contract. `Run` gains `fault`, copies its arrays and makes them read-only
- `make_fault()` and the `fault` parameter in `simulate()`
- `generator/splits.py`: disjoint seed ranges, odd seeds faulted in val, calibrate and test
- `tests/test_faults.py` (6 tests), `tests/test_splits.py` (6 tests), new contract tests
- `test_different_seeds_differ` now runs with `noise=False`. Nozzle temp added to the throttle-step test

### Decided
- Chamber pressure follows flow, not throttle directly (departs from the book)
- Nozzle cooling grows with Q^0.8 plus a slow ambient leak, so nozzle temp varies with throttle
- Friction couples into vibration and speed (extends the book's table, per the napkin)
- Bearing wear: friction rises linearly from onset. Onset 10 to 30 s, rate 0.04 to 0.06 per s (raised from 0.03)
- Ground-truth channels: bearing temp, vibration, pump speed (the direct ones)
- Sub-threshold window N = 15 s. At least 80% of fault runs must stay under redlines
- Visibility floors per channel: bearing temp 5 sigma, vibration 2 sigma
- Calibrate split holds faulted runs too, keeping both calibration methods open
- Every contract violation raises `ValueError`

### Expected vs actual
| Expected | Actual | Why |
| --- | --- | --- |
| Yesterday's passing generator was a sound model | Every channel was a function of pump speed, so a bearing fault would hit one channel | Tests checked behavior, not structure |
| `test_different_seeds_differ` checks engine variation | It would pass with identical engines: noise alone made runs differ | Fixed with `noise=False` |
| Restructure, flow and heat-balance steps leave output unchanged | Matched to within 1e-13 (floating-point rounding) | Refactors done one step at a time |
| Visibility test fails on vibration at low throttle | Failed on vibration, 2.3 to 3.9 sigma against a 5 sigma floor | Matched the prediction: vibration's redline margin is under 5 sigma |
| Fault stays sub-threshold in at least 80% of runs | Passed on the first run | Matched |

### Learned
Passing tests aren't proof of a sound model. Yesterday's generator passed all 33 tests, yet every channel was a function of pump speed, so a bearing fault could only ever move one channel. The tests checked behavior, like "throttle up raises pressure", but nothing checked structure, like "does any channel read the bearing's state?"

### Ledger
- **Built by hand:** fault and split tests, `generator/splits.py`, `FaultLabel` contract tests
- **Delegated, can explain:** the `simulate()` rewrite (guided step by step), `make_fault()`, `FaultLabel`
- **Delegated, not yet understood:** none
- **Moved to "can explain":** `simulate()` (four-piece walkthrough, then rewrote it), `tests/test_generator.py`

### Moved to tomorrow
- Update the site's build log: moved to D4, Fri Nov 13, now that all AI-assisted code is understood

### Next
Fri 9:00: prime on precision-recall for rare events, then write the evaluation harness's exit tests before any baseline code.

## D2 · Wed Nov 11, 2026 · P0 nominal generator

**Exit test:** passed. Seeded runs reproduce exactly, different seeds differ, throttle up raises chamber pressure and pump speed, pump speed lags a step, coupled channels correlate above 0.9, and healthy runs stay under every redline: 33/33 tests.

### Built
- `Run` contract: throttle (6000,) and sensors (6000, 8) as separate arrays, shapes validated
- `generator/engine.py`: `make_throttle()` (startup ramp, 1 to 2 seeded steps, shutdown) and `simulate()` (3 dynamic states, 5 algebraic channels, plus or minus 3% parameter variation, offsets, noise)
- `REDLINES` set from physics before looking at any runs
- `tests/test_generator.py`: shape, reproducibility, step response, lag, correlation, redlines, throttle profile

### Decided
- Throttle stored separately from sensors: no column offset, `sensors[:, i]` is always `CHANNELS[i]`
- Separate random streams for throttle, engine and noise. Stream ids start at 1
- `noise=False` turns off only sensor noise. The engine stays the same
- Redlines are upper limits only. Low-side, phase-gated limits deferred
- Throttle timing rules enforced by import-time asserts
- Each test file tests its own module

### Expected vs actual
| Expected | Actual | Why |
| --- | --- | --- |
| Generator tests fail on `NotImplementedError` | Failed as expected, and the redline test failed on the empty dict | Matched. Without the key check, an empty `REDLINES` would have passed by checking nothing |
| `default_rng([seed, 0])` is a distinct stream | It equals `default_rng(seed)` | numpy pads short seeds with zeros, so streams start at 1 |
| Generator tests pass once `simulate()` exists | 33/33 passed | Matched |

### Learned
`default_rng([seed, 0])` is the same stream as `default_rng(seed)`, so stream ids start at 1, and each kind of randomness gets its own stream so they can't affect each other.

### Ledger
- **Built by hand:** `make_throttle()`, `REDLINES` values
- **Delegated, can explain:** `Run` contract
- **Delegated, not yet understood:** `simulate()`, `tests/test_generator.py`
- **Moved:** `contracts.py` and `tests/test_contracts.py`, from not yet understood to can explain

### Moved to tomorrow
- Update the site's build log (waiting until AI-assisted code is understood)

### Next
Thu 9:00: walk through `simulate()` until it can be explained, then start fault injection.

## D1 · Tue Nov 10, 2026 · P0 contracts

**Exit test:** passed. A dummy encoder emits (batch, 8, 16), `Verdict` round-trips through JSON, and architecture.md lists every frozen contract: 7/7 tests locally and in CI.

### Built
- Package skeleton: `generator/`, `model/`, `evaluation/`, `tests/`
- `reho` conda environment (Python 3.11), pinned `requirements.txt`, `pyproject.toml` with pytest config
- `contracts.py`: channels, command, run shape, window, embedding size, and a frozen, self-validating `Verdict`
- `tests/test_contracts.py`: 7 contract tests, written before the code
- README.md, docs/architecture.md, docs/two-week-plan.md
- Fifth floor commitment in design-doc.md
- GitHub Actions CI running pytest on every push
- `.gitignore` that keeps run configs and metrics, and ignores checkpoints

### Decided
- Five floor commitments: added "the verdict must be legible"
- Contract values from the How It Works book: d = 16, window = 100, 60 s at 10 ms, throttle only
- Training objective: forecasting, all 8 channels 10 to 20 samples ahead, checked against a persistence baseline
- Encoders share weights, with a channel-ID embedding and per-channel normalization from train-split stats
- Scope: bearing wear is the one required fault. A second fault is a stretch
- Calibration: temperature scaling, with an empirical quantile as fallback
- `evaluation/` instead of `eval/` (avoids shadowing a Python built-in)

### Expected vs actual
| Expected | Actual | Why |
| --- | --- | --- |
| Tests fail before `contracts.py` exists | Failed with `ModuleNotFoundError` | Matched |
| 6 of 7 pass once `contracts.py` exists | 6 passed, architecture test failed | Matched: doc was still empty |
| First push goes through | Rejected | GitHub had an older commit (`8908a85`) never pulled. Fixed with `git pull --rebase` |
| Tests run in the project environment | First green run was in `(base)` (Python 3.12) | New terminal starts in base. Re-ran in `reho` |
| CI passes first try | Green in 37 s | Matched |

### Learned
A contract test is only useful if it states the expected value independently of the code. `EXPECTED_CHANNELS` is typed out in the test instead of imported from `contracts.py`, because a test that reads its answer from the code under test agrees with any typo. Writing it first also meant the failures told me what to build next.

### Ledger
- **Built by hand:** package skeleton, `reho` environment, `requirements.txt`, README.md, fifth floor commitment, `runs/` rules in `.gitignore`
- **Delegated, can explain:** `pyproject.toml`, CI workflow, docs/architecture.md
- **Delegated, not yet understood:** `contracts.py`, `tests/test_contracts.py`

### Moved to tomorrow
- Update the site's build log from the 30-day plan to the two-week plan

### Next
Wed 9:00: update the site build log, then prime on first-order lag systems before writing the generator's equations.
