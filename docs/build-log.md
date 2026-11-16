# Build Log

One entry per session, newest first. Full decision entries live in [architecture.md](architecture.md).

## D5 · Mon Nov 16, 2026 · P1 and P2 model

**Exit test:** not run yet. `model/train.py` moved to D6, so the four slow exit tests (loss falls on nominal data, beats persistence per channel, fault residuals rise, end-to-end verdict) can't run until a model is trained.
- Fast tests pass: 18 data, 18 model (encoder shape and sharing, channel IDs, masking, throttle argument, noise floor loss, unscored start).
- Reproduce with `pytest -m "not slow" tests/test_data.py tests/test_model.py`.

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
| With channel masking, fault residuals rise above the healthy twin's in at least 80% of faulted runs (written before training) | | |
| Without masking, fault residuals rise in fewer than 80% of faulted runs, because a forecaster reading a channel's own history continues slow drifts (written before training) | | |

### Learned
A forecaster that reads a channel's own recent history simply continues a slow drift, so its error barely moves when the bearing wears. Hiding each channel's own input forces the forecast to come from the other channels, and that's where a fault shows: one channel inconsistent with the rest.

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
