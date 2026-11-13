# Build Log

One entry per session, newest first. Full decision entries live in [architecture.md](architecture.md).

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
