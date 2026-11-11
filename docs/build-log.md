# Build Log

One entry per session, newest first. Full decision entries live in [architecture.md](architecture.md).

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
- Scope: bearing wear is the one required fault; a second fault is a stretch
- Calibration: temperature scaling, with an empirical quantile as fallback
- `evaluation/` instead of `eval/` (avoids shadowing a Python built-in)

### Predicted vs actual
| Expected | Actual | Why the difference |
| --- | --- | --- |
| Tests fail before `contracts.py` exists | Failed with `ModuleNotFoundError` | Matched |
| 6 of 7 pass once `contracts.py` exists | 6 passed, architecture test failed | Matched: doc was still empty |
| First push goes through | Rejected | GitHub had an older commit (`8908a85`) never pulled; fixed with `git pull --rebase` |
| Tests run in the project environment | First green run was in `(base)` (Python 3.12) | New terminal starts in base; re-ran in `reho` |
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
