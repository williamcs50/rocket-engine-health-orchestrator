# Build Log

One entry per session, newest first. Full decision entries live in [architecture.md](architecture.md).

## D12 · Wed Nov 25, 2026 · Command cross-attention ablation

**Exit test (D12):** the ablation, pre-registered in the decision log (2026-11-25) and amended before the full run. The mid-run step test judges the plan's item: on val2's healthy runs, the command-aware model's median step ratio must be at least 10% lower than the ablation's on every matched seed. The primary test is reported as registered: the pooled transient share at least 20 points lower and lower on every seed, with mean recall no more than 3 points lower.

The ablation ran on commit `d55cc0c` with a clean tree, and the command-aware std thresholds matched D8's. **Both tests fail, so the plan's item reads "tested, not met".**

| Test | Bar | Result |
| --- | --- | --- |
| Mid-run steps, judges the item | Command-aware median at least 10% lower on every seed | 0.232 against 0.241 (3.7% lower), 0.248 against 0.268 (7.5%), 0.215 against 0.257 (16.3%): fails |
| Primary, transient share | At least 20 points lower pooled, and lower on every seed | 78.1% against 86.2%. Seeds 90.0% against 90.0%, 66.7% against 90.0%, 80.0% against 77.8%: fails |
| Primary, recall | At most 3 points lower | 57.7% against 55.4%: holds |

- 757 mid-run steps per model, on val2's 500 healthy runs. False alarms 2.0%, 2.4% and 2.0% with commands, 2.0%, 2.0% and 1.8% without.
- Seen alongside, not part of the test: the raw median step peaks are 14 to 23% lower with commands (1.80, 1.93 and 1.75 against 2.19, 2.24 and 2.27). The command-aware thresholds are lower too (7.76 to 8.14 against 8.36 to 9.07), so dividing by them removes most of the gap.
- Every number and per-run row is in the committed `runs/ablation/metrics.json`.

### Demo (evening)
The plan's last open item, "Demo replays real exported runs", is met: `docs/demo/index.html` replays four real runs.

| Replayed run (val2, model seed 0) | Seed | What happens |
| --- | --- | --- |
| Healthy run, throttle steps | 500000 | No alarm. Peak confidence 0.31 |
| Bearing wear, caught | 500001 | Onset 11.9 s, alarm 23.1 s with every channel under its redline, the highest at 84% |
| Bearing wear, flagged too late | 500009 | Onset 12.1 s, alarm 30.1 s, outside the 15 s window, so a miss |
| Healthy run, false alarm | 500014 | Alarm at 11.3 s with no fault, confidence 0.98 |

- `evaluation/demo.py`: the four runs by fixed rules, 10 Hz replay frames with each channel's share of its redline and its forecast miss against healthy, the score, the latched alarm and the running confidence, checked against `to_verdict`. Also the static figure `docs/demo.png`
- `docs/demo/index.html`: the site's page, with the scenarios, the playback loop, the verdict rule and the tour narration replaced. The ring, hub, 3D engine and tour mechanics are unchanged. Rendered in headless Chrome at the caught run's alarm: verdict "Coupled anomaly", confidence 99%, chips bearing temp, nozzle temp and coolant flow, and those parts lit on the engine
- The chips at the alarm include one true fault channel of three, the logged ranking limitation, shown rather than hidden
- Tests before code for the pure parts. 351 fast tests pass

### Built
- `Forecaster(command_input=False)`: an all-zero throttle history in `forward`, the one place training, scoring and attention pass through. Same seed, same weights
- `--no-command`, `_nocmd` run names, and `load_run` building whichever model the config says. D7's configs, which predate the key, still load with commands
- Three ablation models, 70 to 72 s each. End probe losses 8.27, 8.80 and 8.28, against 6.98, 7.30 and 6.05 for D7, and close to D6's 8.41, 9.02 and 7.93
- `evaluation/ablation.py`: both tests, provenance with the ablation's weight hashes, results in `runs/ablation/metrics.json`. Step windows stop where the next change starts, after a test caught a late step reaching into shutdown on seed 17
- The ablation weights committed under the same rule as D7's
- Tests before code. 344 fast tests pass
- The full run took about 26 minutes for val2, after the reference fits and calibrate

### Decided
- D6 isn't a clean ablation: a different fusion layer, the mean score, and weights not committed. So the ablation is retrained with D7's architecture and a zeroed throttle
- The primary measure is the transient share, since at a fixed budget the threshold pins the total false alarm rate
- Amended before the full run: the mid-run step test judges the plan's item, because the primary's transient window includes startup, which both models face the same way
- No change after the second 20 run check, which pointed against the step test. The full run then pointed the other way
- The plan's item reads "tested, not met". Nothing is retuned, v1 is unchanged and test3 stays untouched

### Expected vs actual
Written before any ablation model is trained.

| Expected | Actual | Why |
| --- | --- | --- |
| Pooled transient share, command-aware models: about 80%. Ablation: about 92%. About 65% confident | 78.1% against 86.2% | Close. The ablation is lower than expected because startup alarms are transient for both models, which pulls the shares together |
| Lower on every matched seed: no. About 55% confident | No. Seed 0 ties at 90.0%, and seed 2 is higher, 80.0% against 77.8% | Right. With about 10 false alarm runs per seed, one run moves a share by 10 points |
| Mean recall at 2%, command-aware: about 58%. Ablation: about 55% | 57.7% against 55.4% | Right. The command-aware recall matches val and test2 |
| Overall, the hypothesis holds: no. About 70% confident | Does not hold | Right. An 8 point drop, against a 20 point bar |

**Amended before the full run (decision log, 2026-11-25):** the mid-run step test judges the plan's item. On every matched seed, the command-aware model's median step ratio (peak std score in the 5 s after a mid-run step, over its 2% threshold) must be at least 10% lower than the ablation's. Written after the `--limit 20` wiring check, before any step measure exists.

| Expected | Actual | Why |
| --- | --- | --- |
| Median step ratio, command-aware: about 0.45. Ablation: about 0.48. About 55% confident | 0.215 to 0.248 against 0.241 to 0.268 | The direction right, the values about half. A typical step peaks near 2 on the std score, against thresholds near 8 |
| The mid-run step test holds: no. About 60% confident | Does not hold. Lower on every seed, but by 3.7%, 7.5% and 16.3% | Right. Commands calm the score at steps, but by less than the bar on two seeds. Most of the raw gap is canceled by the command-aware model's lower thresholds |

### Learned
- A small, consistent effect can fail a fair bar. The command-aware model is lower on every seed, but 10% was the margin set beforehand, and it holds on one seed
- A 20 run check can point the wrong way. It showed the command-aware ratios higher on all three seeds. The full run showed them lower on all three. Not changing the test after it is what kept the result clean
- At a fixed budget, a rate the threshold pins can't show an effect, and a ratio to each model's own threshold can hide one. Raw step peaks differ by 14 to 23%, but the better model also has lower thresholds
- A transient window that includes startup can't isolate commanded steps. The pre-registration should have caught it. The amendment came after a 20 run check, so it's weaker than one written first

### Ledger
- **Built by hand:** none
- **Delegated, can explain:** `command_input` in `Forecaster`, the `load_run` and `run_name` changes, `evaluation/ablation.py` (`step_windows`, `step_ratios`, `step_verdict`, `ablation_verdict`) and `tests/test_ablation.py`, after a five question teach-back without notes. One answer needed a completion: why the seed 17 window reaching into shutdown would have biased the step ratios
- **Delegated, can explain (evening):** the demo, `evaluation/demo.py` (`smooth`, `causal_ranking`, `to_frames`, the run selection) and the replaced parts of `docs/demo/index.html`, after a three question teach-back without notes. It led to one fix: the chips are now labeled as ranked up to the alarm
- **Delegated, not yet understood:** none
- **Moved:** every item above to can explain

### Next
1. Later, pre-registered first: whether command input lowers raw step peaks on test3, the effect seen alongside today's test

## D11 · Tue Nov 24, 2026 · v1.1, false alarms near the cap

**Exit test (D11):** the diagnostic, pre-registered in the decision log (2026-11-24). It holds if, on val2's healthy runs, offset and swing have a Spearman correlation of at least 0.3 on every seed, and the engines that false-alarm on any seed have a higher median offset than the engines that never alarm.

The diagnostic ran as pre-registered, on commit `27900a8` with a clean tree and the std thresholds matching D8's. The hypothesis does not hold, on the first condition.

| Condition | Bar | Result |
| --- | --- | --- |
| Spearman, offset against swing | At least 0.3 on every seed | 0.062, 0.064 and 0.057: fails |
| Median offset, false alarm engines against the rest | Higher | 1.873 against 1.720: holds |

- The std score's false alarms on val2 were 2.0%, 2.4% and 2.0%. 23 of 500 healthy engines false-alarm on at least one seed.
- Every number and per-engine row is in the committed `runs/diagnose_offsets/metrics.json`.

### Built
- Val2 (seeds 500,000 to 500,999) for development and test3 (600,000 to 600,999), locked like test and reserved
- `calibration_offsets(seed)` in `generator/engine.py`: each channel's offset in noise sigmas, recovered by replaying the engine random stream. Checked against `simulate` itself, where the channels that start at 0 read exactly their offset
- `evaluation/diagnose.py`: swing, RMS offset, Spearman from ranks with ties averaged, engines counted once, and the verdict. Tests before code
- The CI actions upgraded to checkout v5 and setup-python v6, which run on Node 24
- 298 fast tests pass

### Decided

### Expected vs actual
Written before the diagnostic runs, and before any val2 data exists.

| Expected | Actual | Why |
| --- | --- | --- |
| Spearman between offset and swing, per seed: about 0.25, 0.30 and 0.25. At least 0.3 on every seed: no. About 55% confident | 0.062, 0.064 and 0.057 | The verdict right, the values far lower. Offsets barely explain the swing, so something else drives it |
| Median offset of false alarm engines against never-alarm engines: about 1.85 against 1.7 noise sigmas. Higher: yes. About 60% confident | 1.873 against 1.720 | The false alarm engines lean slightly toward larger offsets, as expected for a partial cause. With 23 engines, the gap is small and untested for size |
| Overall, the hypothesis holds (both conditions): no. About 60% confident | Does not hold | It failed on the first condition alone. Offsets barely relate to swing, so a true second condition couldn't carry it |
| Healthy val2 engines that false-alarm on at least one seed: about 30 of 500 | 23 | Close. Slightly fewer, with val2's false alarm rates at the 2% target |

### Learned
- A mechanism that's right on paper can be negligible in practice. The `2·offset·noise` term exists, but it's swamped by whatever else drives the swing
- Checking the cause before building the fix saved building one that wouldn't have helped

### Ledger
- **Built by hand:** none
- **Delegated, can explain:** `calibration_offsets` and `evaluation/diagnose.py`, after a short teach-back
- **Delegated, not yet understood:** none
- **Moved:** `calibration_offsets` and `evaluation/diagnose.py` to can explain

### Next
N/A. This was the project's last day

## D10 · Mon Nov 23, 2026 · P6 demo and ship

**Exit test (D10):** a stranger can clone, run one command, and reproduce the val numbers. The not-yet-understood column is empty.

- Not yet understood: met. Every item from D1 to D9 moved to "can explain" through nine teach-backs, answered without notes, one per pipeline stage. One question needed a second try (the calibration offsets and the lagged difference), and the second answer was right. Today's test2 and verdict code moved after a short teach-back of its own
- Reproduce the val numbers: met. The clone check was successful. A fresh clone from GitHub, with the committed weights, passes all 271 fast tests, and `python -m evaluation.gate1` rewrites `runs/gate1_d8/metrics.json` byte for byte identical to the committed file

### Built
- Test2, pre-registered (`a512c38`) before the split existed. Seeds 400,000 to 400,999, locked like test, since `LOCKED_SPLITS` now holds both and each has its own lock. `final.py --split test2` judges the std line with the same D8 threshold check, then claims, then opens, and prints and saves the Gate 1 verdict. Tests before code. A val dry run of the test2 path saved nothing
- The test2 run: the thresholds matched D8's, and the test2 pass took about 26 minutes, from the `started` marker at 3:23 PM to results at 3:49 PM. The std score passes Gate 1. Every number is in the committed `runs/test2/metrics.json`
- The verdict takes the std score. `to_verdict` has a `score` argument, and `StandardizedScore.verdict_score` gives the same per-sample score test2 used. `model/` still imports nothing from `evaluation/`. Tests before code
- The fresh-clone check. A clone from GitHub installs from `requirements.txt` on Python 3.11 and passes all 248 fast tests. The baselines reproduce exactly. Retraining doesn't: it starts from the same loss but the weights drift, so the val numbers come out within about a point. So the three D7 `model.pt` files (58 KB each) are now committed. A second clone, after the weights were committed, was successful: all 271 fast tests passed and gate1 reproduced every val number exactly
- Docs: architecture.md's reference sections match the final code. Decision rows for the verdict ranking limitation, test2's pre-registration and result, and the committed weights. The README for v1, with status, a quickstart without training, and results on val, test and test2. The data.py comment corrected. Commit hashes in backticks, and American spelling
- 271 fast tests pass

### Decided
- The design doc stays as the pre-build rationale, with no Results section, a small deviation from the plan's D10 row. Results live in the README and the decision log

### Expected vs actual
Test2 predictions, written before the run. Checking the arithmetic before the run showed that my per-seed false alarm rates give about a 30% chance that all three seeds stay at or under 3%, so I raised my confidence in a fail from 55% to 70%. The caught prediction was added at the same check, since caught now follows the std line.

| Expected | Actual | Why |
| --- | --- | --- |
| Std line (judged): mean recall about 58%. False alarms per seed about 2.8%, 2.7% and 2.9%, the averages of val and test. Gate 1 fails, because each seed's expected rate sits just under the 3% cap and sampling noise is about 0.7 points per seed, so all three staying at or under 15 of 500 is less likely than not. About 70% confident | Gate 1 passes. Mean recall 58.0%, seeds 52.4%, 64.8% and 56.8%. False alarms 1.8%, 2.6% and 1.4% | Recall right, the verdict wrong. Seeds 0 and 2 came in well under the val and test averages. Each seed's rate swings by about 0.7 points from sampling, and this split drew low ones. One pass doesn't show the true rates are lower |
| Mean line, for information: mean recall about 8%, highest false alarm about 3.3% | Mean recall 8.1%, highest false alarm 3.2% | Right. The same model, so close to val and test |
| Fair z score about 37% | 35.0% | Close. The same as test's 35.2% |
| Calibration exit test on the std line passes. About 60% confident | Passes on every seed and both lines | Right. The q = 0.9 cells, which failed on val, came in at 9.2%, 10.0% and 9.4% |
| All faulted runs: residual overlap about 60%, attention overlap about 41%. Healthy control about 42% | Residual 62.2%, attention 42.2%. Healthy control 42.9% for residuals and 43.9% for attention | Right. The same pattern as test: residuals follow the fault and attention matches its control |
| Caught runs, now following the std line: residual overlap close to the all faulted number, about 60% | 65.3%, on 262, 324 and 284 runs | A few points higher. Caught runs tend to be the larger faults, so the rise is clearer, and with the std line there are enough of them to show it |

### Learned
- Holding the rule is what made the pass count. The std score missed val by one run, cleared test for information only, then passed test2. Adopting it on Friday would have looked the same on paper, but every number after that would have carried an asterisk
- Pre-registration is what makes a second held-out run legitimate. Writing the method, bar, seeds and no-retry rule before the split existed turned "run it again until it passes" into one clean test
- A fixed seed isn't reproducibility. The same seed gave the same starting loss but different trained weights, from floating point order in multithreaded math. Reproducing the numbers needed the weights themselves committed
- Explaining the code finds what tests don't. The teach-backs turned up a comment that stated the opposite of what the code does, and a verdict ranking different from the one that was evaluated. Every test passed on both

### Ledger
- **Built by hand:** none
- **Delegated, can explain:** today's code, after its teach-back: test2's split lock (`LOCKED_SPLITS`), the `--split` path in `final.py`, `gate1_verdict`, the verdict's `score` argument and `verdict_score`
- **Delegated, not yet understood:** none
- **Moved:** the whole backlog from D1 to D9 to "can explain", through the nine teach-backs. Contracts and the generator, data, the model, scoring, training and saving, the harness and gate1, the fair z score and the std score, calibration, and attribution with the test run

### Next
After v1:
1. Rank the verdict's channels by rise relative to the first alarm, the logged limitation, and evaluate it on a fresh split pre-registered first
2. The demo with real traces: replace the scripted scenarios with exported runs and update the tour narration. D10's morning task, not started
3. The std score's false alarms near the 3% cap: passed on test2 but crossed once in three splits. Test the hypothesis that noise scaling with a run's offset widens its swing
4. A second fault type, so attribution can show it tells faults apart
5. Later: the GRU experiment from the design doc, deterministic training for future models, and moving the calibrate pass onto `detector_scores`

## D9 · Fri Nov 20, 2026 · Attribution and the one test run

**Exit test (D9):** met. The final test table is complete, from one test run on commit `47cce60`, with a clean working tree and the model hashes recorded. Against the Gate 1 bar, which still applies, the judged line fails by one healthy run. Reported as the clean result: test was opened once, with every decision committed beforehand.

| At the 2% budget on test | Seed 0 | Seed 1 | Seed 2 | Mean |
| --- | --- | --- | --- | --- |
| Orchestrator, D7 mean (judged) recall | 17.6% | 1.6% | 2.8% | 7.3% ± 8.9% |
| Judged false alarms | 2.6% | 3.2%, fails | 2.2% | |
| Orchestrator, std (failed Gate 1 on val by one false alarm, not adopted, shown for information) recall | 52.0% | 64.0% | 58.6% | 58.2% ± 6.0% |
| Std false alarms | 2.4% | 2.8% | 3.0% | |

- Baselines at 2%: fair z score 35.2% at 2.8% false alarms, D4 z score 0.2% at 2.4%, redline 0.2% at 0.0%.
- Every false alarm on the judged line is transient, as on val. Mean delay is 8.6 to 14.1 s on the judged line, and 11.6 to 11.7 s on the std line.
- The calibration exit test passes on every seed and both lines. The mean line's worst cell is seed 1 at q = 0.8, 21.6% against 20%.
- Attribution, top 3 overlap against 37.5% chance:

| Population | Residual | Attention | Agreement |
| --- | --- | --- | --- |
| All faulted runs | 61.2% ± 2.9% | 40.9% ± 12.4% | 41.3% ± 5.4% |
| Caught runs (n 88, 8, 14) | 56.5% ± 10.4% | 56.1% ± 14.4% | 43.7% ± 7.6% |
| Healthy control | 41.9% ± 1.0% | 43.1% ± 12.1% | 44.6% ± 4.9% |

- Every number, including the 3,000 per-run attribution rows, is in the committed `runs/test/metrics.json`. The run can't be repeated, because the `started` marker refuses a second one.

### Built
- The test split guard. `make_run` refuses test with `PermissionError` outside `open_test_split()`, a context manager that relocks even after an error. `claim_test_run` writes a `started` marker before any test run is simulated, and refuses if the marker or results already exist
- `attention_by_channel` in `model/forecaster.py`: the self-attention each channel receives, from one unmasked pass. Hooks switch the weights on and are always removed afterward
- `evaluation/attribution.py`: the rise, top 3, overlap, a 95% bootstrap interval and the healthy control's borrowed onset
- gate1.py: `fit_detectors` and `detector_scores` shared with the test script. `--limit 20` output is identical before and after
- `evaluation/final.py`: the D8 threshold check, provenance (commit, clean tree, model hashes), claim and then open test, one pass with attribution per run and model, and a val dry run that saves nothing
- Tests before code for the guards, the attribution and most of the script. The three tests on `main` were written after its code. 248 fast tests pass
- The one test run, about 18 minutes for the test pass after fitting and calibrate

### Decided
- The verdict's ranked channels follow the residuals, not attention. Attention is reported separately. This is a logged deviation from the design doc and the napkin
- Confidence is per run, from the peak, so the flag and confidence agree. Conditioning on the command comes only from the forecaster's throttle input, a stated limitation
- The second fault type is cut, under cut list item 2. Bearing wear is the only fault
- Attribution, set before test: top 3 overlap on residual and attention rise in the catch window, all faulted runs plus the caught runs, a healthy control, and uncertainty per seed and across seeds. No pass bar
- The test result is held under the adoption rule. The std line clears the bar on test but isn't adopted
- The std score is pre-registered for a fresh split, `test2`, after v1, with the same bar and the split opened once

### Expected vs actual
Written before the test run, from drafted estimates I adjusted. The healthy control prediction is influenced by a glimpse of the val dry run (20 seeds per split), not by test.

| Expected | Actual | Why |
| --- | --- | --- |
| Judged mean line at 2% on test: mean recall about 9%, seeds about 15%, 2% and 6%. False alarms about 3%, 4% and 2%. Gate 1 fails on false alarms. About 75% confident | Mean recall 7.3%, seeds 17.6%, 1.6% and 2.8%. False alarms 2.6%, 3.2% and 2.2%. Gate 1 fails on false alarms, seed 1 by one run | The same model and thresholds as val, so close to val's numbers. Only one seed is over this time. My 9% mean didn't match my own seed guesses, which average 7.7% |
| Std line, for information: mean recall about 55%, false alarms 2.5 to 3.5%. Whether it clears the bar is a coin flip, about 50% | Recall 58.2% plus or minus 6.0%, false alarms 2.4%, 2.8% and 3.0%. It clears the bar | Nearly the same as val (58.1%) on an independent split. The coin flip landed on passes, with seed 2 exactly at the limit |
| Baselines at 2%: fair z score about 40%, D4 z score about 1%, redline 0%. About 80% confident | Fair z score 35.2%, D4 z score 0.2%, redline 0.2% | Close. The fair z score is a few points below val's 39.6%, and the redline caught one run |
| Calibration exit test on the mean line fails, at q = 0.8 or 0.95 on at least one seed. About 75% confident | Passes on every seed and both lines | Wrong. Calibrate transfers to test within 3 points, so val's failure was at least partly that split's sample. One split isn't enough to call a gap systematic |
| All faulted runs: residual overlap about 55%, attention overlap about 40%, against 37.5% chance. About 60% confident | Residual 61.2%, attention 40.9% | Attention right, residual 6 points higher. Residual attribution clearly follows the fault, and attention matches its healthy control |
| Caught runs: residual overlap about 70%. About 55% confident | Residual 56.5%, seeds 66.7%, 45.8% and 57.1% on 88, 8 and 14 runs | Wrong. Seed 0, with 88 runs, is near my guess. The other two have too few runs for their means to say much |
| Healthy control: both overlaps near chance, 35 to 45%. About 65% confident | Residual 41.9%, attention 43.1% | Right. Both are a little above chance with no fault, a mild preference for the wear channels. On seed 1, attention reads 57% on healthy runs |

### Learned
- Holding the rule cost the pass. The std line cleared the bar on test but wasn't adopted. The same result on two independent splits, 58.1% and 58.2%, is stronger evidence than one pass would have been
- The healthy control is what made attribution readable. Seed 1's attention looked well above chance on faulted runs (55.2%) until the control showed 57.0% with no fault at all
- In a masked forecaster, attention shows routing for prediction, not anomaly. Residuals beat their control by 19 points, and attention doesn't beat its control at all
- A calibration failure on one split can be that split's sample. Val failed and test passed with the same reference
- Checking a mean against its parts takes seconds and should happen before a prediction is committed

### Ledger
- **Built by hand:** none
- **Delegated, can explain:** none yet. I walked through every file today, but the questions are still to come
- **Delegated, not yet understood:** the split guard and `claim_test_run`, `attention_by_channel`, `evaluation/attribution.py`, the gate1 refactor, `evaluation/final.py`, and today's tests in `tests/test_splits.py`, `tests/test_attribution.py` and `tests/test_final.py`. D8's items are still here too
- **Moved:** none

### Moved to tomorrow
- The understanding questions for D8 and D9, so items can move to "can explain"
- D5's weekly review blanks and the site build log

### Next
Sat Nov 21, review, 9:00 AM to 12:30 PM, in order:
1. Understanding questions on D8 and D9, moving items to "can explain"
2. D5's review blanks and the site build log
3. Prep for D10: the README results table from `runs/test/metrics.json`, and which real runs to export for the demo

## D8 · Thu Nov 19, 2026 · P4 calibration + P5 experiment

**Exit test (calibration):** failed, judged per seed on the mean line, since the new score wasn't adopted. The share of healthy val runs with confidence at least q must be within 3 points of 1 minus q, for q = 0.8, 0.9 and 0.95. The P5 experiment (GRU against cross-attention) is dropped, per the Gate 1 bar.

| Line, seed | q = 0.8 (target 20%) | q = 0.9 (target 10%) | q = 0.95 (target 5%) |
| --- | --- | --- | --- |
| mean, seed 0 | 22.4% | 12.6% | 8.6%, fails |
| mean, seed 1 | 23.8%, fails | 10.4% | 6.0% |
| mean, seed 2 | 20.8% | 9.0% | 3.8% |
| std, seed 0 (shown for information) | 23.4%, fails | 14.4%, fails | 8.0% |
| std, seed 1 (shown for information) | 22.8% | 14.0%, fails | 6.8% |
| std, seed 2 (shown for information) | 20.6% | 9.0% | 6.2% |

**Gate 1 rerun with the new score:** failed by one healthy run, so the new score is not adopted, and the Nov 20 test runs with D7's mean scoring. Same bar, unchanged. Reported as optimistic, since this is val's third round of decisions.

| At the 2% budget | Seed 0 | Seed 1 | Seed 2 | Bar |
| --- | --- | --- | --- | --- |
| New score (std) recall | 51.4% | 64.0% | 58.8% | mean 58.1% ± 6.3%, at least 5%: passes |
| New score false alarms | 3.2% | 2.6% | 2.8% | each at most 3%: fails on seed 0, 16 of 500 runs against 15 allowed |
| New score transient false alarms | 3.0% | 1.8% | 2.2% | |
| D7 mean recall (for comparison) | 17.6% | 1.4% | 5.2% | mean 8.1% |

- Baselines at the 2% budget: redline 0.0%, D4 z score 1.2% at 0.6% false alarms, fair z score 39.6% at 2.4% false alarms.
- The new score's thresholds are steady across seeds, 7.76 to 8.14 at 2%, against 145 to 425 for D7's mean line.
- Reproduce with `python -m evaluation.gate1` (about 35 minutes, including the reference fits). Every number is in the committed `runs/gate1_d8/metrics.json`.

### Plan and cutoffs, set at 11:47 AM
- Order: the fair z score fix first, then calibration, then the new score. The z score fix is independent of the model and the smallest piece, and the final table compares against it. Calibration is today's exit test, and its code works on any score. The new score is the largest and least certain, so it goes last, where running long can't cost tomorrow's test run anything
- If the new score is adopted, calibration is rerun on it before tomorrow. That only means scoring calibrate's runs again
- Adoption rule: the new score is adopted only if it passes Gate 1 on val, all three conditions. Otherwise tomorrow's test runs with D7's scoring (mean aggregate). If both fail Gate 1, the test still runs with D7's scoring and is reported as having failed Gate 1 on val. All 3 seeds go to test, reported per seed plus mean and spread. No seed is chosen
- Calibration exit test: held out on val's 500 healthy runs today, and test's tomorrow. Counted per run, using each run's peak confidence, which matches how Gate 1 counts false alarms
- The z score fix runs from 1:00 to 2:30 PM, after lunch. It also builds the reference logic the new score reuses
- Calibration starts by 2:30 PM. If temperature scaling hasn't met the ±3 point check by 4:00 PM, use the plan's fallback, the empirical quantile
- The new score starts only after calibration and the z score fix. It must be built and tested by about 5:00 PM, so the 25 minute Gate 1 rerun finishes by 5:30 PM. If it isn't evaluated by 5:30 PM, it's dropped, and tomorrow's test runs with D7's scoring
- Revised before lunch: the first version said calibration by 1:30 PM and the fallback at 3:00 PM, which left the z score fix only 30 minutes
- Revised in the afternoon: the z score fix runs over, so calibration starts as soon as the conditioned z score is built, the fallback stays at 4:00 PM, and the new score keeps its 5:30 PM cutoff, so it will almost certainly be dropped
- Revised again in the evening: calibration finished early, so the new score's cutoff moved from 5:30 PM to 8:30 PM, following my tutor's suggestion. This line was logged after the new score's code was written, not before

### Diagnostic (predictions written on D7, run this morning)
`python -m evaluation.checks slow_state`, model `h10_mask0.25_cmd_seed0`. Its count of seed 0's false alarm runs matches Gate 1's (17), so the script scores runs the same way Gate 1 did. It ran first in the base environment, then again in `reho` with identical output.

Nozzle temp peak residual, ratio unmasked ÷ masked (smaller means unmasking helped more), first 10 healthy calibrate runs:

| Seed | Startup ratio | Steady-state ratio | Both conditions |
| --- | --- | --- | --- |
| 200000 | 0.24 | 0.29 | yes |
| 200002 | 0.05 | 0.06 | yes |
| 200004 | 0.08 | 0.13 | yes |
| 200006 | 0.28 | 0.22 | no |
| 200008 | 0.13 | 0.15 | yes |
| 200010 | 0.13 | 0.13 | no |
| 200012 | 0.22 | 0.19 | no |
| 200014 | 0.18 | 0.19 | yes |
| 200016 | 0.07 | 0.15 | yes |
| 200018 | 0.23 | 0.75 | yes |

Seed 0's first false alarm times on healthy val runs, at its 2% mean threshold of 145.021: 1.1, 1.1, 1.1, 1.1, 1.1, 2.6, 3.1, 3.2, 3.2, 4.5, 4.6, 4.9, 5.0, 5.5, 13.2, 28.7 and 47.9 s.

### Built
- The fair z score, done at 3:42 PM (commit `e353e7a`). `evaluation/profile.py` bins every sample by the throttle profile and fits a healthy reference with merging, fallback counts and a std floor. `ConditionedZScore` in `evaluation/baselines.py` standardizes each channel against it and combines channels as D4 does. It's wired into gate1.py as `zscore_fair`, next to D4's unchanged `zscore`. The full run fits both on the 2,000 train runs, simulating them twice, about a minute or two extra
- Calibration (`evaluation/calibration.py`): a run's confidence is the share of calibrate's healthy peaks strictly below its own peak, one reference per detector. A NaN peak is refused, and a share exactly on the tolerance edge passes. `to_verdict` takes the confidence as a function, so `model/` imports nothing from `evaluation/`. The exit test prints in gate1.py
- The new score (`evaluation/score.py`): `mean_residual` keeps the blind rows NaN, `lagged_difference` subtracts the mean over [t − 2000, t − 1500), and `StandardizedScore` fits a level and a difference reference per model on train seeds 500 to 999, scoring each run once. Blind rows are refused if filled in, and become −∞ last. `ProfileReference` refuses a quantity with no spread
- gate1.py: baselines as a dict, the fair z score, a `/std` line per seed, the adoption verdict, the calibration exit test on both lines, results in `runs/gate1_d8/`
- 209 fast tests pass

### Decided
- Next scoring fix: a standardized, differenced residual, fully specified in the decision log before any code. Judged by the Gate 1 bar, unchanged, with no separate exit test
- Calibration by percentile rank, chosen on purpose over temperature scaling, which can't go below 0.5 without an offset and answers "probably faulted"
- Shutdown progress bins considered and dropped: shutdown is 4.1% of samples but 0.2% of the tail
- The new score fails Gate 1 by one run and is not adopted. The rule holds, and nothing on val is retuned
- On the Nov 20 test run, the std line is reported as a secondary line, labeled as failed on val and not adopted, shown for information. Decided before test exists

### Expected vs actual
| Expected | Actual | Why |
| --- | --- | --- |
| Diagnostic 1 (from D7): in at least 6 of 10 runs, nozzle temp's unmasked startup peak is at most half its masked startup peak, and unmasking shrinks the startup peak by a larger fraction than the steady-state peak. About 50% confident | Met as written, 7 of 10, but the margins on the control are negligible. In most runs the two ratios are nearly equal (0.05 against 0.06, 0.13 against 0.15, 0.18 against 0.19). Only 200018 (0.23 against 0.75) shows a clear difference | My own refutation clause said that if the steady-state peak shrinks just as much, the effect is general, and that's essentially what happened. Unmasking cuts nozzle temp's residual by 70 to 95% everywhere. The evidence points to a general masked nozzle problem, such as a per-run calibration offset invisible while the channel is hidden, not a startup-specific one. The masked startup peak is still larger than the masked steady peak in all 10 runs |
| Diagnostic 2 (from D7): at least 12 of seed 0's 17 false alarm runs first alarm before 10 s. About 70% confident | Supported: 14 of 17 first alarm before 10 s. 5 of them alarm at 1.1 s, the first sample the model can score, 9 more between 2.6 and 5.5 s, and 3 mid-run | The false alarms are a startup problem, and almost a third fire at the very first forecast. This read val again, but only described a result already in hand |
| With the standardized differenced score, each seed's val false alarm rate falls to at most 3%, so Gate 1 passes. Mean recall stays at least 8.1%, D7's value. Seed 1's false alarm rate, 4.2% on D7, drops the most of the three. About 40% confident, because my last two predictions about false alarms both missed and this score is untested (written before any code) | Two of three parts right. Gate 1 failed by one run: seed 0's false alarm rate is 3.2%, against at most 3%. Mean recall is 58.1%, far above 8.1%. Seed 1's false alarm rate dropped the most, 4.2% to 2.6%, while seed 0 went 3.4% to 3.2% and seed 2 rose 1.8% to 2.8% | The standardization did what it was meant to: steady thresholds across seeds and a sevenfold rise in recall. The false alarm transfer from calibrate to val still runs slightly high, now on seed 0 alone |
| Calibration exit test, judged per seed. It fails, because seed 0 is already at 8.6% at q = 0.95 against 5 plus or minus 3. I think q = 0.8 and q = 0.9 also come out above 1 minus q on at least one seed each, since it's the same calibrate-to-val transfer problem as Gate 1. About 70% confident on 0.8 and 0.9 (written before any calibration code, knowing the q = 0.95 numbers from D7's Gate 1 run) | Matched. It fails, on seed 0 at q = 0.95 (8.6%) and seed 1 at q = 0.8 (23.8%). q = 0.8 is above 20% on all three seeds, and q = 0.9 above 10% on seeds 0 and 1 | The same calibrate-to-val transfer problem as Gate 1, at other levels. Seed 2 passes all three |

### Learned
- A control compared with a strict "<" needs a margin, or noise decides the result. Next time, write how much lower counts, for example "at least 0.05 lower"
- Masked nozzle temp is badly predicted at all times (steady-state peaks of 31 to 260, against 13 to 23 unmasked), and worst at startup. That fits a per-run offset the other channels can't reveal
- The fair baseline changes the story. Conditioned only on the throttle profile, with no learning, the z score catches 39.6% at 2.4% false alarms, against D4's 1.2%. D7's model (8.1%) loses to it badly. The std score (58.1%) beats it, which hints the model adds value once scored properly, but that's a hint, not a claim, since the std score wasn't adopted and the false alarm rates differ
- A conditioned baseline with no learning went from 1.2% to 39.6%, and the same D7 models went from 8.1% to 58.1% with only a scoring change. Most of what held the model back was scoring (startup and per-run offsets), not architecture
- A 20 run quick check can't judge calibration: with 10 healthy runs a share moves in steps of 10%, so q = 0.95 can only read 0% or 10%
- Holding a pre-registered rule is easiest to justify and hardest to do when a result misses by one run

### Ledger
- **Built by hand:** none. `profile.py` was typed from code I was given, which counts as delegated until I can explain it
- **Delegated, can explain:** none
- **Delegated, not yet understood:** `slow_state` in `evaluation/checks.py`, `evaluation/profile.py`, `ConditionedZScore`, `evaluation/calibration.py`, `evaluation/score.py`, the `to_verdict` change, the gate1.py wiring, and today's tests: `tests/test_profile.py`, `tests/test_score.py`, `tests/test_calibration.py`, the conditioned z score tests in `tests/test_baselines.py` and the verdict confidence test in `tests/test_model.py`
- **Moved:** none

### Moved to tomorrow
- Walk through everything in "not yet understood", starting with `profile.py`. The backlog grew today, so Saturday's review is mostly walkthroughs
- D5's weekly review blanks and the site build log, both still open

### Next
Fri Nov 20, D9, in order:
1. A 15 minute reread of tonight's diffs, fresh
2. The test run script, with both guards: the plan's flag in the split loader, and refusing to run if test results already exist. Every detector: D7 mean (primary, judged), std (secondary, labeled), redline, D4 z score and the fair z score
3. Decide what attribution means before opening test: residual top k against the ground truth channels, and how attention weights are read from the decoder, or state that attention attribution is dropped
4. The one test run

## D7 · Wed Nov 18, 2026 · P3 command awareness

**Exit test (D7):** failed on both conditions. Set before the rerun: at the 2% budget on val, over 3 seeds, the mean transient false alarm rate is at most 1.6% (half of D6's 3.2%), and the mean recall is at least 9.4% (D6's lowest seed). Judged on the mean aggregate only, the one chosen on D6, decided before the run. The max line is printed but doesn't count.
- Mean transient false alarm rate 3.1%, against at most 1.6%: fails
- Mean recall 8.1%, against at least 9.4%: fails

**Gate 1 rerun:** failed again, on the same condition as D6. Same bar, unchanged: at the 2% budget on val, the 3 masked seeds' mean recall is at least 5%, every seed's recall is above 1.2%, and every seed's false alarm rate is at most 3%. Reported as optimistic, since it reuses val.
- Mean recall 8.1%: passes
- Lowest seed 1.4%: passes, by one catch over the z score's 6
- False alarms 3.4%, 4.2%, 1.8%: two seeds over 3%, fails

| At the 2% budget, mean aggregate | Before (D6) | After (D7, command-aware) |
| --- | --- | --- |
| Recall, seeds 0, 1, 2 | 10.0%, 15.4%, 9.4% | 17.6%, 1.4%, 5.2% |
| Mean recall over 3 seeds | 11.6% ± 3.3% | 8.1% ± 8.5% |
| False alarms, seeds 0, 1, 2 | 3.4%, 3.4%, 2.8% | 3.4%, 4.2%, 1.8% |
| Transient false alarms, seeds 0, 1, 2 | 3.4%, 3.4%, 2.8% (mean 3.2%) | 3.4%, 4.2%, 1.8% (mean 3.1%) |
| Threshold, seeds 0, 1, 2 | 225, 202, 256 | 145, 425, 220 |
| Mean delay | 11.9 to 12.2 s | 9.8 to 12.9 s |

- Every false alarm is still transient: the overall and transient rates match on every seed, as on D6.
- Reproduce with `python -m evaluation.gate1` (about 25 minutes). D7's numbers are in the committed `runs/gate1_cmd/metrics.json`, D6's in `runs/gate1/metrics.json`.

### Built
- `model/forecaster.py`: command awareness. The throttle history is cut into 10-sample chunks (`THROTTLE_CHUNK`), each embedded to 16 numbers plus a learned position tag (`n_command_tokens` rows, from the horizon). The fusion block is `TransformerDecoderLayer`: self-attention across channels, cross-attention from channel tokens to command tokens, feed-forward. `throttle_hist` is required, its length is checked, and a horizon that doesn't give whole chunks is refused. `CHUNK` renamed `SCORE_BATCH`
- `model/train.py`: `architecture()` records `command_aware` and `throttle_chunk`, so D6 runs are refused. `run_name` adds `_cmd`, so D6's folders and weights survive
- `evaluation/gate1.py`: the three `_cmd` seeds, results in `runs/gate1_cmd/` so D6's stay as the ablation, a transient false alarm column, and the mean transient rate the exit test is judged on
- Tests: the throttle changes the forecast, a masked channel still reads the throttle, wrong throttle lengths and a horizon of 15 are refused, D6 runs are refused, run names carry `_cmd`. 158 fast tests pass locally and in CI
- Trained 3 command-aware seeds, 65 to 67 s each

| Seed | D6 last-10 probe mean | D7 last-10 probe mean | Drop |
| --- | --- | --- | --- |
| 0 | 8.73 | 7.18 | 17.8% |
| 1 | 9.26 | 7.40 | 20.1% |
| 2 | 8.98 | 6.30 | 29.8% |
| Mean ± spread | 9.0 ± 0.3 | 7.0 ± 0.6 | |

The last-10 probe mean averages the last 10 probe values (steps 9,100 to 10,000). It's used instead of the final value because the tail wobbles: seed 2's D6 final value (7.93) was a lucky low against its last-10 mean of 8.98, which would have understated its drop. Each seed's probe batch is drawn from the same stream as on D6, so each row compares identical windows and masks. Seed 0's last 10 values only wobble (6.85 to 7.7), so 10,000 steps is enough. Reproduce from the committed `runs/*/metrics.json`:

```bash
python -c "
import json, numpy as np
for s in range(3):
    d6 = json.load(open(f'runs/h10_mask0.25_seed{s}/metrics.json'))['losses']
    d7 = json.load(open(f'runs/h10_mask0.25_cmd_seed{s}/metrics.json'))['losses']
    f6, f7, m6, m7 = d6[-1], d7[-1], np.mean(d6[-10:]), np.mean(d7[-10:])
    print(s, 'final', round(f6,3), round(f7,3), f'{1-f7/f6:.1%}', '| last-10', round(m6,2), round(m7,2), f'{1-m7/m6:.1%}')
"
```

Expected output:

```
0 final 8.412 6.982 17.0% | last-10 8.73 7.18 17.8%
1 final 9.022 7.298 19.1% | last-10 9.26 7.4 20.1%
2 final 7.933 6.053 23.7% | last-10 8.98 6.3 29.8%
```

- The D5 exit tests still pass on a command-aware model (`pytest -m slow tests/test_model.py -s`), with the fast channels much closer to their floors: feed pressure 0.0012 to 0.0007, speed 0.0006 to 0.0003

### Decided
- Fix order: command awareness first, the differenced residual second
- Command awareness design: 10-sample throttle chunks with position tags, `TransformerDecoderLayer`, a required throttle, never masked
- The ablation is D6's committed results, and new runs are named `_cmd` so D6's weights survive
- D7 exit test: transient false alarms at most 1.6% rather than below 3.2%. Each seed's false alarm rate wobbles by roughly ±0.6 to ±0.9 points from sampling alone, so new seeds of the D6 model could land at 2.9% and pass "below 3.2%". It would also be weaker than the Gate 1 rerun, which needs every seed at most 3%: seeds at 3.4%, 3.4% and 2.7% would pass "below 3.2%" and still fail Gate 1

### Expected vs actual
| Expected | Actual | Why |
| --- | --- | --- |
| With cross attention, the mean transient false alarm rate over 3 seeds drops from 3.2% to about 1%, because the throttle step in the forecast gap now has its own token. Mean recall rises from 11.6% to about 14%, because the thresholds set by those spikes fall while the bearing signal stays the same size. Gate 1's 3% false alarm cap per seed is the uncertain part: once transients are gone, per-run bias on bearing temp may set the tail instead. I think Gate 1 passes, but I'm only about 60% sure (written before any D7 code) | Wrong on every part. The mean transient false alarm rate barely moved, 3.2% to 3.1%. Mean recall fell to 8.1%, with seeds at 17.6%, 1.4% and 5.2%. Gate 1 failed again on false alarms | I assumed the transient spikes came from not knowing the commands. The sampled healthy peaks moved to startup: on the same 10 calibrate runs, D6 had 1 of 10 under 8 s and 9 mid-run (10.8 to 49.5 s), while D7 has 8 of 10 at 2.8 to 6.8 s. The driving channel shifted too, from mostly fast channels on D6 (speed 3, feed pressure 3, chamber pressure 1, coolant flow 1, nozzle temp 2) to mostly slow thermal ones on D7 (nozzle temp 5, bearing temp 2, feed pressure 2, chamber pressure 1). That fits command awareness fixing what the throttle can explain and exposing what it can't. It's 10 runs on one seed, so a hint, not a finding. Peak times come from `peak_times` in each day's gate1 output: D7 with `python -m evaluation.gate1 --limit 20`, D6 from gate1.py at commit `c2cf7fe`. Lower average forecasting error also didn't shrink the tail: seeds 0 and 1 have nearly the same last-10 probe mean (7.18 and 7.40) but thresholds of 145 and 425 |
| Diagnostic 1, written before running it. Both diagnostic predictions were written and reviewed by my tutor before the run, but committed only after it, so the commit timestamp doesn't prove the order. The tutor's review does. The text is unchanged from what was reviewed. Using `h10_mask0.25_cmd_seed0` and the first 10 healthy calibrate runs, I'll compare nozzle temp's peak residual masked against unmasked, both at startup (0 to 8 s) and in steady state (after 8 s, outside the transient window). I predict that in at least 6 of the 10 runs, the unmasked startup peak is at most half the masked startup peak, and unmasking shrinks the startup peak by a larger fraction than the steady-state peak. If fewer than 6 runs meet both conditions, the slow-state mechanism is not supported. If the startup peak halves but the steady-state peak shrinks just as much, the effect is general and not about slow state. I'm about 50% confident, because the mechanism is untested, the control makes it a stricter claim, and my last two predictions about where false alarms come from both missed | Run on D8 (Nov 19), see D8 | See D8 |
| Diagnostic 2, written before running it. For the same seed at its 2% mean threshold of 145.0 (from `runs/gate1_cmd/metrics.json`), I'll take the first alarm time of each of the 17 healthy val runs that raise a false alarm. I predict that at least 12 of the 17 first alarm before 10 s, during startup, and if fewer than 12 do, startup isn't the main source of false alarms. I'm about 70% confident, because 8 of 10 sampled healthy peaks already sat between 2.8 and 6.8 s, though those were peaks on calibrate rather than false alarms on val. This reads val again, but it only describes a result I already have and tunes nothing | Run on D8 (Nov 19), see D8 | See D8 |

### Learned
- Every D6 false alarm was transient. On each seed (17, 17 and 14 false alarm runs at 2%), every false alarm run alarmed within 5 s of a throttle change, a window covering 34.4% of a healthy run on average (measured over val's 500 healthy runs). Chance alone would rarely do that (about 0.344¹⁷ for one seed, as a rough illustration, since alarm timing isn't random), so the missing command input explains the false alarm failure. Reproduce the rates from `runs/gate1/metrics.json` and the 34.4% with `python -c "import numpy as np; from evaluation.harness import transient_mask; from generator.splits import SPLITS, fault_for, make_run; print(np.mean([transient_mask(make_run('val', s).throttle).mean() for s in SPLITS['val'] if fault_for('val', s) is None]))"`
- The model is blind for the first 1.09 s of every run (`WINDOW - 1 + horizon` samples with no forecast), and `aggregate` fills those rows with a score of 0. That's safe because floor-weighted squared residuals can't be negative, so 0 can never cross a positive threshold or set a run's peak. It would not be safe for a score that can go negative, like the differenced residual, or one where low means anomalous: there the fill must be a value that can never alarm. The blind spot costs nothing here, because bearing wear onset is drawn from 10 to 30 s (D3 exit criteria), so no fault can start before the first forecast. With real telemetry, a fault right at startup would fall in this blind spot
- Probe loss measures the typical window, while thresholds come from the worst moment in each healthy run. Command awareness cut the last-10 probe mean by 18 to 30% on every seed and still left the false alarm rate where it was. A forecasting improvement isn't a detection improvement until the tail moves
- Hypothesis, not yet tested: the remaining startup peaks come from slow state the window can't see. Nozzle temp is a heat balance with a time constant of several seconds at low coolant flow, and at startup its value depends on how long the engine has been running. In masked scoring its own history is hidden, and 1.1 s of throttle can't reconstruct that state. If so, no amount of command awareness fixes these peaks, and it would also explain the per-run bias (healthy bearing temp level 4.8 to 92.7 at the 5th to 95th percentile, std 30, against a median fault rise of 74)
- A 60% prediction that misses is normal. Recording the miss is what gives the 60% meaning

### Ledger
- **Built by hand:** the `THROTTLE_CHUNK` constant and the horizon check in `Forecaster.__init__`, `test_d6_run_is_refused`
- **Delegated, can explain:** `save_run` (why the normalizer is committed), `Normalizer.to_dict` and `from_dict`, `aggregate`, the probe curve, `load_run`'s horizon and architecture checks (all walked through today)
- **Delegated, not yet understood:** `command_tokens` and the decoder fusion, the new fast tests in `test_model.py`, the `run_name` and `architecture()` changes, the gate1.py changes
- **Moved:** from D6's "not yet understood", `save_run`, `load_run`, `to_dict` and `from_dict`, `aggregate` and the probe curve are now understood. The rest of D6's and D5's carried list remains

### Moved to tomorrow
- The diagnostic before any further fix: for the 10 healthy calibrate runs, each channel's peak residual masked against unmasked. Plus, for one seed, when each false alarm happens across all 500 healthy val runs
- Choose the next fix with the tutor from the diagnostic: the differenced residual (logged second) or a longer throttle history
- Walk through the rest of "not yet understood", D7's, D6's and D5's carried items
- This evening's architecture.md gaps: model and training settings, pipeline details, the fusion row's "1 to 2 blocks"
- Fair z score fix, due before Nov 20
- The site build log update

### Next
Thu 9:00: run the diagnostic, outside the bar, then choose the next fix with the tutor. Val has now been used for two rounds of decisions, so any further val result is reported as more optimistic, and Friday's test run is the real judgment

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
- **Delegated, can explain:** `forecast_run`, `residual_by_channel`, `residual_scores`, `to_verdict` (walked through on D6)
- **Delegated, not yet understood:** `save_run` and `load_run` (`weights_only`, the architecture check), the probe curve, `Normalizer.to_dict` and `from_dict`, `aggregate` and its tests, `tests/test_train.py`, and in `gate1.py`: `peak_times`, `evaluate_val`, `bias_check` and the printing and saving in `main`
- **Still not yet understood, carried from D5:** `evaluation/checks.py`, `tests/test_data.py`, `tests/test_model.py`, the threshold consistency test in `tests/test_harness.py`, the `Normalizer` fixes (read-only arrays, flat-channel tolerance) and the `make_windows` guards
- **Moved:** D5's walkthroughs, partly. Four functions walked through, the rest carried above

### Moved to tomorrow
- Walk through everything in "not yet understood", D6's and the items carried from D5, starting with `load_run`
- Fair z score fix, still due before Nov 20
- The site build log update, not done since D4 moved it to the weekly review

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
