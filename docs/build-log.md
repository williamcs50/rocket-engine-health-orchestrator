# Build Log

One entry per session, newest first. Full decision entries live in [architecture.md](architecture.md).

## D8 · Thu Nov 19, 2026 · P4 calibration + P5 experiment

**Exit test (calibration):** _pass or fail_. From the plan: the held-out nominal share above q is within ±3 points of 1 − q, for q = 0.8, 0.9 and 0.95. The P5 experiment (GRU against cross-attention) is dropped, per the Gate 1 bar.

**Gate 1 rerun with the new score:** _pass, fail or not run_. Same bar, unchanged. Reported as optimistic, since this is val's third round of decisions.

### Plan and cutoffs, set at 11:47 AM
- Order: the fair z score fix first, then calibration, then the new score. The z score fix is independent of the model and the smallest piece, and the final table compares against it. Calibration is today's exit test, and its code works on any score. The new score is the largest and least certain, so it goes last, where running long can't cost tomorrow's test run anything
- If the new score is adopted, calibration is rerun on it before tomorrow. That only means scoring calibrate's runs again
- Adoption rule: the new score is adopted only if it passes Gate 1 on val, all three conditions. Otherwise tomorrow's test runs with D7's scoring (mean aggregate). If both fail Gate 1, the test still runs with D7's scoring and is reported as having failed Gate 1 on val. All 3 seeds go to test, reported per seed plus mean and spread. No seed is chosen
- Calibration exit test: held out on val's 500 healthy runs today, and test's tomorrow. Counted per run, using each run's peak confidence, which matches how Gate 1 counts false alarms
- Calibration starts by 1:30 PM. If temperature scaling hasn't met the ±3 point check by 3:00 PM, use the plan's fallback, the empirical quantile
- The new score starts only after calibration and the z score fix. If it isn't built and evaluated by 5:30 PM, it's dropped, and tomorrow's test runs with D7's scoring

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
- _

### Decided
- Next scoring fix: a standardized, differenced residual, fully specified in the decision log before any code. Judged by the Gate 1 bar, unchanged, with no separate exit test

### Expected vs actual
| Expected | Actual | Why |
| --- | --- | --- |
| Diagnostic 1 (from D7): in at least 6 of 10 runs, nozzle temp's unmasked startup peak is at most half its masked startup peak, and unmasking shrinks the startup peak by a larger fraction than the steady-state peak. About 50% confident | Met as written, 7 of 10, but the margins on the control are negligible. In most runs the two ratios are nearly equal (0.05 against 0.06, 0.13 against 0.15, 0.18 against 0.19). Only 200018 (0.23 against 0.75) shows a clear difference | My own refutation clause said that if the steady-state peak shrinks just as much, the effect is general, and that's essentially what happened. Unmasking cuts nozzle temp's residual by 70 to 95% everywhere. The evidence points to a general masked nozzle problem, such as a per-run calibration offset invisible while the channel is hidden, not a startup-specific one. The masked startup peak is still larger than the masked steady peak in all 10 runs |
| Diagnostic 2 (from D7): at least 12 of seed 0's 17 false alarm runs first alarm before 10 s. About 70% confident | Supported: 14 of 17 first alarm before 10 s. 5 of them alarm at 1.1 s, the first sample the model can score, 9 more between 2.6 and 5.5 s, and 3 mid-run | The false alarms are a startup problem, and almost a third fire at the very first forecast. This read val again, but only described a result already in hand |
| With the standardized differenced score, each seed's val false alarm rate falls to at most 3%, so Gate 1 passes. Mean recall stays at least 8.1%, D7's value. Seed 1's false alarm rate, 4.2% on D7, drops the most of the three. About 40% confident, because my last two predictions about false alarms both missed and this score is untested (written before any code) | _ | _ |

### Learned
- A control compared with a strict "<" needs a margin, or noise decides the result. Next time, write how much lower counts, for example "at least 0.05 lower"
- Masked nozzle temp is badly predicted at all times (steady-state peaks of 31 to 260, against 13 to 23 unmasked), and worst at startup. That fits a per-run offset the other channels can't reveal

### Ledger
- **Built by hand:** _
- **Delegated, can explain:** _
- **Delegated, not yet understood:** `slow_state` in `evaluation/checks.py`
- **Moved:** _

### Moved to tomorrow
- _

### Next
_

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
| With cross attention, the mean transient false alarm rate over 3 seeds drops from 3.2% to about 1%, because the throttle step in the forecast gap now has its own token. Mean recall rises from 11.6% to about 14%, because the thresholds set by those spikes fall while the bearing signal stays the same size. Gate 1's 3% false alarm cap per seed is the uncertain part: once transients are gone, per-run bias on bearing temp may set the tail instead. I think Gate 1 passes, but I'm only about 60% sure (written before any D7 code) | Wrong on every part. The mean transient false alarm rate barely moved, 3.2% to 3.1%. Mean recall fell to 8.1%, with seeds at 17.6%, 1.4% and 5.2%. Gate 1 failed again on false alarms | I assumed the transient spikes came from not knowing the commands. The sampled healthy peaks moved to startup: on the same 10 calibrate runs, D6 had 1 of 10 under 8 s and 9 mid-run (10.8 to 49.5 s), while D7 has 8 of 10 at 2.8 to 6.8 s. The driving channel shifted too, from mostly fast channels on D6 (speed 3, feed pressure 3, chamber pressure 1, coolant flow 1, nozzle temp 2) to mostly slow thermal ones on D7 (nozzle temp 5, bearing temp 2, feed pressure 2, chamber pressure 1). That fits command awareness fixing what the throttle can explain and exposing what it can't. It's 10 runs on one seed, so a hint, not a finding. Peak times come from `peak_times` in each day's gate1 output: D7 with `python -m evaluation.gate1 --limit 20`, D6 from gate1.py at commit c2cf7fe. Lower average forecasting error also didn't shrink the tail: seeds 0 and 1 have nearly the same last-10 probe mean (7.18 and 7.40) but thresholds of 145 and 425 |
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
