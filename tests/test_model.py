from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

from contracts import CHANNELS, D_EMBED, DT_S, N_STEPS, WINDOW, Run, Verdict
from evaluation.baselines import persistence_forecast
from evaluation.budget import CATCH_WINDOW_S, REPORT_SPLIT
from generator.engine import make_throttle, simulate
from generator.splits import SPLITS, fault_for, make_run
from model.data import Normalizer
from model.encoder import ChannelEncoder
from model.forecaster import Forecaster, floor_weighted_mse, forecast_run, residual_scores, to_verdict

# Type hints only. train.py is imported inside the slow fixture, so a half-written
# train.py can't stop the fast tests from running
if TYPE_CHECKING:
    from model.train import TrainResult

HORIZON = 10
FAST_CHANNELS = ("turbopump_speed", "chamber_pressure", "coolant_flow", "feed_pressure")
DYNAMIC_CHANNELS = tuple(name for name in CHANNELS if name != "mixture_ratio")

def constant_run(value: float) -> Run:
    return Run(throttle=np.zeros(N_STEPS), sensors=np.full((N_STEPS, len(CHANNELS)), value), seed=0)

def report_seeds(healthy: bool, count: int) -> list[int]:
    return [s for s in SPLITS[REPORT_SPLIT] if (fault_for(REPORT_SPLIT, s) is None) == healthy][:count]

# Fast tests: the pieces, no training

def test_encoder_emits_contract_shape() -> None:
    assert ChannelEncoder()(torch.randn(4, len(CHANNELS), WINDOW)).shape == (4, len(CHANNELS), D_EMBED)

def test_encoder_is_shared() -> None:
    # One window copied to all 8 channels
    same = torch.randn(1, 1, WINDOW).repeat(1, len(CHANNELS), 1)
    out = ChannelEncoder()(same)[0]
    # Identical outputs: one shared encoder
    torch.testing.assert_close(out, out[:1].expand_as(out))

def test_forecaster_tells_channels_apart() -> None:
    same = torch.randn(1, 1, WINDOW).repeat(1, len(CHANNELS), 1)
    # Encoder output plus each channel's ID embedding
    tokens = Forecaster().tokens(same)[0]
    # Without channel IDs, fusion couldn't tell them apart
    assert not torch.allclose(tokens[0], tokens[1])

def test_forward_without_mask_still_works() -> None:
    assert Forecaster()(torch.randn(4, len(CHANNELS), WINDOW)).shape == (4, len(CHANNELS))

def test_forecaster_accepts_throttle_history() -> None:
    # The argument exists today so Wednesday's cross attention doesn't break every caller
    out = Forecaster(horizon=HORIZON)(torch.randn(4, len(CHANNELS), WINDOW), throttle_hist=torch.randn(4, WINDOW + HORIZON))
    assert out.shape == (4, len(CHANNELS))

def test_masked_channel_input_is_ignored() -> None:
    # Eval: no dropout, so the same input gives the same output
    model = Forecaster().eval()
    a = torch.randn(2, len(CHANNELS), WINDOW)
    b = a.clone()
    # Change only channel 3's window
    b[:, 3] = torch.randn(2, WINDOW)
    mask = torch.zeros(2, len(CHANNELS), dtype=torch.bool)
    mask[:, 3] = True
    # Its content can't leak
    torch.testing.assert_close(model(a, mask=mask), model(b, mask=mask))

def test_masked_forecast_uses_other_channels() -> None:
    model = Forecaster().eval()
    windows = torch.randn(2, len(CHANNELS), WINDOW)
    changed = windows.clone()
    # Change a different channel
    changed[:, 1] += 5.0
    mask = torch.zeros(2, len(CHANNELS), dtype=torch.bool)
    mask[:, 0] = True
    assert not torch.allclose(model(windows, mask=mask)[:, 0], model(changed, mask=mask)[:, 0])

def test_scoring_forecasts_each_channel_with_its_own_input_masked() -> None:
    # The fault shows up as one channel disagreeing with the others, so each channel's
    # scored forecast must come from the other channels, never from its own history
    model = Forecaster(horizon=HORIZON).eval()
    normalizer = Normalizer.fit([constant_run(1.0), constant_run(3.0)])
    run = make_run(REPORT_SPLIT, SPLITS[REPORT_SPLIT][0])
    sensors = np.array(run.sensors)
    sensors[:, CHANNELS.index("bearing_temp")] += 50.0
    shifted = Run(throttle=run.throttle, sensors=sensors, seed=run.seed)
    i = CHANNELS.index("bearing_temp")
    before = forecast_run(model, normalizer, run, masked=True)[:, i]
    after = forecast_run(model, normalizer, shifted, masked=True)[:, i]
    np.testing.assert_allclose(before, after)

def test_unmasked_scoring_uses_each_channels_own_history() -> None:
    # The switch for the no-masking comparison run: with masked=False, a channel's own history counts
    model = Forecaster(horizon=HORIZON).eval()
    normalizer = Normalizer.fit([constant_run(1.0), constant_run(3.0)])
    run = make_run(REPORT_SPLIT, SPLITS[REPORT_SPLIT][0])
    sensors = np.array(run.sensors)
    sensors[:, CHANNELS.index("bearing_temp")] += 50.0
    shifted = Run(throttle=run.throttle, sensors=sensors, seed=run.seed)
    i = CHANNELS.index("bearing_temp")
    # Compare only rows with a forecast, so placeholder values at the start can't decide the result
    start = WINDOW + HORIZON - 1
    before = forecast_run(model, normalizer, run, masked=False)[start:, i]
    after = forecast_run(model, normalizer, shifted, masked=False)[start:, i]
    assert not np.allclose(before, after)

@pytest.mark.parametrize("channel", range(len(CHANNELS)))
def test_loss_weights_by_noise_floor(channel: int) -> None:
    floor = torch.tensor([0.5, 0.001, 0.02, 3.0, 0.1, 0.004, 0.06, 0.7])
    target = torch.zeros(2, len(CHANNELS))
    # A miss of exactly one noise sigma on one channel
    target[:, channel] = floor[channel].sqrt()
    loss = floor_weighted_mse(torch.zeros_like(target), target, floor)
    # Same loss whichever channel missed
    assert loss.item() == pytest.approx(1 / len(CHANNELS))

def test_scores_are_zero_before_the_first_full_window() -> None:
    # Untrained, so no train.py needed
    model = Forecaster(horizon=HORIZON)
    normalizer = Normalizer.fit([constant_run(1.0), constant_run(3.0)])
    scores = residual_scores(model, normalizer, make_run(REPORT_SPLIT, SPLITS[REPORT_SPLIT][0]))
    assert scores.shape == (N_STEPS,)
    # Nothing to forecast from yet
    assert not scores[: WINDOW + HORIZON - 1].any()

# Slow exit tests: one trained model shared by all of them

@pytest.fixture(scope="module")
def trained() -> TrainResult:
    from model.train import train
    return train(horizon=HORIZON)

@pytest.mark.slow
def test_exit_loss_falls_on_nominal(trained: TrainResult) -> None:
    assert trained.losses[-1] < 0.5 * trained.losses[0], f"loss {trained.losses[0]:.3f} -> {trained.losses[-1]:.3f}"

@pytest.mark.slow
def test_exit_beats_persistence_per_channel(trained: TrainResult) -> None:
    # First sample the model can forecast. Both are scored on the same rows
    start = WINDOW + HORIZON - 1
    model_err, pers_err = [], []
    for seed in report_seeds(healthy=True, count=100):
        run = make_run(REPORT_SPLIT, seed)
        x = trained.normalizer.transform(run.sensors)
        # Unmasked: the persistence bar measures forecasting skill, and only a channel's own history
        # reveals its calibration offset and per-run level. Masked forecasts are for fault scoring
        forecast = forecast_run(trained.model, trained.normalizer, run, masked=False)
        model_err.append(np.mean((forecast[start:] - x[start:]) ** 2, axis=0))
        pers_err.append(np.mean((persistence_forecast(x, HORIZON)[start:] - x[start:]) ** 2, axis=0))
    model_err, pers_err, floor = np.mean(model_err, axis=0), np.mean(pers_err, axis=0), trained.normalizer.floor

    for name in DYNAMIC_CHANNELS:
        i = CHANNELS.index(name)
        print(f"{name:20s} model {model_err[i]:.4f}  persistence {pers_err[i]:.4f}  floor {floor[i]:.4f}")
        assert model_err[i] < pers_err[i], f"{name} doesn't beat persistence"
    for name in FAST_CHANNELS:
        i = CHANNELS.index(name)
        # Half the gap from persistence to the floor
        target = floor[i] + 0.5 * (pers_err[i] - floor[i])
        assert model_err[i] <= target, f"{name}: {model_err[i]:.4f} above half-gap target {target:.4f}"

@pytest.mark.slow
def test_exit_fault_residuals_rise(trained: TrainResult) -> None:
    rose = 0
    seeds = report_seeds(healthy=False, count=20)
    for seed in seeds:
        faulty = make_run(REPORT_SPLIT, seed)
        # Same engine and throttle, no fault
        twin = simulate(make_throttle(seed), seed)
        window = slice(round(faulty.fault.onset_s / DT_S), round((faulty.fault.onset_s + CATCH_WINDOW_S) / DT_S))
        rose += residual_scores(trained.model, trained.normalizer, faulty)[window].mean() > residual_scores(trained.model, trained.normalizer, twin)[window].mean()
    assert rose >= 0.8 * len(seeds), f"residuals rose in only {rose} of {len(seeds)} faulted runs"

@pytest.mark.slow
def test_exit_end_to_end_verdict(trained: TrainResult) -> None:
    run = make_run(REPORT_SPLIT, report_seeds(healthy=False, count=1)[0])
    result = to_verdict(trained.model, trained.normalizer, run, threshold=1.0)
    assert isinstance(result, Verdict)
    assert set(result.ranked_channels) == set(CHANNELS)
