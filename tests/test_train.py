import json
from pathlib import Path

import numpy as np
import pytest
import torch

from contracts import CHANNELS, N_STEPS, Run
from evaluation.budget import REPORT_SPLIT
from generator.splits import SPLITS, make_run
from model.data import Normalizer
from model.forecaster import Forecaster, forecast_run
from model.train import TrainResult, architecture, load_run, random_mask, run_name, save_run, train

def constant_run(value: float) -> Run:
    return Run(throttle=np.zeros(N_STEPS), sensors=np.full((N_STEPS, len(CHANNELS)), value), seed=0)

def untrained() -> TrainResult:
    torch.manual_seed(0)
    normalizer = Normalizer.fit([constant_run(1.0), constant_run(3.0)])
    config = {"horizon": 10, "mask_prob": 0.25, "seed": 0, **architecture()}
    return TrainResult(model=Forecaster().eval(), normalizer=normalizer, losses=[2.0, 1.0], config=config, seconds=1.5)

def test_random_mask_hides_at_the_requested_rate() -> None:
    mask = random_mask(100_000, 0.25, np.random.default_rng(0))
    assert mask.shape == (100_000, len(CHANNELS))
    assert mask.dtype == torch.bool
    assert mask.float().mean().item() == pytest.approx(0.25, abs=0.005)

def test_random_mask_never_hides_every_channel() -> None:
    # At p = 1 every row starts fully hidden, so this is the case that exercises the fix-up
    mask = random_mask(1_000, 1.0, np.random.default_rng(0))
    assert (mask.sum(dim=1) == len(CHANNELS) - 1).all()

@pytest.mark.parametrize("p", [-0.1, 1.5, float("nan")])
def test_random_mask_rejects_impossible_rates(p: float) -> None:
    with pytest.raises(ValueError):
        random_mask(10, p, np.random.default_rng(0))

@pytest.mark.parametrize("mask_prob", [-0.1, 1.5, float("nan")])
def test_train_rejects_impossible_mask_probs(mask_prob: float) -> None:
    # random_mask's own check never runs for these, since only mask_prob > 0 calls it
    with pytest.raises(ValueError):
        train(mask_prob=mask_prob, steps=1)

def test_train_records_the_settings_it_used() -> None:
    # A short run, so a saved config can only describe the model it came from
    result = train(mask_prob=0.0, seed=3, steps=100)
    assert result.config["mask_prob"] == 0.0
    assert result.config["seed"] == 3
    assert result.config["steps"] == 100
    # The starting probe, then one per LOG_EVERY steps
    assert len(result.losses) == 2

@pytest.mark.parametrize("mask_prob, name", [(0.0, "h10_mask0_cmd_seed0"), (0.25, "h10_mask0.25_cmd_seed0")])
def test_run_name_is_tidy(mask_prob: float, name: str) -> None:
    # Command-aware runs get their own folders, so retraining can never overwrite D6's weights
    assert run_name({"horizon": 10, "mask_prob": mask_prob, "seed": 0, "command_aware": True}) == name

def test_saved_run_scores_exactly_like_the_original(tmp_path: Path) -> None:
    # Proof that what Gate 1 evaluates is exactly what was trained
    original = untrained()
    save_run(original, tmp_path)
    loaded = load_run(tmp_path)
    run = make_run(REPORT_SPLIT, SPLITS[REPORT_SPLIT][0])
    np.testing.assert_array_equal(
        forecast_run(loaded.model, loaded.normalizer, run), forecast_run(original.model, original.normalizer, run)
    )
    assert loaded.losses == original.losses
    assert loaded.config == original.config

def test_load_refuses_a_different_architecture(tmp_path: Path) -> None:
    save_run(untrained(), tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    # Same weight shapes, different model. Loading must fail loudly
    config["n_heads"] = 2
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_run(tmp_path)

@pytest.mark.parametrize("edit", ["remove", "false"])
def test_d6_run_is_refused(tmp_path: Path, edit: str) -> None:
    # A run saved before command awareness has no command_aware in its config, so it must not load
    save_run(untrained(), tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    if edit == "remove":
        del config["command_aware"]
    else:
        config["command_aware"] = False

    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        load_run(tmp_path)