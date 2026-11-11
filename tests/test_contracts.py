import json
from pathlib import Path

import numpy as np
import pytest
import torch

from contracts import (
    CHANNELS, COMMANDS, DT_S, RUN_S, N_STEPS,
    WINDOW, D_EMBED, Run, Verdict,
)

EXPECTED_CHANNELS = (
    "chamber_pressure", "turbopump_speed", "turbopump_vibration",
    "bearing_temp", "nozzle_temp", "coolant_flow",
    "feed_pressure", "mixture_ratio",
)


def test_channel_contract() -> None:
    assert CHANNELS == EXPECTED_CHANNELS          # names AND order are frozen
    assert COMMANDS == ("throttle",)


def test_run_shape() -> None:
    assert DT_S == 0.01 and RUN_S == 60.0
    assert N_STEPS == 6000


def test_model_dimensions() -> None:
    assert WINDOW == 100 and D_EMBED == 16


def test_dummy_encoder_emits_contract_shape() -> None:
    batch = 4
    window = torch.randn(batch, len(CHANNELS), WINDOW)   # (batch, 8, 100)
    dummy_encoder = torch.nn.Linear(WINDOW, D_EMBED)     # stand-in, not the real encoder
    assert dummy_encoder(window).shape == (batch, 8, 16)


def test_verdict_round_trips_to_json() -> None:
    v = Verdict(flag=True, confidence=0.87,
                ranked_channels=("bearing_temp", "turbopump_vibration"))
    text = v.to_json()
    json.loads(text)                                     # valid JSON
    assert Verdict.from_json(text) == v


def test_verdict_rejects_bad_values() -> None:
    with pytest.raises(ValueError):
        Verdict(flag=True, confidence=1.5, ranked_channels=())
    with pytest.raises(ValueError):
        Verdict(flag=True, confidence=0.5, ranked_channels=("not_a_channel",))


def test_run_accepts_contract_shapes() -> None:
    run = Run(throttle=np.zeros(N_STEPS), sensors=np.zeros((N_STEPS, 8)), seed=0)
    assert run.throttle.shape == (N_STEPS,)
    assert run.sensors.shape == (N_STEPS, 8)              # sensors[:, i] is CHANNELS[i]


def test_run_converts_lists_to_float_arrays() -> None:
    run = Run(throttle=[0] * N_STEPS, sensors=[[0] * 8] * N_STEPS, seed=0)
    assert isinstance(run.throttle, np.ndarray) and run.throttle.dtype == float
    assert isinstance(run.sensors, np.ndarray) and run.sensors.dtype == float


@pytest.mark.parametrize("throttle, sensors", [
    (np.zeros((N_STEPS, 1)), np.zeros((N_STEPS, 8))),     # throttle as a column
    (np.zeros(N_STEPS - 1), np.zeros((N_STEPS, 8))),      # throttle one step short
    (np.zeros(N_STEPS), np.zeros((N_STEPS, 9))),          # throttle stacked into sensors
    (np.zeros(N_STEPS), np.zeros((8, N_STEPS))),          # sensors transposed
])
def test_run_rejects_wrong_shapes(throttle: np.ndarray, sensors: np.ndarray) -> None:
    with pytest.raises(ValueError):
        Run(throttle=throttle, sensors=sensors, seed=0)


def test_architecture_doc_lists_every_contract() -> None:
    doc = (Path(__file__).parents[1] / "docs" / "architecture.md").read_text()
    for name in EXPECTED_CHANNELS:
        assert name in doc, f"architecture.md is missing channel {name}"
    for term in ("throttle", "100", "16", "6000", "Run", "Verdict"):
        assert term in doc, f"architecture.md is missing {term}"