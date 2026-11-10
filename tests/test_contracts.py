import json
from pathlib import Path

import pytest
import torch

from contracts import (
    CHANNELS, COMMANDS, DT_S, RUN_S, N_STEPS,
    WINDOW, D_EMBED, Verdict,
)

EXPECTED_CHANNELS = (
    "chamber_pressure", "turbopump_speed", "turbopump_vibration",
    "bearing_temp", "nozzle_temp", "coolant_flow",
    "feed_pressure", "mixture_ratio",
)


def test_channel_contract():
    assert CHANNELS == EXPECTED_CHANNELS          # names AND order are frozen
    assert COMMANDS == ("throttle",)


def test_run_shape():
    assert DT_S == 0.01 and RUN_S == 60.0
    assert N_STEPS == 6000


def test_model_dimensions():
    assert WINDOW == 100 and D_EMBED == 16


def test_dummy_encoder_emits_contract_shape():
    batch = 4
    window = torch.randn(batch, len(CHANNELS), WINDOW)   # (batch, 8, 100)
    dummy_encoder = torch.nn.Linear(WINDOW, D_EMBED)     # stand-in, not the real encoder
    assert dummy_encoder(window).shape == (batch, 8, 16)


def test_verdict_round_trips_to_json():
    v = Verdict(flag=True, confidence=0.87,
                ranked_channels=("bearing_temp", "turbopump_vibration"))
    text = v.to_json()
    json.loads(text)                                     # valid JSON
    assert Verdict.from_json(text) == v


def test_verdict_rejects_bad_values():
    with pytest.raises(ValueError):
        Verdict(flag=True, confidence=1.5, ranked_channels=())
    with pytest.raises(ValueError):
        Verdict(flag=True, confidence=0.5, ranked_channels=("not_a_channel",))


def test_architecture_doc_lists_every_contract():
    doc = (Path(__file__).parents[1] / "docs" / "architecture.md").read_text()
    for name in EXPECTED_CHANNELS:
        assert name in doc, f"architecture.md is missing channel {name}"
    for term in ("throttle", "100", "16", "6000", "Verdict"):
        assert term in doc, f"architecture.md is missing {term}"