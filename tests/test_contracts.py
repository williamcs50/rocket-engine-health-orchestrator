import json
from pathlib import Path

import numpy as np
import pytest
import torch

from contracts import (
    CHANNELS, COMMANDS, DT_S, RUN_S, N_STEPS,
    WINDOW, D_EMBED, FaultLabel, Run, Verdict,
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


GOOD_LABEL = dict(kind="bearing_wear", onset_s=20.0, rate=0.05,
                  channels=("bearing_temp", "turbopump_vibration", "turbopump_speed"))


def test_run_carries_a_fault_label() -> None:
    label = FaultLabel(**GOOD_LABEL)
    healthy = Run(throttle=np.zeros(N_STEPS), sensors=np.zeros((N_STEPS, 8)), seed=0)
    faulty = Run(throttle=np.zeros(N_STEPS), sensors=np.zeros((N_STEPS, 8)), seed=0, fault=label)
    assert healthy.fault is None                          # old calls still mean healthy
    assert faulty.fault == label


def test_run_rejects_fault_that_is_not_a_label() -> None:
    with pytest.raises(ValueError):
        Run(throttle=np.zeros(N_STEPS), sensors=np.zeros((N_STEPS, 8)), seed=0, fault=GOOD_LABEL)


def test_run_arrays_are_read_only() -> None:
    sensors = np.zeros((N_STEPS, 8))
    run = Run(throttle=np.zeros(N_STEPS), sensors=sensors, seed=0)
    with pytest.raises(ValueError):
        run.sensors[0, 0] = 999.0
    with pytest.raises(ValueError):
        run.throttle[0] = 1.0
    sensors[0, 0] = 1.0                                   # edit the caller's original array...
    assert run.sensors[0, 0] == 0.0                       # ...the run's copy is unaffected


def test_fault_label_converts_channels_to_tuple() -> None:
    label = FaultLabel(**{**GOOD_LABEL, "channels": ["bearing_temp", "turbopump_vibration"]})
    assert label.channels == ("bearing_temp", "turbopump_vibration")
    from_generator = FaultLabel(**{**GOOD_LABEL, "channels": (c for c in ["bearing_temp"])})
    assert from_generator.channels == ("bearing_temp",)  # not used up by validation


@pytest.mark.parametrize("onset_s", [0.0, RUN_S])
def test_fault_label_accepts_onset_at_run_edges(onset_s: float) -> None:
    assert FaultLabel(**{**GOOD_LABEL, "onset_s": onset_s}).onset_s == onset_s


@pytest.mark.parametrize("field, bad", [
    ("kind", "turbine_crack"),                            # unknown fault type
    ("onset_s", -0.01),                                   # just before the run starts
    ("onset_s", RUN_S + 0.01),                            # just after the run ends
    ("rate", 0.0),                                        # wear must increase friction
    ("rate", -0.05),
    ("rate", float("nan")),                               # NaN fails every comparison
    ("channels", ("bearing_tmp",)),                       # typo in a channel name
    ("channels", ()),                                     # a fault must act on something
])
def test_fault_label_rejects_bad_values(field: str, bad: object) -> None:
    with pytest.raises(ValueError):
        FaultLabel(**{**GOOD_LABEL, field: bad})


def test_architecture_doc_lists_every_contract() -> None:
    doc = (Path(__file__).parents[1] / "docs" / "architecture.md").read_text()
    for name in EXPECTED_CHANNELS:
        assert name in doc, f"architecture.md is missing channel {name}"
    for term in ("throttle", "100", "16", "6000", "`Run`", "`Verdict`", "`FaultLabel`"):
        assert term in doc, f"architecture.md is missing {term}"