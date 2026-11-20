from itertools import combinations

import numpy as np
import pytest

from contracts import CHANNELS, DT_S, N_STEPS, Run
from evaluation.attribution import TOP_K, bootstrap_interval, control_onset, overlap, rise, top_k
from evaluation.budget import CATCH_WINDOW_S, REPORT_SPLIT
from generator.engine import WEAR_CHANNELS
from generator.splits import FAULTED_SPLITS, SPLITS, fault_for, make_run
from model.data import Normalizer
from model.forecaster import Forecaster, attention_by_channel, forecast_run

# The model's first forecast. Rows before it have no forecast and no attention
FIRST_FORECAST = 109
# An onset inside the fault's 10 to 30 s range, as a sample index
ONSET_S = 20.0
ONSET = round(ONSET_S / DT_S)
END = round((ONSET_S + CATCH_WINDOW_S) / DT_S)


def blank() -> np.ndarray:
    """(N_STEPS, 8) zeros with the blind rows NaN, like a real residual or attention reading."""
    v = np.zeros((N_STEPS, len(CHANNELS)))
    v[:FIRST_FORECAST] = np.nan
    return v


# The rise: mean in the catch window minus the run's own mean before onset

def test_a_per_channel_offset_cancels() -> None:
    # The reason for the rise: masked nozzle temp's constant offset must not top every ranking
    rng = np.random.default_rng(0)
    v = blank() + rng.normal(size=(N_STEPS, len(CHANNELS)))
    offset = np.arange(len(CHANNELS)) * 50.0
    np.testing.assert_allclose(rise(v + offset, ONSET_S), rise(v, ONSET_S), atol=1e-9)


def test_a_step_at_onset_shows_in_full_on_its_channel_only() -> None:
    v = blank()
    v[ONSET:, 3] += 5.0
    expected = np.zeros(len(CHANNELS))
    expected[3] = 5.0
    np.testing.assert_allclose(rise(v, ONSET_S), expected)


def test_the_window_ends_at_onset_plus_15_s() -> None:
    # Anything after the catch window must not count
    v = blank()
    v[END:, 2] += 100.0
    np.testing.assert_allclose(rise(v, ONSET_S), 0.0)


def test_the_blind_rows_are_skipped_in_the_baseline() -> None:
    v = blank()
    v[ONSET:, 0] += 1.0
    assert np.isfinite(rise(v, ONSET_S)).all()


# Top k and overlap

def test_top_k_names_the_largest_channels_in_order() -> None:
    scores = np.array([0.1, 0.9, 0.3, 0.8, 0.0, 0.2, 0.7, 0.4])
    assert top_k(scores) == (CHANNELS[1], CHANNELS[3], CHANNELS[6])
    assert TOP_K == 3


def test_top_k_refuses_nan() -> None:
    # argsort would quietly put a NaN channel last, hiding a broken reading
    scores = np.zeros(len(CHANNELS))
    scores[2] = np.nan
    with pytest.raises(ValueError):
        top_k(scores)


def test_overlap_counts_shared_channels_out_of_k() -> None:
    assert overlap(WEAR_CHANNELS, WEAR_CHANNELS) == 1.0
    assert overlap(("chamber_pressure", "nozzle_temp", "coolant_flow"), WEAR_CHANNELS) == 0.0
    assert overlap(("bearing_temp", "nozzle_temp", "coolant_flow"), WEAR_CHANNELS) == pytest.approx(1 / 3)


def test_chance_is_37_5_percent() -> None:
    # The chance line reported next to every overlap: a random 3 of 8 against the 3 ground truth channels
    picks = list(combinations(CHANNELS, TOP_K))
    assert np.mean([overlap(p, WEAR_CHANNELS) for p in picks]) == pytest.approx(3 / 8)


# Bootstrap interval over runs

def test_bootstrap_is_reproducible() -> None:
    values = np.random.default_rng(1).choice([0, 1 / 3, 2 / 3, 1], size=200)
    assert bootstrap_interval(values) == bootstrap_interval(values)


def test_bootstrap_brackets_the_mean() -> None:
    values = np.random.default_rng(2).choice([0, 1 / 3, 2 / 3, 1], size=200)
    lo, hi = bootstrap_interval(values)
    assert lo < values.mean() < hi


def test_bootstrap_of_constant_values_has_no_width() -> None:
    assert bootstrap_interval(np.full(50, 2 / 3)) == pytest.approx((2 / 3, 2 / 3))


def test_bootstrap_refuses_no_runs() -> None:
    # The caught subset can be empty on a seed. That's reported as n = 0, never as an interval
    with pytest.raises(ValueError):
        bootstrap_interval(np.array([]))


# The healthy control's borrowed onset

@pytest.mark.parametrize("split", FAULTED_SPLITS)
def test_every_healthy_seed_borrows_the_next_seeds_onset(split: str) -> None:
    for seed in SPLITS[split]:
        if fault_for(split, seed) is None:
            assert control_onset(split, seed) == fault_for(split, seed + 1).onset_s


def test_control_onset_refuses_a_faulted_seed() -> None:
    faulted = next(s for s in SPLITS["val"] if fault_for("val", s) is not None)
    with pytest.raises(ValueError):
        control_onset("val", faulted)


# Attention each channel receives, from one unmasked pass

def model_and_run() -> tuple[Forecaster, Normalizer, Run]:
    run = make_run(REPORT_SPLIT, SPLITS[REPORT_SPLIT][0])
    return Forecaster().eval(), Normalizer.fit([run]), run


def test_attention_has_one_row_per_sample_and_one_column_per_channel() -> None:
    # Cross-attention to the 11 command tokens would give 11 columns, so this also checks it's self-attention
    model, normalizer, run = model_and_run()
    a = attention_by_channel(model, normalizer, run)
    assert a.shape == (N_STEPS, len(CHANNELS))
    assert np.isnan(a[:FIRST_FORECAST]).all()
    assert np.isfinite(a[FIRST_FORECAST:]).all()


def test_attention_received_sums_to_one_per_sample() -> None:
    # Each querying channel's weights sum to 1, so their average over queries does too
    model, normalizer, run = model_and_run()
    a = attention_by_channel(model, normalizer, run)
    np.testing.assert_allclose(a[FIRST_FORECAST:].sum(axis=1), 1.0, atol=1e-5)


def test_reading_attention_leaves_the_model_unchanged() -> None:
    # The hook must come off afterwards, or every later forecast would carry it
    model, normalizer, run = model_and_run()
    before = forecast_run(model, normalizer, run, masked=False)
    attention_by_channel(model, normalizer, run)
    after = forecast_run(model, normalizer, run, masked=False)
    np.testing.assert_array_equal(before, after)
    attn = model.fusion.layers[0].self_attn
    assert not attn._forward_hooks and not attn._forward_pre_hooks
