import pytest

from evaluation.ablation import (
    BAR_RECALL_DROP, BAR_SHARE_MARGIN, NOCMD_MODELS, ablation_verdict, pooled_share, share, transient_counts,
)
from evaluation.gate1 import MODELS
from evaluation.harness import RunResult


def result(false_alarm: bool = False, transient: bool = False, caught: bool = False) -> RunResult:
    return RunResult(false_alarm, False, transient, caught, 1.0 if caught else None, int(false_alarm))


# The bar, as pre-registered on 2026-11-25

def test_bar_is_as_pre_registered() -> None:
    assert BAR_SHARE_MARGIN == 0.20
    assert BAR_RECALL_DROP == 0.03


def test_ablation_models_match_d7_seed_for_seed() -> None:
    assert NOCMD_MODELS == tuple(m.replace("_cmd_", "_nocmd_") for m in MODELS)


# Counting and shares

def test_transient_counts_only_healthy_false_alarms() -> None:
    rows = [
        (True, result(false_alarm=True, transient=True)),
        (True, result(false_alarm=True)),
        (True, result()),
        # A faulted run never counts toward false alarms, even if it carries the transient flag
        (False, result(transient=True, caught=True)),
    ]
    assert transient_counts(rows) == (1, 2)


def test_share_and_pooled_share() -> None:
    assert share(3, 4) == 0.75
    assert share(0, 0) is None
    # Every seed's transient runs over every seed's false alarm runs, not the mean of the shares
    assert pooled_share([(1, 1), (1, 3)]) == pytest.approx(0.5)
    assert pooled_share([(0, 0), (0, 0)]) is None


# The verdict, at its edges

def verdict(cmd, nocmd, cmd_recall=(0.5, 0.5, 0.5), nocmd_recall=(0.5, 0.5, 0.5)) -> bool:
    return ablation_verdict(list(cmd), list(nocmd), list(cmd_recall), list(nocmd_recall))[0]


def test_holds_with_a_clear_drop() -> None:
    assert verdict([(5, 10)] * 3, [(10, 10)] * 3)


def test_a_drop_of_exactly_20_points_holds() -> None:
    # Pooled 24 of 30 is 80%, against 100%
    assert verdict([(8, 10)] * 3, [(10, 10)] * 3)


def test_a_drop_just_under_20_points_fails() -> None:
    assert not verdict([(9, 10), (8, 10), (8, 10)], [(10, 10)] * 3)


def test_one_seed_not_lower_fails() -> None:
    # Pooled the drop is large, but seed 2 is level with its ablation
    assert not verdict([(0, 10), (0, 10), (10, 10)], [(10, 10)] * 3)


def test_a_model_with_no_false_alarms_fails() -> None:
    assert not verdict([(0, 0), (2, 10), (2, 10)], [(10, 10)] * 3)
    assert not verdict([(2, 10)] * 3, [(0, 0), (10, 10), (10, 10)])


def test_recall_exactly_3_points_lower_holds() -> None:
    assert verdict([(5, 10)] * 3, [(10, 10)] * 3, cmd_recall=(0.47,) * 3, nocmd_recall=(0.50,) * 3)


def test_recall_more_than_3_points_lower_fails() -> None:
    assert not verdict([(5, 10)] * 3, [(10, 10)] * 3, cmd_recall=(0.46,) * 3, nocmd_recall=(0.50,) * 3)


def test_conditions_read_as_text() -> None:
    _, conditions = ablation_verdict([(5, 10)] * 3, [(10, 10)] * 3, [0.5] * 3, [0.5] * 3)
    assert len(conditions) == 2
    assert all(isinstance(text, str) and ok for text, ok in conditions.items())
