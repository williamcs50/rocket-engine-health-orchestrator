from itertools import combinations

import numpy as np
import pytest

from generator.splits import SPLITS, fault_for, make_run

def test_split_seeds_never_overlap() -> None:
    assert set(SPLITS) == {"train", "val", "calibrate", "test"}
    for a, b in combinations(SPLITS, 2):
        shared = set(SPLITS[a]) & set(SPLITS[b])
        assert not shared, f"{a} and {b} share {len(shared)} seeds, e.g. {min(shared)}"

def test_train_is_healthy() -> None:
    faulted = [seed for seed in SPLITS["train"] if fault_for("train", seed) is not None]
    assert not faulted, f"{len(faulted)} train seeds get a fault, e.g. {faulted[0]}"

@pytest.mark.parametrize("split", ["val", "calibrate", "test"])
def test_val_calibrate_test_contain_faults(split: str) -> None:
    faults = [fault_for(split, seed) for seed in SPLITS[split]]
    assert any(f is None for f in faults), f"{split} has no healthy seeds"
    assert any(f is not None for f in faults), f"{split} has no faulted seeds"

@pytest.mark.parametrize("split", ["train", "val", "calibrate", "test"])
def test_split_runs_are_reproducible(split: str) -> None:
    for seed in SPLITS[split][:2]:
        a, b = make_run(split, seed), make_run(split, seed)
        np.testing.assert_array_equal(a.sensors, b.sensors)
        assert a.fault == b.fault == fault_for(split, seed)

def test_make_run_rejects_seed_from_another_split() -> None:
    with pytest.raises(ValueError):
        make_run("test", SPLITS["train"][0])

def test_fault_for_rejects_unknown_split() -> None:
    with pytest.raises(ValueError):
        fault_for("holdout", 0)
