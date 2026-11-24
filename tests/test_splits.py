from itertools import combinations

import numpy as np
import pytest

import generator.splits as splits
from generator.splits import LOCKED_SPLITS, SPLITS, fault_for, make_run, open_test_split

def test_split_seeds_never_overlap() -> None:
    assert set(SPLITS) == {"train", "val", "calibrate", "test", "test2", "val2", "test3"}
    for a, b in combinations(SPLITS, 2):
        shared = set(SPLITS[a]) & set(SPLITS[b])
        assert not shared, f"{a} and {b} share {len(shared)} seeds, e.g. {min(shared)}"

def test_train_is_healthy() -> None:
    faulted = [seed for seed in SPLITS["train"] if fault_for("train", seed) is not None]
    assert not faulted, f"{len(faulted)} train seeds get a fault, e.g. {faulted[0]}"

@pytest.mark.parametrize("split", ["val", "calibrate", "test", "test2", "val2", "test3"])
def test_val_calibrate_test_contain_faults(split: str) -> None:
    faults = [fault_for(split, seed) for seed in SPLITS[split]]
    assert any(f is None for f in faults), f"{split} has no healthy seeds"
    assert any(f is not None for f in faults), f"{split} has no faulted seeds"

# Test and test2 are left out, since only the test run script may simulate them. Val and calibrate run the same code
@pytest.mark.parametrize("split", ["train", "val", "calibrate"])
def test_split_runs_are_reproducible(split: str) -> None:
    for seed in SPLITS[split][:2]:
        a, b = make_run(split, seed), make_run(split, seed)
        np.testing.assert_array_equal(a.sensors, b.sensors)
        assert a.fault == b.fault == fault_for(split, seed)

def test_make_run_rejects_seed_from_another_split() -> None:
    # Val, not test, so this checks the seed range and not the test split guard
    with pytest.raises(ValueError):
        make_run("val", SPLITS["train"][0])


# The test split guard

def test_test_split_is_locked_by_default() -> None:
    with pytest.raises(PermissionError):
        make_run("test", SPLITS["test"][0])


def test_test_split_opens_inside_the_context() -> None:
    with open_test_split():
        run = make_run("test", SPLITS["test"][0])
    assert run.fault == fault_for("test", SPLITS["test"][0])


def test_test_split_locks_again_after_the_context() -> None:
    with open_test_split():
        pass
    with pytest.raises(PermissionError):
        make_run("test", SPLITS["test"][0])


def test_test_split_locks_again_after_an_error_inside() -> None:
    # A crash mid-run must not leave the split open for whatever runs next
    with pytest.raises(RuntimeError):
        with open_test_split():
            raise RuntimeError("crash mid-run")
    with pytest.raises(PermissionError):
        make_run("test", SPLITS["test"][0])


def test_test_labels_stay_readable_without_the_context() -> None:
    # Labels come from the seed alone, with no simulated data, so fault_for isn't guarded
    assert fault_for("test", SPLITS["test"][1]) is not None

# Test2, pre-registered on 2026-11-23. No test here may simulate a test2 run, so opening it is checked with the
# simulator stubbed out

def test_test2_is_seeds_400000_to_400999() -> None:
    assert SPLITS["test2"] == range(400_000, 401_000)


def test_every_held_out_split_is_locked() -> None:
    assert set(LOCKED_SPLITS) == {"test", "test2", "test3"}


# Val2 and test3, pre-registered on 2026-11-24. Val2 is for development, test3 is held out

def test_val2_and_test3_seed_ranges() -> None:
    assert SPLITS["val2"] == range(500_000, 501_000)
    assert SPLITS["test3"] == range(600_000, 601_000)


def test_val2_is_open_for_development() -> None:
    run = make_run("val2", SPLITS["val2"][1])
    assert run.fault == fault_for("val2", SPLITS["val2"][1])


def test_test3_is_refused_by_default() -> None:
    # Refused before anything is simulated, so this reads no data
    with pytest.raises(PermissionError):
        make_run("test3", SPLITS["test3"][0])


@pytest.mark.parametrize("split", ["test", "test2"])
def test_a_locked_split_is_refused_by_default(split: str) -> None:
    # Refused before anything is simulated, so this reads no data
    with pytest.raises(PermissionError):
        make_run(split, SPLITS[split][0])


def test_test2_opens_inside_its_own_context(monkeypatch: pytest.MonkeyPatch) -> None:
    simulated = []
    monkeypatch.setattr(splits, "simulate", lambda throttle, seed, fault=None: simulated.append(seed))
    with open_test_split("test2"):
        make_run("test2", SPLITS["test2"][0])
    assert simulated == [SPLITS["test2"][0]]


def test_opening_one_locked_split_leaves_the_other_locked() -> None:
    with open_test_split("test2"):
        with pytest.raises(PermissionError):
            make_run("test", SPLITS["test"][0])
    with open_test_split("test"):
        with pytest.raises(PermissionError):
            make_run("test2", SPLITS["test2"][0])


def test_test2_locks_again_after_an_error_inside() -> None:
    with pytest.raises(RuntimeError):
        with open_test_split("test2"):
            raise RuntimeError("crash mid-run")
    with pytest.raises(PermissionError):
        make_run("test2", SPLITS["test2"][0])


def test_only_locked_splits_can_be_opened() -> None:
    # Opening val would do nothing, so asking for it is a mistake worth catching
    with pytest.raises(ValueError):
        with open_test_split("val"):
            pass


def test_fault_for_rejects_unknown_split() -> None:
    with pytest.raises(ValueError):
        fault_for("holdout", 0)
