"""Disjoint seed ranges for the four dataset splits, and which seeds carry a fault."""

from collections.abc import Iterator
from contextlib import contextmanager

from contracts import FaultLabel, Run
from generator.engine import make_fault, make_throttle, simulate


# Seed ranges per split. Gaps between blocks leave room to grow a split without overlap.
# Never move a range once datasets exist.
SPLITS: dict[str, range] = {
    "train":     range(0, 2_000),
    "val":       range(100_000, 101_000),
    "calibrate": range(200_000, 201_000),
    "test":      range(300_000, 301_000),
}

# Sorted by start, checking each neighbor covers every pair because ranges are contiguous.
_ranges = sorted(SPLITS.values(), key=lambda r: r.start)
for a, b in zip(_ranges, _ranges[1:]):
    assert a.stop <= b.start, f"seed ranges overlap: {a} and {b}"

FAULTED_SPLITS = ("val", "calibrate", "test")  # train stays healthy: the model learns normal only


def fault_for(split: str, seed: int) -> FaultLabel | None:
    """Which fault (if any) this seed gets in this split. No simulation."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {tuple(SPLITS)}")

    if seed not in SPLITS[split]:
        raise ValueError(f"seed {seed} is not in the {split} range {SPLITS[split]}")

    if split not in FAULTED_SPLITS or seed % 2 == 0:
        return None

    return make_fault(seed)


# The test split is opened once, by the D9 script. Only open_test_split changes this
_test_open = False


@contextmanager
def open_test_split() -> Iterator[None]:
    """Allow make_run on the test split inside this block only. It locks again on exit, even after an error."""
    global _test_open
    _test_open = True
    try:
        yield
    finally:
        _test_open = False


def make_run(split: str, seed: int) -> Run:
    """Simulate one run of a split: its throttle, its fault (if any), noise on."""
    # Checked before anything is simulated. fault_for stays open, since a label comes from the seed alone
    if split == "test" and not _test_open:
        raise PermissionError("the test split is locked. Only the D9 script opens it, inside open_test_split")
    fault = fault_for(split, seed)  # also rejects a seed outside this split
    return simulate(make_throttle(seed), seed, fault=fault)
