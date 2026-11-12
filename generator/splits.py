"""Disjoint seed ranges for the four dataset splits, and which seeds carry a fault."""

from contracts import FaultLabel, Run
from generator.engine import make_fault, make_throttle, simulate


# Seed ranges per split. Gaps between blocks leave room to grow a split without overlap.
# Never move a range once datasets exist.
SPLITS: dict[str, range] = {
    "train":     range(0, 2_000),
    "val":       range(100_000, 100_400),
    "calibrate": range(200_000, 200_300),
    "test":      range(300_000, 300_400),
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


def make_run(split: str, seed: int) -> Run:
    """Simulate one run of a split: its throttle, its fault (if any), noise on."""
    fault = fault_for(split, seed)  # also rejects a seed outside this split
    return simulate(make_throttle(seed), seed, fault=fault)
