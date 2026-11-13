from evaluation.budget import (
    CATCH_WINDOW_S, FA_BUDGET, FA_SECONDARY, REPORT_SPLIT, THRESHOLD_SPLIT, TRANSIENT_S,
)
from generator.engine import SUB_THRESHOLD_S
from generator.splits import SPLITS

def test_budget_is_locked() -> None:
    # Logged 2026-11-13, before any model exists. Changing these needs a decision-log entry.
    assert FA_BUDGET == 0.02
    assert FA_SECONDARY == (0.01, 0.05)
    assert CATCH_WINDOW_S == SUB_THRESHOLD_S == 15.0
    assert TRANSIENT_S == 5.0
    assert (THRESHOLD_SPLIT, REPORT_SPLIT) == ("calibrate", "val")
    assert THRESHOLD_SPLIT in SPLITS and REPORT_SPLIT in SPLITS  # a typo fails here, not mid-run
