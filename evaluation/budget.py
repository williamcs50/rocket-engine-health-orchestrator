"""Evaluation rules fixed before any model exists. Changing one needs a decision-log entry."""

from generator.engine import SUB_THRESHOLD_S

FA_BUDGET = 0.02  # at most 2% of healthy runs may raise any alarm
FA_SECONDARY = (0.01, 0.05)  # pre-registered secondary operating points
CATCH_WINDOW_S = SUB_THRESHOLD_S  # a catch is a first alarm within onset + this many seconds
TRANSIENT_S = 5.0  # s after a commanded throttle change during which a false alarm is "transient"
THRESHOLD_SPLIT = "calibrate"  # thresholds come from this split's healthy runs
REPORT_SPLIT = "val"  # results are reported here. Test is opened once, on D9
