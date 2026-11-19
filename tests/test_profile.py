import numpy as np
import pytest

from contracts import CHANNELS, DT_S, N_STEPS, Run
from evaluation.profile import (
    N_BINS,
    N_PHASES,
    RAMP_BINS,
    STD_FLOOR,
    ProfileReference,
    bin_index,
    merge_along_time,
    unpack,
)
from generator.splits import SPLITS, fault_for, make_run


def throttle_from(knots: list[tuple[float, float]]) -> np.ndarray:
    """A throttle drawn as straight lines between (time, level) knots, the way make_throttle draws one."""
    t, v = zip(*knots)
    return np.interp(np.arange(N_STEPS) * DT_S, t, v)


def throttle_profile() -> np.ndarray:
    """A known throttle: ramp to 0.75 over 3 s, step up to 0.95 at 20 s, down to 0.65 at 40 s, shutdown at 55 s."""
    return throttle_from([(0.0, 0.0), (3.0, 0.75), (20.0, 0.75), (20.2, 0.95), (40.0, 0.95), (40.2, 0.65), (55.0, 0.65), (57.5, 0.0), (60.0, 0.0)])


def throttle_stepping_down() -> np.ndarray:
    """A second, different throttle: ramp to 0.9 over 3 s, one step down to 0.65 at 20 s, shutdown at 55 s."""
    return throttle_from([(0.0, 0.0), (3.0, 0.9), (20.0, 0.9), (20.2, 0.65), (55.0, 0.65), (57.5, 0.0), (60.0, 0.0)])


def healthy_runs(n: int) -> list[Run]:
    """n healthy runs that share the known throttle. Their sensors don't matter, the tests supply values"""
    return [Run(throttle=throttle_profile(), sensors=np.zeros((N_STEPS, len(CHANNELS))), seed=s) for s in range(n)]


# Bin assignment

def test_every_sample_gets_a_valid_bin() -> None:
    keys = bin_index(throttle_profile())
    assert keys.shape == (N_STEPS,)
    assert keys.min() >= 0 and keys.max() < N_BINS


def test_bins_follow_the_throttle_profile() -> None:
    phase, direction, level = unpack(bin_index(throttle_profile()))
    # Sample: (phase, direction, level). Direction 1 is up, 0 is down. Every sample sits mid-bin, away from any edge
    expected = {
        # The ramp ends at sample 301, its first held sample. 160 × 10 // 301 is ramp bin 5. Throttle 0.4 is below 0.6
        160: (5, 1, 0),
        # 7.25 s after the ramp's last change at sample 300: time bin 14 (725 // 50), so phase 10 + 14
        1025: (24, 1, 2),
        # 18.8 s after the step up: steady bin, up, throttle 0.95
        3900: (30, 1, 4),
        # 0.1 s after the step down ended: first time bin, down, throttle 0.65
        4030: (10, 0, 1),
    }
    for sample, want in expected.items():
        assert (phase[sample], direction[sample], level[sample]) == want, f"sample {sample}"


def test_the_ramp_edge_is_where_the_throttle_stops_changing() -> None:
    # The edge convention, checked once on purpose. Sample 300 is the ramp's last changing sample, so it's
    # in the last ramp bin. Sample 301 is the first held sample, one sample after the ramp's last change
    phase, direction, level = unpack(bin_index(throttle_profile()))
    assert (phase[300], direction[300], level[300]) == (9, 1, 2)
    assert (phase[301], direction[301], level[301]) == (10, 1, 2)


def test_real_throttles_match_the_assumptions() -> None:
    # Every other test uses a hand-drawn throttle. This one ties bin_index to the generator: real throttles
    # must hold exactly still between changes and start ramping at once. If they wobbled, the ramp would
    # never end. If they sat still first, there'd be no ramp at all
    for seed in SPLITS["train"][:5]:
        phase, _, _ = unpack(bin_index(make_run("train", seed).throttle))
        # Samples in the ramp bins. The ramp lasts 2.5 to 3.5 s, so it ends at about 100 × seconds + 1
        ramp_end = int((phase < RAMP_BINS).sum())
        assert 250 <= ramp_end <= 351, f"seed {seed}: ramp ends at sample {ramp_end}"
        # Long holds reach the steady bin, which a wobbling throttle never would
        assert (phase == N_PHASES - 1).any(), f"seed {seed}: never reaches the steady bin"


def test_shutdown_bins() -> None:
    # Known limitation: during a change the time since the last change stays at 0, so all of shutdown shares
    # the first time bin and is told apart only by level. Tested as it behaves, not assumed
    phase, direction, level = unpack(bin_index(throttle_profile()))
    # 56 s, mid-shutdown: still changing, so the first time bin, down, throttle 0.39 is below 0.6
    assert (phase[5600], direction[5600], level[5600]) == (10, 0, 0)
    # 59 s, 1.5 s after shutdown ended at sample 5750: time bin 3, down, throttle 0
    assert (phase[5900], direction[5900], level[5900]) == (13, 0, 0)


# Merging along time

def test_sparse_bins_merge_with_later_ones() -> None:
    # Bins 0 to 3 only reach 1,000 together, bin 4 is full on its own, and bins 5 and 6 never reach 1,000
    groups = merge_along_time(np.array([0, 0, 500, 600, 2000, 0, 10]), min_samples=1000)
    # The leftover sparse bins at the end merge backwards into bin 4's group
    np.testing.assert_array_equal(groups, [0, 0, 0, 0, 1, 1, 1])


def test_a_column_too_sparse_to_merge_has_no_groups() -> None:
    # 30 samples in total can never reach 1,000, so the caller falls back to pooled stats
    assert merge_along_time(np.array([10, 20]), min_samples=1000) is None


# Fitting and standardizing

def test_standardizing_is_exact() -> None:
    # Half the runs read 1.0 and 10.0, the other half 3.0 and 14.0, in every sample.
    # So every bin's mean is (2, 12) and its std is (1, 2)
    runs = healthy_runs(4)
    values = {0: (1.0, 10.0), 1: (3.0, 14.0), 2: (1.0, 10.0), 3: (3.0, 14.0)}
    ref = ProfileReference.fit(runs, lambda run: np.tile(values[run.seed], (N_STEPS, 1)), min_samples=1)
    z = ref.standardize(np.tile((4.0, 16.0), (N_STEPS, 1)), throttle_profile())
    # (4 - 2) / 1 and (16 - 12) / 2
    assert z.shape == (N_STEPS, 2)
    np.testing.assert_allclose(z, 2.0)


def test_standardize_looks_up_each_samples_own_bin() -> None:
    # Every bin gets a different mean: keys % 7. Two runs read one below and one above it, so each bin's
    # mean is keys % 7 and its std is 1. Standardizing keys % 7 + 2 gives exactly 2, but only if every
    # sample is looked up in its own bin
    runs = healthy_runs(2)
    keys = bin_index(throttle_profile())
    base = (keys % 7).astype(float)
    ref = ProfileReference.fit(runs, lambda run: base - 1 if run.seed == 0 else base + 1, min_samples=1)
    np.testing.assert_allclose(ref.standardize(base + 2, throttle_profile()), 2.0)


def test_each_run_is_binned_by_its_own_throttle() -> None:
    # One run steps up, the other steps down, and they read 1.0 and 3.0. A bin only the first run visits
    # must have mean 1.0, and one only the second visits mean 3.0. If fit binned every run with the first
    # run's throttle, the second run's values would land in the first run's bins
    up = Run(throttle=throttle_profile(), sensors=np.zeros((N_STEPS, len(CHANNELS))), seed=0)
    down = Run(throttle=throttle_stepping_down(), sensors=np.zeros((N_STEPS, len(CHANNELS))), seed=1)
    ref = ProfileReference.fit([up, down], lambda run: np.full(N_STEPS, 1.0 + 2 * run.seed), min_samples=1)
    up_bins, down_bins = set(bin_index(up.throttle)), set(bin_index(down.throttle))
    only_up, only_down = sorted(up_bins - down_bins), sorted(down_bins - up_bins)
    assert only_up and only_down
    np.testing.assert_allclose(ref.mean[only_up], 1.0)
    np.testing.assert_allclose(ref.mean[only_down], 3.0)


def test_one_dimensional_values_keep_their_shape() -> None:
    # The new score has one value per sample, so standardize must hand back (N_STEPS,), not (N_STEPS, 1)
    runs = healthy_runs(2)
    ref = ProfileReference.fit(runs, lambda run: np.full(N_STEPS, 1.0 + 2 * run.seed), min_samples=1)
    assert ref.standardize(np.full(N_STEPS, 2.0), throttle_profile()).shape == (N_STEPS,)


def test_blind_rows_are_excluded() -> None:
    # The model has no forecast for the first 109 samples. If those NaNs leaked in, means would be NaN
    def values(run: Run) -> np.ndarray:
        v = np.full(N_STEPS, 5.0 + run.seed)
        v[:109] = np.nan
        return v

    ref = ProfileReference.fit(healthy_runs(2), values, min_samples=1)
    assert np.isfinite(ref.mean).all() and np.isfinite(ref.std).all()


def test_a_flat_bin_gets_the_std_floor() -> None:
    # Every run reads the same value in a given bin, so every bin's own std is 0. The values differ across
    # bins, so the pooled std isn't 0, and every bin's std must be the floor instead of 0
    runs = healthy_runs(2)
    keys = bin_index(throttle_profile())
    values = (keys % 7).astype(float)
    ref = ProfileReference.fit(runs, lambda run: values, min_samples=1)
    # Only the bins the throttle visits. Columns it never visits fall back to the pooled std instead
    visited = np.unique(keys)
    np.testing.assert_allclose(ref.std[visited], STD_FLOOR * values.std())


def test_an_unseen_column_falls_back_to_pooled_stats() -> None:
    # With an impossible minimum, no column can reach it, so every bin uses the pooled mean and std
    runs = healthy_runs(2)
    ref = ProfileReference.fit(runs, lambda run: np.full(N_STEPS, 1.0 + 2 * run.seed), min_samples=10**9)
    # Pooled over both runs: half 1.0, half 3.0
    np.testing.assert_allclose(ref.mean, 2.0)
    np.testing.assert_allclose(ref.std, 1.0)
    # The fallback is counted, so it's visible how much of the reference is coarse. Here it's everything
    assert ref.fallback_bins == N_BINS
    assert ref.fallback_samples == 2 * N_STEPS


def test_a_well_filled_reference_reports_no_fallback() -> None:
    # With merging off, every bin the throttle visits has its own stats, so nothing it visits falls back
    runs = healthy_runs(2)
    ref = ProfileReference.fit(runs, lambda run: np.full(N_STEPS, 1.0 + 2 * run.seed), min_samples=1)
    assert ref.fallback_samples == 0


def test_reference_arrays_are_read_only() -> None:
    # Frozen stops reassigning a field, not editing an array in place. Same lock as Run and Normalizer
    ref = ProfileReference.fit(healthy_runs(2), lambda run: np.full(N_STEPS, 1.0 + 2 * run.seed), min_samples=1)
    for arr in (ref.mean, ref.std):
        with pytest.raises(ValueError):
            arr[0] = 0.0


def test_fit_rejects_a_quantity_with_no_spread() -> None:
    # If every value is the same, the pooled std is 0, so the std floor is 0 too and standardizing would divide
    # by 0. Refuse to fit rather than return infinities later
    with pytest.raises(ValueError):
        ProfileReference.fit(healthy_runs(2), lambda run: np.full(N_STEPS, 4.0), min_samples=1)


def test_fit_rejects_faulted_runs() -> None:
    faulty_seed = next(s for s in SPLITS["val"] if fault_for("val", s) is not None)
    with pytest.raises(ValueError):
        ProfileReference.fit([make_run("val", faulty_seed)], lambda run: run.sensors)


def test_fit_rejects_no_runs() -> None:
    with pytest.raises(ValueError):
        ProfileReference.fit([], lambda run: run.sensors)
