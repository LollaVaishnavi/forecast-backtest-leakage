"""Determinism and the regression that pins the anchor war story."""

from __future__ import annotations

from backtest_leakage.gap import calibrate_null, evaluate_series
from backtest_leakage.generator import make_series


def test_an_evaluation_is_byte_identical_across_runs(policy):
    y = make_series(3)
    first = evaluate_series(y, policy, leaky=True)
    second = evaluate_series(y, policy, leaky=True)
    assert first.gap == second.gap
    assert first.kfold.per_fold_rmse == second.kfold.per_fold_rmse


def test_the_null_band_is_reproducible(policy, small_params):
    a = calibrate_null(policy, small_params, seeds=range(0, 8))
    b = calibrate_null(policy, small_params, seeds=range(0, 8))
    assert a.gaps == b.gaps
    assert a.threshold == b.threshold


def test_the_honest_leaky_error_stays_in_a_believable_range(policy):
    # War story, pinned. The first anchor used a fixed calendar window and fell back to
    # the global mean when a wide test block had no in-window past. That made the honest
    # walk-forward RMSE of the leaked pipeline on seed 0 come out at 26.2, which read as
    # a broken feature rather than a leak and would have discredited the gap. Switching
    # to a nearest-available-rows anchor (ADR 003) brought it to about 11.4. This test
    # fails if the degeneracy ever returns: an honest error above 20 here means the
    # anchor has collapsed to a global average again.
    y = make_series(0)
    leaked = evaluate_series(y, policy, leaky=True)
    assert leaked.walk_forward.rmse < 20.0
    assert evaluate_series(y, policy, leaky=False).gap < 3.0
