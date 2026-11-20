"""Properties of the synthetic series: determinism, shape, and non-stationarity."""

from __future__ import annotations

import numpy as np

from backtest_leakage.generator import SeriesParams, make_series


def test_the_same_seed_returns_a_byte_identical_series():
    a = make_series(7)
    b = make_series(7)
    assert np.array_equal(a, b)


def test_different_seeds_return_different_series():
    assert not np.array_equal(make_series(1), make_series(2))


def test_the_series_has_the_requested_length():
    y = make_series(0, SeriesParams(n=200))
    assert y.shape == (200,)


def test_turning_regimes_off_removes_the_jumps():
    flat = make_series(0, SeriesParams(regime_every=0, ar_sigma=0.0, weekly_amplitude=0.0))
    # With no regimes, no noise and no seasonality, the series is a pure linear drift.
    diffs = np.diff(flat)
    assert np.allclose(diffs, diffs[0])


def test_the_level_drifts_so_the_second_half_differs_from_the_first():
    y = make_series(3)
    first = y[: y.size // 2].mean()
    second = y[y.size // 2:].mean()
    # A non-stationary mean is the whole reason a time-ordered backtest is needed.
    assert abs(second - first) > 1.0
