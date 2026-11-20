"""Properties of the design matrix, including which features consult the future."""

from __future__ import annotations

import numpy as np

from backtest_leakage.config import load_policy
from backtest_leakage.features import (
    _anchor,
    build_matrix,
    feature_names,
    usable_rows,
    warmup,
)
from backtest_leakage.generator import make_series


def _fp():
    return load_policy().features


def test_warmup_is_the_furthest_reaching_causal_lookback():
    fp = _fp()
    assert warmup(fp) == max(max(fp.lags), fp.rolling_window)


def test_usable_rows_begin_after_the_warmup():
    fp = _fp()
    rows = usable_rows(160, fp)
    assert rows[0] == warmup(fp)
    assert rows[-1] == 159


def test_the_matrix_has_one_column_per_named_feature():
    fp = _fp()
    y = make_series(0)
    rows = usable_rows(y.size, fp)[:20]
    for leaky in (False, True):
        x = build_matrix(y, rows, rows, fp, leaky)
        assert x.shape == (20, len(feature_names(fp, leaky)))


def test_the_causal_columns_ignore_which_rows_are_available():
    # The lag and calendar columns must not change when the available set changes;
    # only the leaky anchor is allowed to. This pins the causal/leaky boundary.
    fp = _fp()
    y = make_series(1)
    rows = usable_rows(y.size, fp)[50:70]
    past = rows - 30
    everything = usable_rows(y.size, fp)
    clean_a = build_matrix(y, rows, past, fp, leaky=False)
    clean_b = build_matrix(y, rows, everything, fp, leaky=False)
    assert np.array_equal(clean_a, clean_b)


def test_the_anchor_moves_when_the_future_becomes_available():
    fp = _fp()
    y = make_series(1)
    rows = usable_rows(y.size, fp)[60:80]
    past_only = np.arange(0, int(rows[0]))
    with_future = usable_rows(y.size, fp)
    anchor_past = _anchor(y, rows, past_only, fp.anchor_halfwidth)
    anchor_future = _anchor(y, rows, with_future, fp.anchor_halfwidth)
    # On a drifting series the centred anchor sits higher than the trailing one.
    assert not np.allclose(anchor_past, anchor_future)


def test_with_only_the_past_available_the_anchor_is_a_recent_trailing_mean():
    # The war-story property: a deep test row must not collapse to the global mean.
    # It must average the nearest available (most recent past) rows instead.
    fp = _fp()
    y = make_series(2)
    rows = np.array([120])
    past_only = np.arange(0, 100)
    k = 2 * fp.anchor_halfwidth
    got = _anchor(y, rows, past_only, fp.anchor_halfwidth)[0]
    expected_trailing = float(np.mean(y[100 - k:100]))
    global_mean = float(np.mean(y[past_only]))
    assert abs(got - expected_trailing) < 1e-9
    assert abs(got - global_mean) > 1.0


def test_an_empty_available_set_yields_a_defined_anchor():
    fp = _fp()
    y = make_series(0)
    rows = usable_rows(y.size, fp)[:5]
    got = _anchor(y, rows, np.array([], dtype=np.int64), fp.anchor_halfwidth)
    assert np.array_equal(got, np.zeros(5))
