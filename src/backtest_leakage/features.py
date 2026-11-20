"""Build the design matrix, including the one feature that is allowed to leak.

Every feature here except the level anchor is strictly causal: a row at time t reads
only the target at times strictly before t, or the calendar, which is known in
advance. The level anchor is the deliberate exception. Its value at t is the mean of
the target over the rows the protocol has made available for fitting that fall within
a fixed half-width of t. That one design choice is what turns a shuffled split into a
leak: under a time-ordered split the available rows near t are all in the past and the
anchor trails the level, while under a shuffled split the available rows straddle t and
the anchor sits on the level. The leak is a property of the protocol, not of the code
that builds the feature.
"""

from __future__ import annotations

import numpy as np

from .config import FeaturePolicy


def warmup(fp: FeaturePolicy) -> int:
    """Rows at the start that cannot carry a full set of causal features.

    The longest lag or the trailing window, whichever reaches further back, sets how
    many leading rows have an incomplete history. Those rows are excluded from the
    usable universe rather than imputed, so no fabricated value ever enters a fit.
    """
    return max(max(fp.lags), fp.rolling_window)


def usable_rows(n: int, fp: FeaturePolicy) -> np.ndarray:
    """Absolute row indices that carry a full causal feature set, in time order."""
    return np.arange(warmup(fp), n, dtype=np.int64)


def _anchor(
    y: np.ndarray, rows: np.ndarray, available: np.ndarray, half: int
) -> np.ndarray:
    """Local mean of the target over available rows within ``half`` days of each row.

    Implemented with prefix sums over the available-masked series, so the cost is linear
    in the series length. When a row has no available neighbour inside the window, the
    anchor falls back to the mean over all available rows, which keeps the feature defined
    at the edges without inventing a value.
    """
    n = y.size
    mask = np.zeros(n, dtype=np.float64)
    mask[available] = 1.0
    masked_y = np.where(mask > 0, y, 0.0)

    prefix_y = np.concatenate([[0.0], np.cumsum(masked_y)])
    prefix_c = np.concatenate([[0.0], np.cumsum(mask)])

    lo = np.maximum(rows - half, 0)
    hi = np.minimum(rows + half, n - 1)
    window_sum = prefix_y[hi + 1] - prefix_y[lo]
    window_cnt = prefix_c[hi + 1] - prefix_c[lo]

    global_cnt = prefix_c[-1]
    global_mean = prefix_y[-1] / global_cnt if global_cnt > 0 else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        anchor = np.where(
            window_cnt > 0, window_sum / np.where(window_cnt > 0, window_cnt, 1.0), global_mean
        )
    return anchor


def build_matrix(
    y: np.ndarray,
    rows: np.ndarray,
    available: np.ndarray,
    fp: FeaturePolicy,
    leaky: bool,
) -> np.ndarray:
    """Return the feature matrix for ``rows`` given the ``available`` fitting set.

    ``available`` is the set of row indices the protocol permits a fitted transform to
    read. For a causal, time-ordered fold it is the training rows, all strictly in the
    past of the test fold. For a shuffled fold it is every row outside the test fold,
    which includes the future. The causal columns ignore ``available`` entirely; only
    the leaky anchor consults it, which is precisely why the leak is invisible until the
    split stops shuffling.
    """
    rows = np.asarray(rows, dtype=np.int64)
    columns = []

    for lag in fp.lags:
        columns.append(y[rows - lag])

    # Trailing rolling mean over the `rolling_window` days strictly before each row.
    w = fp.rolling_window
    prefix = np.concatenate([[0.0], np.cumsum(y)])
    trailing = (prefix[rows] - prefix[rows - w]) / w
    columns.append(trailing)

    # Calendar: a weekly cycle encoded as sine and cosine, fully known in advance.
    phase = 2.0 * np.pi * (rows % 7) / 7.0
    columns.append(np.sin(phase))
    columns.append(np.cos(phase))

    if leaky:
        columns.append(_anchor(y, rows, available, fp.anchor_halfwidth))

    return np.column_stack(columns)


def feature_names(fp: FeaturePolicy, leaky: bool) -> list[str]:
    """Column names matching :func:`build_matrix`, for reporting and tests."""
    names = [f"lag_{lag}" for lag in fp.lags]
    names.append(f"trailing_mean_{fp.rolling_window}")
    names.extend(["weekday_sin", "weekday_cos"])
    if leaky:
        names.append("level_anchor")
    return names
