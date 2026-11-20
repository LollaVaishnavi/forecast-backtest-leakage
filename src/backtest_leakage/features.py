"""Build the design matrix, including the one feature that is allowed to leak.

Every feature here except the level anchor is strictly causal: a row at time t reads
only the target at times strictly before t, or the calendar, which is known in
advance. The level anchor is the deliberate exception. Its value at t is the mean of
the target over the available rows nearest t in time, a fixed count of them. That one
design choice is what turns a shuffled split into a leak: under a time-ordered split the
nearest available rows are all in the past and the anchor trails the level, while under a
shuffled split they straddle t and the anchor sits on the level. The leak is a property of
the protocol, not of the code that builds the feature.
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
    """Mean of the target over the available rows nearest each row in time.

    The anchor at t is the mean of the ``2 * half`` available rows closest to t by day
    index. Using the nearest available rows rather than a fixed calendar window is what
    keeps the feature well behaved under a time-ordered split: with only the past
    available, the nearest rows are the most recent ones, so the anchor is a proper
    trailing mean that never degenerates into a global average. Under a shuffled split
    the nearest available rows straddle t, so the same code returns a centred mean that
    sits on the level. The number of rows averaged is identical in both cases; only
    their position relative to t changes, which is the leak.
    """
    av = np.sort(np.asarray(available, dtype=np.int64))
    k = min(2 * half, av.size)
    if k == 0:
        return np.zeros(rows.size, dtype=np.float64)

    pos = np.searchsorted(av, rows)
    out = np.empty(rows.size, dtype=np.float64)
    m = av.size
    for i, (t, p) in enumerate(zip(rows, pos, strict=True)):
        lo = max(0, p - k)
        hi = min(m, p + k)
        candidate = av[lo:hi]
        # Of the up-to-2k candidates around the insertion point, keep the k nearest in
        # index distance; those are the k rows whose target the anchor averages.
        nearest = np.argsort(np.abs(candidate - t), kind="stable")[:k]
        chosen = candidate[nearest]
        out[i] = float(np.mean(y[chosen]))
    return out


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
