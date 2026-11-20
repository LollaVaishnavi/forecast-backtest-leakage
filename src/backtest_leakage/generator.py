"""A seeded synthetic demand series with a known, tunable level process.

The fixture is synthetic on purpose (ADR 005): the only way to measure whether a
leakage detector fires on a leak and stays quiet on a clean series is to hold a
ground-truth switch for the leak, which no real dataset gives you. The series is
built so that the level a forecaster needs is genuinely hard to track from the past
alone, which is what gives a future-aware feature something to leak. Everything is
driven by one numpy Generator seeded per series, so a reviewer reproduces every row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeriesParams:
    """Parameters of the data-generating process.

    The defaults produce a non-stationary daily series: a drifting level with
    occasional regime shifts, a weekly seasonal cycle, and autoregressive noise. The
    drift and the shifts are what make the level hard to predict from trailing history,
    so a feature that is allowed to see the future level gains a real, measurable edge.
    """

    n: int = 420
    base_level: float = 100.0
    drift_per_day: float = 0.18
    weekly_amplitude: float = 8.0
    ar_rho: float = 0.55
    ar_sigma: float = 6.0
    regime_every: int = 84
    regime_jump: float = 22.0


def make_series(seed: int, params: SeriesParams | None = None) -> np.ndarray:
    """Return one demand series of length ``params.n`` as a float array.

    The construction is deterministic in ``seed``: the same seed returns a
    byte-identical array, which is what lets the null band and every published figure
    be reproduced. Two different seeds give two independent series drawn from the same
    process, which is how the null band is estimated across clean series.
    """
    p = params or SeriesParams()
    rng = np.random.default_rng(seed)

    t = np.arange(p.n)

    # A deterministic drift plus regime shifts: the shifts are the part trailing
    # history cannot anticipate, so a past-only level estimate lags right after one.
    level = p.base_level + p.drift_per_day * t
    if p.regime_every > 0:
        # Signed jumps at each regime boundary, direction chosen by the seed.
        n_regimes = p.n // p.regime_every
        signs = rng.choice((-1.0, 1.0), size=n_regimes)
        for k, sign in enumerate(signs, start=1):
            level = level + np.where(t >= k * p.regime_every, sign * p.regime_jump, 0.0)

    # Weekly seasonality: a fixed weekday shape, not a pure sine, so it is learnable
    # from lag-7 but not trivially so.
    weekday = t % 7
    weekly_shape = np.array([0.0, 0.3, 0.1, -0.2, 0.4, 1.0, 0.7])
    seasonal = p.weekly_amplitude * weekly_shape[weekday]

    # AR(1) noise: serially correlated, so neighbouring rows are informative, which is
    # exactly the structure a shuffled split quietly exploits.
    noise = np.empty(p.n)
    noise[0] = rng.normal(0.0, p.ar_sigma)
    innovations = rng.normal(0.0, p.ar_sigma, size=p.n)
    for i in range(1, p.n):
        noise[i] = p.ar_rho * noise[i - 1] + innovations[i]

    series = level + seasonal + noise
    return series.astype(np.float64)
