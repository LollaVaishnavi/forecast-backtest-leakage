"""Shared fixtures: a loaded policy and a short series parameter set for speed.

Most tests do not care about the exact magnitude of a gap, only about a property of
it, so they use a shorter series and a narrow seed range. The few tests that assert the
headline separation use the full default parameters on purpose.
"""

from __future__ import annotations

import pytest

from backtest_leakage.config import load_policy
from backtest_leakage.generator import SeriesParams


@pytest.fixture(scope="session")
def policy():
    return load_policy()


@pytest.fixture(scope="session")
def small_params():
    # Long enough for ten folds past the warmup, short enough to keep the suite quick.
    return SeriesParams(n=160)
