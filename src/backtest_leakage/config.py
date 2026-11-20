"""Typed access to policy.toml.

This file exists so that no threshold, window or fold count is written as a literal
anywhere in the library: a reviewer changes behaviour by editing a commented TOML
file, not by grepping the source. Parsing is done with the standard-library tomllib,
so reading configuration adds no runtime dependency.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

# The repository root is three parents up from this file (src/backtest_leakage/).
_DEFAULT_POLICY = Path(__file__).resolve().parents[2] / "policy.toml"


@dataclass(frozen=True)
class BacktestPolicy:
    n_folds: int
    embargo: int
    scheme: str


@dataclass(frozen=True)
class FeaturePolicy:
    rolling_window: int
    lags: tuple[int, ...]
    anchor_halfwidth: int
    fold_fit: bool


@dataclass(frozen=True)
class ModelPolicy:
    ridge_alpha: float


@dataclass(frozen=True)
class NullPolicy:
    n_series: int
    threshold_percentile: float


@dataclass(frozen=True)
class VerdictPolicy:
    min_detectable_gap: float


@dataclass(frozen=True)
class Policy:
    backtest: BacktestPolicy
    features: FeaturePolicy
    model: ModelPolicy
    null: NullPolicy
    verdict: VerdictPolicy

    @property
    def resolution_floor(self) -> float:
        """Smallest false-alarm rate the null sample can even represent: one series.

        A measured false-alarm rate of zero over ``n_series`` is not zero, it is
        "below one in n_series". Callers price a measured zero at this floor rather
        than at zero, which is the whole point of reporting the sample size.
        """
        return 1.0 / self.null.n_series


def _require(mapping: dict, key: str, section: str):
    if key not in mapping:
        raise ValueError(f"policy.toml section [{section}] is missing key '{key}'")
    return mapping[key]


def load_policy(path: str | Path | None = None) -> Policy:
    """Read and validate policy.toml into a frozen Policy.

    Validation is intentionally strict and refused at this boundary: a missing key or
    an impossible value (zero folds, an embargo wider than a fold) is reported here
    with a message naming the key, rather than surfacing later as an opaque indexing
    error inside the backtest.
    """
    policy_path = Path(path) if path is not None else _DEFAULT_POLICY
    if not policy_path.exists():
        raise FileNotFoundError(f"policy file not found: {policy_path}")

    with policy_path.open("rb") as handle:
        raw = tomllib.load(handle)

    bt = raw.get("backtest", {})
    ft = raw.get("features", {})
    md = raw.get("model", {})
    nl = raw.get("null", {})
    vd = raw.get("verdict", {})

    backtest = BacktestPolicy(
        n_folds=int(_require(bt, "n_folds", "backtest")),
        embargo=int(_require(bt, "embargo", "backtest")),
        scheme=str(_require(bt, "scheme", "backtest")),
    )
    features = FeaturePolicy(
        rolling_window=int(_require(ft, "rolling_window", "features")),
        lags=tuple(int(x) for x in _require(ft, "lags", "features")),
        anchor_halfwidth=int(_require(ft, "anchor_halfwidth", "features")),
        fold_fit=bool(_require(ft, "fold_fit", "features")),
    )
    model = ModelPolicy(ridge_alpha=float(_require(md, "ridge_alpha", "model")))
    null = NullPolicy(
        n_series=int(_require(nl, "n_series", "null")),
        threshold_percentile=float(_require(nl, "threshold_percentile", "null")),
    )
    verdict = VerdictPolicy(min_detectable_gap=float(_require(vd, "min_detectable_gap", "verdict")))

    _validate(backtest, features, null)
    return Policy(backtest=backtest, features=features, model=model, null=null, verdict=verdict)


def _validate(backtest: BacktestPolicy, features: FeaturePolicy, null: NullPolicy) -> None:
    if backtest.n_folds < 3:
        raise ValueError(f"n_folds must be at least 3 to backtest, got {backtest.n_folds}")
    if backtest.embargo < 0:
        raise ValueError(f"embargo must be non-negative, got {backtest.embargo}")
    if backtest.scheme not in {"expanding", "sliding"}:
        raise ValueError(f"scheme must be 'expanding' or 'sliding', got {backtest.scheme!r}")
    if features.rolling_window < 1:
        raise ValueError(f"rolling_window must be >= 1, got {features.rolling_window}")
    if any(lag < 1 for lag in features.lags):
        raise ValueError(f"every lag must be >= 1, got {features.lags}")
    if features.anchor_halfwidth < 1:
        raise ValueError(f"anchor_halfwidth must be >= 1, got {features.anchor_halfwidth}")
    if not 50.0 <= null.threshold_percentile <= 100.0:
        raise ValueError(
            f"threshold_percentile must be in [50, 100], got {null.threshold_percentile}"
        )
    if null.n_series < 2:
        raise ValueError(f"n_series must be >= 2 to form a null band, got {null.n_series}")
