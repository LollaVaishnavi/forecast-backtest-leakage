"""Run one pipeline through one set of splits and return its error.

A pipeline here is a pair of switches: whether the leaky level anchor is included, and
whether fitted transforms may read the whole series (global) or only each fold's
training rows (fold-fit). The harness holds the model fixed, a plain ridge on
standardised features, so that any change in error between two runs is attributable to
the split and the switches rather than to the estimator. That is the discipline that
lets a gap be called a leak. See ADR 004 for why the model is deliberately dull.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .config import FeaturePolicy
from .features import build_matrix
from .splitters import Split


@dataclass(frozen=True)
class SchemeResult:
    """The error of a pipeline under a scheme, pooled and per fold."""

    rmse: float
    per_fold_rmse: tuple[float, ...]
    n_test_rows: int

    @property
    def fold_rmse_std(self) -> float:
        """Spread of per-fold RMSE, the honest measure of how noisy this estimate is."""
        if len(self.per_fold_rmse) < 2:
            return 0.0
        return float(np.std(self.per_fold_rmse, ddof=1))


def _rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    residual = actual - predicted
    return float(np.sqrt(np.mean(residual**2)))


def run_scheme(
    y: np.ndarray,
    usable: np.ndarray,
    splits: list[Split],
    fp: FeaturePolicy,
    *,
    leaky: bool,
    fold_fit: bool,
    ridge_alpha: float,
) -> SchemeResult:
    """Fit and score the pipeline on every split, then pool the residuals.

    ``fold_fit`` decides what the level anchor may read. When true, the available set is
    the fold's own training rows, so under a time-ordered split the anchor sees only the
    past. When false, the available set is every usable row, so the anchor reads the
    whole series even when the split itself respects time. The pooled RMSE weights each
    fold by its size, and the per-fold RMSE is kept so the estimate carries its own
    spread.
    """
    all_usable = np.asarray(usable, dtype=np.int64)
    residual_sq_sum = 0.0
    residual_count = 0
    per_fold: list[float] = []

    for train_idx, test_idx in splits:
        available = train_idx if fold_fit else all_usable

        x_train = build_matrix(y, train_idx, available, fp, leaky)
        x_test = build_matrix(y, test_idx, available, fp, leaky)
        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)

        model = Ridge(alpha=ridge_alpha, solver="cholesky")
        model.fit(x_train_s, y_train)
        predicted = model.predict(x_test_s)

        per_fold.append(_rmse(y_test, predicted))
        residual = y_test - predicted
        residual_sq_sum += float(np.sum(residual**2))
        residual_count += residual.size

    if residual_count == 0:
        raise ValueError(
            "no test rows were scored; check n_folds and embargo against the series length"
        )

    pooled = float(np.sqrt(residual_sq_sum / residual_count))
    return SchemeResult(rmse=pooled, per_fold_rmse=tuple(per_fold), n_test_rows=residual_count)
