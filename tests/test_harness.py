"""Properties of the scoring harness: pooling, spread, and empty-split refusal."""

from __future__ import annotations

import numpy as np
import pytest

from backtest_leakage.features import usable_rows
from backtest_leakage.generator import make_series
from backtest_leakage.harness import SchemeResult, run_scheme
from backtest_leakage.splitters import shuffled_kfold_splits, walk_forward_splits


def test_a_single_fold_estimate_has_no_spread():
    result = SchemeResult(rmse=3.0, per_fold_rmse=(3.0,), n_test_rows=10)
    assert result.fold_rmse_std == 0.0


def test_the_spread_is_the_sample_std_of_the_per_fold_errors():
    result = SchemeResult(rmse=0.0, per_fold_rmse=(2.0, 4.0), n_test_rows=20)
    assert result.fold_rmse_std == pytest.approx(np.std([2.0, 4.0], ddof=1))


def test_running_a_scheme_pools_residuals_across_folds(policy, small_params):
    y = make_series(0, small_params)
    usable = usable_rows(y.size, policy.features)
    splits = walk_forward_splits(usable, policy.backtest.n_folds, policy.backtest.embargo,
                                 policy.backtest.scheme)
    result = run_scheme(y, usable, splits, policy.features,
                        leaky=False, fold_fit=True, ridge_alpha=policy.model.ridge_alpha)
    assert result.rmse > 0
    assert result.n_test_rows == sum(test.size for _, test in splits)
    assert len(result.per_fold_rmse) == len(splits)


def test_fold_fit_and_global_differ_only_through_the_leaky_feature(policy, small_params):
    # With no leaky feature, fitting transforms on all rows versus training rows alone
    # must give the same error, because nothing in the causal pipeline reads the fit set.
    y = make_series(0, small_params)
    usable = usable_rows(y.size, policy.features)
    splits = shuffled_kfold_splits(usable, policy.backtest.n_folds, seed=y.size)
    fold_fit = run_scheme(y, usable, splits, policy.features,
                          leaky=False, fold_fit=True, ridge_alpha=1.0)
    global_fit = run_scheme(y, usable, splits, policy.features,
                            leaky=False, fold_fit=False, ridge_alpha=1.0)
    assert fold_fit.rmse == pytest.approx(global_fit.rmse)


def test_scoring_no_test_rows_is_refused(policy, small_params):
    y = make_series(0, small_params)
    usable = usable_rows(y.size, policy.features)
    with pytest.raises(ValueError, match="no test rows"):
        run_scheme(y, usable, [], policy.features,
                   leaky=False, fold_fit=True, ridge_alpha=1.0)
