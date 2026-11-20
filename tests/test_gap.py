"""Properties of the gap, the null calibration, and the three-way verdict.

The headline tests pin the claim the repository is built on: a shuffled k-fold score
cannot separate a leaked pipeline from a clean one, while the ordered backtest can.
"""

from __future__ import annotations

import pytest

from backtest_leakage.config import load_policy
from backtest_leakage.gap import (
    VERDICT_CLEAN,
    VERDICT_INCONCLUSIVE,
    VERDICT_LEAK,
    NullBand,
    OperatingCharacteristics,
    calibrate_null,
    classify,
    evaluate_series,
    measure_operating_characteristics,
    series_gap,
)
from backtest_leakage.generator import SeriesParams, make_series


def test_kfold_cannot_tell_the_leaked_pipeline_from_a_clean_one(policy):
    # The impossibility, measured: the k-fold score of the leaked pipeline is no worse
    # (in fact a touch better) than the k-fold score of the clean pipeline, so k-fold
    # ranks the leak as fine. Meanwhile its honest error is far higher.
    y = make_series(2)
    clean = evaluate_series(y, policy, leaky=False)
    leaked = evaluate_series(y, policy, leaky=True)
    assert leaked.kfold.rmse <= clean.kfold.rmse + 0.5
    assert leaked.walk_forward.rmse > leaked.kfold.rmse


def test_the_ordered_backtest_opens_a_far_larger_gap_on_the_leak(policy):
    y = make_series(2)
    clean = evaluate_series(y, policy, leaky=False)
    leaked = evaluate_series(y, policy, leaky=True)
    assert leaked.gap > clean.gap
    assert leaked.gap > 2.0
    assert clean.gap < 3.0


def test_the_gap_decomposes_into_feature_fit_and_membership_channels(policy):
    y = make_series(4)
    leaked = evaluate_series(y, policy, leaky=True)
    # Both channels are present and positive for this leak: fitting on all rows and
    # letting the future into training each contribute optimism.
    assert leaked.feature_fit_gap > 0
    assert leaked.membership_gap > 0


def test_series_gap_matches_the_evaluation_gap(policy):
    params = SeriesParams()
    y = make_series(5, params)
    assert series_gap(5, policy, params, leaky=True) == pytest.approx(
        evaluate_series(y, policy, leaky=True).gap
    )


def test_the_null_threshold_sits_at_or_above_the_clean_median(policy, small_params):
    null = calibrate_null(policy, small_params, seeds=range(0, 10))
    assert null.threshold >= null.median
    assert null.band_margin >= 0


def test_a_gap_above_the_threshold_is_a_leak(policy):
    null = NullBand(threshold=3.0, median=0.5, gaps=(0.1, 0.5, 1.0))
    assert classify(5.0, null, policy) == VERDICT_LEAK


def test_a_small_gap_under_a_wide_band_is_inconclusive_not_clean(policy):
    null = NullBand(threshold=3.0, median=0.5, gaps=(0.1, 0.5, 1.0))
    # Band margin 2.5 is far wider than the smallest leak worth catching (0.05), so a
    # below-threshold gap cannot certify the pipeline clean.
    assert classify(1.0, null, policy) == VERDICT_INCONCLUSIVE


def test_a_small_gap_under_a_tight_band_is_clean():
    policy = load_policy()
    narrow = NullBand(threshold=0.5, median=0.49, gaps=(0.4, 0.49, 0.5))
    assert narrow.band_margin < policy.verdict.min_detectable_gap
    assert classify(0.2, narrow, policy) == VERDICT_CLEAN


def test_operating_characteristics_separate_leak_from_clean(policy):
    # On the full-length series the detector catches most leaks while rarely crying
    # wolf. The short fixture used elsewhere is deliberately too small for this, which
    # is itself the point: detection power depends on how many folds the series affords.
    params = SeriesParams()
    null = calibrate_null(policy, params, seeds=range(0, 15))
    oc = measure_operating_characteristics(
        policy, params, null, clean_eval=range(15, 30), leak_eval=range(30, 45)
    )
    assert oc.detection_rate > oc.false_alarm_rate
    assert oc.detection_rate >= 0.8


def test_a_measured_zero_false_alarm_is_priced_at_the_resolution_floor():
    oc = OperatingCharacteristics(
        false_alarm_rate=0.0, detection_rate=1.0, n_clean_eval=40, n_leak_eval=40
    )
    assert oc.priced_false_alarm() == pytest.approx(1.0 / 40)


def test_a_nonzero_false_alarm_is_reported_as_measured():
    oc = OperatingCharacteristics(
        false_alarm_rate=0.1, detection_rate=1.0, n_clean_eval=40, n_leak_eval=40
    )
    assert oc.priced_false_alarm() == pytest.approx(0.1)


def test_an_empty_eval_block_floors_the_rates_without_dividing_by_zero():
    oc = OperatingCharacteristics(
        false_alarm_rate=0.0, detection_rate=0.0, n_clean_eval=0, n_leak_eval=0
    )
    assert oc.false_alarm_floor == 1.0


def test_measuring_on_empty_ranges_returns_zero_rates(policy, small_params):
    null = calibrate_null(policy, small_params, seeds=range(0, 6))
    oc = measure_operating_characteristics(
        policy, small_params, null, clean_eval=range(0, 0), leak_eval=range(0, 0)
    )
    assert oc.false_alarm_rate == 0.0
    assert oc.detection_rate == 0.0
