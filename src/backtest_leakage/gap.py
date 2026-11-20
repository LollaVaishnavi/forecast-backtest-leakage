"""The leakage gap, its null calibration, and the three-way verdict.

The gap is one number: how much higher the honest walk-forward error is than the
k-fold estimate the same pipeline would report. A leak makes the k-fold estimate
optimistic, so the gap opens. The difficulty, and the reason this is not a one-line
threshold, is that a non-stationary series opens a gap even with no leak at all, purely
because a time-ordered backtest trains on less and faces regime shifts the past cannot
see. So the gap is calibrated against a null distribution estimated from clean series,
and a verdict of "clean" is only returned when the backtest had the power to have found
a leak worth caring about. See ADR 003.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Policy
from .features import usable_rows
from .generator import SeriesParams, make_series
from .harness import SchemeResult, run_scheme
from .splitters import (
    blocked_kfold_splits,
    shuffled_kfold_splits,
    walk_forward_splits,
)

# Exit codes, surfaced by the CLI. A leak-present answer and a could-not-answer are
# deliberately distinct, so a caller can tell "the tool found a leak" from "the tool
# could not rule one out".
VERDICT_CLEAN = "clean"
VERDICT_LEAK = "leak"
VERDICT_INCONCLUSIVE = "inconclusive"

EXIT_FOR_VERDICT = {VERDICT_CLEAN: 0, VERDICT_INCONCLUSIVE: 2, VERDICT_LEAK: 3}


@dataclass(frozen=True)
class SeriesEvaluation:
    """Every scheme's error for one series, plus the headline gap."""

    kfold: SchemeResult
    walk_forward: SchemeResult
    walk_forward_global: SchemeResult
    blocked_kfold: SchemeResult

    @property
    def gap(self) -> float:
        """Honest error minus the k-fold estimate: the optimism k-fold hides."""
        return self.walk_forward.rmse - self.kfold.rmse

    @property
    def feature_fit_gap(self) -> float:
        """Extra optimism from fitting transforms on all rows under a time-ordered split."""
        return self.walk_forward.rmse - self.walk_forward_global.rmse

    @property
    def membership_gap(self) -> float:
        """Optimism from letting future blocks into training, test rows held contiguous."""
        return self.walk_forward.rmse - self.blocked_kfold.rmse


def evaluate_series(y: np.ndarray, policy: Policy, *, leaky: bool) -> SeriesEvaluation:
    """Run every scheme on one series and return their errors.

    The k-fold scheme uses a seed derived from the series length so its permutation is
    reproducible without a free parameter. All schemes share the fold count and the
    feature policy, so the only thing that varies is how the rows are assigned to train
    and test and whether transforms may read the future.
    """
    usable = usable_rows(y.size, policy.features)
    nf = policy.backtest.n_folds
    alpha = policy.model.ridge_alpha

    wf_splits = walk_forward_splits(usable, nf, policy.backtest.embargo, policy.backtest.scheme)
    kf_splits = shuffled_kfold_splits(usable, nf, seed=y.size)
    bk_splits = blocked_kfold_splits(usable, nf)

    kfold = run_scheme(y, usable, kf_splits, policy.features,
                       leaky=leaky, fold_fit=True, ridge_alpha=alpha)
    walk_forward = run_scheme(y, usable, wf_splits, policy.features,
                              leaky=leaky, fold_fit=True, ridge_alpha=alpha)
    walk_forward_global = run_scheme(y, usable, wf_splits, policy.features,
                                     leaky=leaky, fold_fit=False, ridge_alpha=alpha)
    blocked = run_scheme(y, usable, bk_splits, policy.features,
                         leaky=leaky, fold_fit=True, ridge_alpha=alpha)

    return SeriesEvaluation(
        kfold=kfold,
        walk_forward=walk_forward,
        walk_forward_global=walk_forward_global,
        blocked_kfold=blocked,
    )


def series_gap(seed: int, policy: Policy, params: SeriesParams, *, leaky: bool) -> float:
    """The gap for one generated series, the unit of the null and detection sets.

    Only the two schemes the gap is made of are run here, not the full four-scheme
    decomposition, because the null and detection sets call this hundreds of times and
    the extra two schemes would double that cost for no new information.
    """
    y = make_series(seed, params)
    usable = usable_rows(y.size, policy.features)
    nf = policy.backtest.n_folds
    alpha = policy.model.ridge_alpha

    wf_splits = walk_forward_splits(usable, nf, policy.backtest.embargo, policy.backtest.scheme)
    kf_splits = shuffled_kfold_splits(usable, nf, seed=y.size)

    kfold = run_scheme(y, usable, kf_splits, policy.features,
                       leaky=leaky, fold_fit=True, ridge_alpha=alpha)
    walk_forward = run_scheme(y, usable, wf_splits, policy.features,
                              leaky=leaky, fold_fit=True, ridge_alpha=alpha)
    return walk_forward.rmse - kfold.rmse


@dataclass(frozen=True)
class NullBand:
    """The clean-gap distribution and the threshold drawn from it."""

    threshold: float
    median: float
    gaps: tuple[float, ...]

    @property
    def band_margin(self) -> float:
        """How far above the typical clean gap a gap must sit to be flagged."""
        return self.threshold - self.median


def calibrate_null(
    policy: Policy, params: SeriesParams, seeds: range | None = None
) -> NullBand:
    """Estimate the clean-gap band from a block of clean (leak-free) series.

    The threshold is the configured percentile of the clean gaps. Because a clean series
    still opens a gap on a non-stationary process, this band is what separates a leak
    from a drift. The seeds used here are the calibration set; the false-alarm rate is
    measured on a disjoint block of clean seeds, so the threshold is never scored on the
    same series that set it.
    """
    span = seeds if seeds is not None else range(policy.null.n_series)
    gaps = np.array([series_gap(s, policy, params, leaky=False) for s in span])
    threshold = float(np.percentile(gaps, policy.null.threshold_percentile))
    median = float(np.median(gaps))
    return NullBand(threshold=threshold, median=median, gaps=tuple(float(g) for g in gaps))


def classify(gap: float, null: NullBand, policy: Policy) -> str:
    """Turn a gap and a null band into clean, leak, or inconclusive.

    A gap above the threshold is a leak. Below the threshold, the verdict depends on
    power: if the band is wider than the smallest leak worth catching, then a leak that
    small would have sat under the threshold undetected, so the tool returns
    inconclusive rather than certifying the series clean. Absence of evidence is reported
    as absence of evidence, not as evidence of absence.
    """
    if gap > null.threshold:
        return VERDICT_LEAK
    if null.band_margin > policy.verdict.min_detectable_gap:
        return VERDICT_INCONCLUSIVE
    return VERDICT_CLEAN


@dataclass(frozen=True)
class OperatingCharacteristics:
    """How the detector behaves on held-out clean and leaked series."""

    false_alarm_rate: float
    detection_rate: float
    n_clean_eval: int
    n_leak_eval: int

    @property
    def false_alarm_floor(self) -> float:
        """The smallest false-alarm rate this eval sample can represent: one series."""
        return 1.0 / self.n_clean_eval if self.n_clean_eval else 1.0

    def priced_false_alarm(self) -> float:
        """A measured zero is not zero: price it at the resolution floor."""
        return max(self.false_alarm_rate, self.false_alarm_floor)


def measure_operating_characteristics(
    policy: Policy,
    params: SeriesParams,
    null: NullBand,
    clean_eval: range,
    leak_eval: range,
) -> OperatingCharacteristics:
    """Measure false-alarm and detection rates on seeds disjoint from calibration."""
    clean_flags = [
        classify(series_gap(s, policy, params, leaky=False), null, policy) == VERDICT_LEAK
        for s in clean_eval
    ]
    leak_flags = [
        classify(series_gap(s, policy, params, leaky=True), null, policy) == VERDICT_LEAK
        for s in leak_eval
    ]
    false_alarm = float(np.mean(clean_flags)) if clean_flags else 0.0
    detection = float(np.mean(leak_flags)) if leak_flags else 0.0
    return OperatingCharacteristics(
        false_alarm_rate=false_alarm,
        detection_rate=detection,
        n_clean_eval=len(clean_eval),
        n_leak_eval=len(leak_eval),
    )
