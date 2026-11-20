# ADR 003: The gap is calibrated against a non-stationarity null

Status: accepted

## Context

The leakage gap is one number: the honest walk-forward error minus the k-fold estimate of
the same pipeline. A leak makes k-fold optimistic, so the gap opens. The temptation is to
threshold it directly: a gap above zero means a leak. That is wrong, and the reason is the
whole difficulty of the problem. A non-stationary series opens a gap with no leak at all,
because a time-ordered backtest trains on less and meets regime shifts the past cannot
anticipate, so its error is genuinely higher than a shuffled split would report. On the
fixture here the gap of a clean, leak-free pipeline reaches as high as 11.8 RMSE on an
unlucky series, which is larger than the typical gap of an actual leak. A fixed threshold
would call that clean series leaky.

## Decision

The gap is calibrated against a null distribution estimated from clean series drawn from
the same process. The threshold is the 95th percentile of the clean gap. A gap above it is
a leak. The false-alarm rate is measured on a block of clean series disjoint from the ones
that set the threshold, so the threshold is never scored on the series that defined it.

Below the threshold, the verdict is not automatically clean. The band margin, the distance
from the typical clean gap up to the threshold, is the smallest leak the backtest can
resolve. If that margin is wider than the smallest leak worth catching
(`min_detectable_gap`), then a leak that small would sit under the threshold undetected, so
the tool returns "inconclusive" rather than certifying the pipeline clean. Absence of
evidence is reported as absence of evidence. Three exit codes carry the distinction: 0
clean, 3 leak, 2 inconclusive.

## The war story

The first anchor used a fixed calendar window and fell back to the mean of all available
rows whenever a wide test block left a row with no in-window past. On seed 0 that made the
honest walk-forward error of the leaked pipeline come out at 26.2 RMSE, roughly triple the
clean pipeline's honest error. The gap looked enormous, but for the wrong reason: the
feature had collapsed into a global average on the deep rows of each block, so the number
measured a degenerate feature, not a leak, and would not have survived review. The fix was
to anchor on the nearest available rows rather than a fixed window, which keeps the feature
a proper trailing mean under a time-ordered split. The honest error dropped to about 11.4
RMSE and the gap became an honest measure of the leak. A regression test
(`test_the_honest_leaky_error_stays_in_a_believable_range`) fails if the degeneracy returns.

## Consequences

The tool is deliberately asymmetric. It flags leaks reliably, but on the shipped ten folds
its band margin stays several RMSE wide, so it will not certify a clean pipeline against a
small leak. The fold-power experiment shows the margin never drops below 0.9 RMSE anywhere
from ten up to 20 folds, which is still 18 times the smallest leak worth catching, so clean
certification is out of reach at every fold count tried. This is the honest boundary, not a
defect: a short, serially correlated backtest simply does not have the power to prove a
negative, and the tool says so instead of pretending otherwise.
