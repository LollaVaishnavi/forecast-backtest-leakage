# ADR 001: Walk-forward with an expanding window is the honest estimator

Status: accepted

## Context

A forecaster is scored by cross validation to decide whether it may ship. The default
in most codebases is random k-fold: shuffle the rows, cut into folds, train on the rest,
test on each fold. On a time series that shuffle is the whole problem. A training fold
then contains rows that are later in time than the test rows, so the model is fitted with
access to the future of the very points it is scored on. Any estimate built that way is
an upper bound on how good the model looks, not on how it will do when it only has the
past.

## Decision

The honest estimator is a walk-forward backtest. The usable rows are cut into contiguous,
time-ordered blocks. Each block after the first is a test fold, and its training set is
only the blocks before it in time. An embargo of `embargo` rows is dropped from the end of
training so a trailing-window feature cannot straddle the boundary and carry the last
training rows into the test region. The default window is expanding: train on all past
rows, because a forecaster in production has all its history. A sliding window is offered
for series where distant history comes from a different regime and hurts.

## Alternatives considered

- **Random k-fold.** Rejected as the honest estimator precisely because it is the thing
  this repository exists to expose. It is still computed, as the naive baseline the gap
  is measured against.
- **Blocked k-fold** (contiguous test blocks, train on every other block including future
  ones). Kept, but only as a decomposition tool: comparing it to walk-forward isolates the
  effect of letting the future into training from the effect of how test rows are chosen.
- **Leave-one-out in time.** Rejected: one-row test folds make each fold's error so noisy
  that the per-fold spread swamps the signal, and the cost is linear in the series length.

## Consequences

Walk-forward trains on less data than k-fold, especially in its early folds, and its test
points are serially correlated, so its error estimate is noisier. That cost is real and is
priced in ADR 003: it is why the tool can flag a leak but is slow to certify a clean
pipeline. The benefit is that the estimate does not depend on the future, so the gap
between it and the k-fold estimate is attributable to information the k-fold split leaked.
