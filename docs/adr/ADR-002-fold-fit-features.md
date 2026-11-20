# ADR 002: Features are fit per fold, and the one leaky feature is explicit

Status: accepted

## Context

Look-ahead leakage has two doors, and ordering the split only closes one of them. The
first door is the split itself: future rows in the training set. The second is the
feature pipeline: a transform fit on the whole series before the split is made, such as a
scaler whose mean and standard deviation are computed over all rows, or a target encoding
built from the full history. A transform fit that way leaks the future into every row it
touches, and it leaks even under a perfectly time-ordered split, because the leak is baked
into the feature value, not into which rows are in training.

## Decision

Every transform is fit per fold by default: its statistics come from the fold's training
rows and are only applied to the test rows. The pipeline carries exactly one feature that
can leak, the level anchor, and it is named and isolated so the leak can be measured
rather than hidden. The anchor at a row is the mean of the target over the available rows
nearest that row in time, where "available" is whatever set the protocol permits a fit to
read. Under a time-ordered, fold-fit protocol that set is the past, so the anchor is a
trailing mean; under a shuffled split or a global fit it straddles the row, so the same
feature sits on the level and leaks it. Running the pipeline with fold-fit on and off, both
under the time-ordered split, measures the feature-fit channel on its own.

## Alternatives considered

- **Fit transforms once on the whole series.** This is the common real-world mistake, so
  it is kept as a switch (`fold_fit = false`) to measure, not as a default.
- **A library pipeline that fits inside cross-validation automatically.** Rejected for the
  fixture: the point is to make the leak visible and switchable, which a black-box pipeline
  hides. The technique transfers directly to such a pipeline.

## Consequences

Fold-fit features cost a refit per fold, which is cheap here and worth it. The decision
draws a sharp line the README's boundary section states plainly: the tool measures leakage
that lives in the protocol (the split, or fitting on the eval set). It cannot catch a
feature that is non-causal by construction and computed identically in train and test, for
example a centred rolling mean that contains the row's own future, because that leak is in
the feature's definition, not in the split. That is a feature-engineering bug this tool is
not built to find.
