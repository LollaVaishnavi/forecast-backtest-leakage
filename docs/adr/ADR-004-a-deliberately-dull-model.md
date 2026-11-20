# ADR 004: A deliberately dull model

Status: accepted

## Context

The repository needs a forecaster to score. The obvious instinct is to reach for a
gradient-boosted tree or anything that would post an impressive error on a demand series.
That instinct is wrong for this project, and resisting it is the decision.

## Decision

The model is a plain ridge regression on standardised features, with a single
regularisation strength in `policy.toml`. It is held fixed across every scheme and every
experiment. The feature standardiser is fit per fold, which is a legitimate per-fold fit,
not a leak.

## Alternatives considered

- **Gradient-boosted trees.** Lower error, and entirely beside the point. A boosted model
  has its own variance between fits and its own interactions with the split, which would
  confound the one thing being measured: the gap a protocol opens. A dull linear model
  makes a change in error attributable to the protocol rather than to the estimator.
- **Several models, reported together.** Rejected as scope creep. The claim is about the
  evaluation protocol, and it holds for any model whose error a leak can lower. One model
  states that claim without diluting it.

## Consequences

The absolute RMSE values here are unremarkable, around 7 to 13 on a series with a standard
deviation near 40, and that is fine: no absolute error is presented as a headline. The
headline is always a ratio or a difference between two schemes running the same model, so
the model's mediocrity cancels and the protocol's effect is what remains. The boring choice
is the one that makes the measurement clean.
