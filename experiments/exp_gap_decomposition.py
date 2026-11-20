"""Which leak channel does the gap come from: fitting on all rows, or future in training?

The gap has two sources. Fitting a transform on every row (feature-fit leakage) survives
even a time-ordered split; letting future blocks into training (membership leakage) is
what a shuffled split adds. This experiment splits the leaked gap into the two and
reports each as a share, because the fix differs: one is fixed by fitting per fold, the
other only by ordering the split.
"""

from __future__ import annotations

import numpy as np

from backtest_leakage.config import load_policy
from backtest_leakage.gap import evaluate_series
from backtest_leakage.generator import SeriesParams, make_series
from harness import print_table, write_result


def run(policy) -> dict:
    params = SeriesParams()
    feature_fit, membership = [], []
    for s in range(policy.null.n_series):
        y = make_series(s, params)
        leaked = evaluate_series(y, policy, leaky=True)
        feature_fit.append(leaked.feature_fit_gap)
        membership.append(leaked.membership_gap)

    m_feature = float(np.mean(feature_fit))
    m_member = float(np.mean(membership))
    total = m_feature + m_member
    feature_share = round(100.0 * m_feature / total)
    membership_share = round(100.0 * m_member / total)

    rows = [
        {"channel": "feature-fit (survives time order)", "mean_rmse_gap": round(m_feature, 2),
         "share_pct": feature_share},
        {"channel": "membership (future in training)", "mean_rmse_gap": round(m_member, 2),
         "share_pct": membership_share},
    ]
    metrics = {
        "feature_fit_gap_mean": round(m_feature, 2),
        "membership_gap_mean": round(m_member, 2),
        "feature_fit_share_pct": feature_share,
        "membership_share_pct": membership_share,
    }
    return {"name": "gap_decomposition", "metrics": metrics, "rows": rows}


if __name__ == "__main__":
    result = run(load_policy())
    write_result(result["name"], result)
    print_table("gap decomposition (mean over the corpus)", result["rows"])
    print("\nmetrics:", result["metrics"])
