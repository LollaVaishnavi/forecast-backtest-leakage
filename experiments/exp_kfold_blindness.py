"""Can a k-fold score tell a leaked pipeline from a clean one? Measured: no.

For each series we score both pipelines two ways. The k-fold column is what a
practitioner who shuffles would report; the walk-forward column is the honest error. The
experiment reports how close k-fold rates the two pipelines (its blindness) and how far
walk-forward separates them (its sight).
"""

from __future__ import annotations

import numpy as np

from backtest_leakage.config import load_policy
from backtest_leakage.gap import evaluate_series
from backtest_leakage.generator import SeriesParams, make_series
from harness import print_table, write_result


def run(policy) -> dict:
    params = SeriesParams()
    seeds = range(policy.null.n_series)
    kfold_clean, kfold_leak, wf_clean, wf_leak = [], [], [], []
    for s in seeds:
        y = make_series(s, params)
        clean = evaluate_series(y, policy, leaky=False)
        leaked = evaluate_series(y, policy, leaky=True)
        kfold_clean.append(clean.kfold.rmse)
        kfold_leak.append(leaked.kfold.rmse)
        wf_clean.append(clean.walk_forward.rmse)
        wf_leak.append(leaked.walk_forward.rmse)

    m_kfold_clean = float(np.mean(kfold_clean))
    m_kfold_leak = float(np.mean(kfold_leak))
    m_wf_clean = float(np.mean(wf_clean))
    m_wf_leak = float(np.mean(wf_leak))

    # How close k-fold rates the two pipelines, as a percentage of the clean score.
    kfold_separation_pct = round(100.0 * abs(m_kfold_leak - m_kfold_clean) / m_kfold_clean)
    # How much worse the leak honestly is than the clean pipeline.
    wf_degradation_pct = round(100.0 * (m_wf_leak - m_wf_clean) / m_wf_clean)
    # How much k-fold understates the leaked pipeline's true error.
    kfold_understatement_pct = round(100.0 * (m_wf_leak - m_kfold_leak) / m_kfold_leak)
    # Does k-fold actually prefer the leak (rate it at least as good as clean)?
    kfold_prefers_leak = bool(m_kfold_leak <= m_kfold_clean)

    rows = [
        {"pipeline": "clean", "kfold_rmse": round(m_kfold_clean, 2),
         "walkforward_rmse": round(m_wf_clean, 2)},
        {"pipeline": "leaked", "kfold_rmse": round(m_kfold_leak, 2),
         "walkforward_rmse": round(m_wf_leak, 2)},
    ]
    metrics = {
        "n_series": policy.null.n_series,
        "kfold_clean_rmse": round(m_kfold_clean, 2),
        "kfold_leak_rmse": round(m_kfold_leak, 2),
        "wf_clean_rmse": round(m_wf_clean, 2),
        "wf_leak_rmse": round(m_wf_leak, 2),
        "kfold_separation_pct": kfold_separation_pct,
        "wf_degradation_pct": wf_degradation_pct,
        "kfold_understatement_pct": kfold_understatement_pct,
        "kfold_prefers_leak": kfold_prefers_leak,
    }
    return {"name": "kfold_blindness", "metrics": metrics, "rows": rows}


if __name__ == "__main__":
    result = run(load_policy())
    write_result(result["name"], result)
    print_table("k-fold blindness (mean RMSE over the corpus)", result["rows"])
    print("\nmetrics:", result["metrics"])
