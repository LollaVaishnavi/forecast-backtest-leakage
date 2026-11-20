"""What does the backtest cost in power, and can it ever certify a pipeline clean?

A time-ordered backtest spends folds: each one trains on less and tests on a serially
correlated slice, so the null band around the gap is wide and only a leak larger than
that band clears the threshold. This experiment sweeps the fold count and reports the
band margin, which is the smallest leak the backtest can resolve. The margin wanders with
the fold count rather than shrinking cleanly, because it is a percentile spread of a
small sample, but across the whole swept range it never comes close to the smallest leak
worth catching. So the asymmetry holds at every fold count tried: the detector flags
leaks reliably, yet a below-threshold gap can never certify a clean bill of health.
"""

from __future__ import annotations

from dataclasses import replace

from backtest_leakage.config import load_policy
from backtest_leakage.gap import calibrate_null, measure_operating_characteristics
from backtest_leakage.generator import SeriesParams
from harness import print_table, write_result

FOLD_SWEEP = (5, 8, 10, 15, 20)
SEEDS_PER_SETTING = 20


def _with_folds(policy, n_folds: int):
    return replace(policy, backtest=replace(policy.backtest, n_folds=n_folds))


def run(policy) -> dict:
    params = SeriesParams()
    rows = []
    margin_by_fold = {}
    for nf in FOLD_SWEEP:
        p = _with_folds(policy, nf)
        n = SEEDS_PER_SETTING
        null = calibrate_null(p, params, seeds=range(0, n))
        oc = measure_operating_characteristics(
            p, params, null, clean_eval=range(n, 2 * n), leak_eval=range(2 * n, 3 * n)
        )
        margin_by_fold[nf] = null.band_margin
        rows.append({
            "n_folds": nf,
            "band_margin": round(null.band_margin, 2),
            "detection_rate": round(oc.detection_rate, 2),
            "false_alarm": round(oc.priced_false_alarm(), 3),
        })

    shipped = policy.backtest.n_folds
    min_detectable = policy.verdict.min_detectable_gap
    smallest_margin = min(margin_by_fold.values())
    # Can any swept fold count certify clean, i.e. push the margin below the smallest
    # leak worth catching?
    certifiable = any(m <= min_detectable for m in margin_by_fold.values())
    # Even at its narrowest, how many times the smallest leak worth catching is the band?
    margin_to_detectable_ratio = round(smallest_margin / min_detectable)

    metrics = {
        "shipped_n_folds": shipped,
        "min_detectable_gap": min_detectable,
        "smallest_band_margin_in_sweep": round(smallest_margin, 2),
        "margin_to_detectable_ratio": margin_to_detectable_ratio,
        "folds_swept_low": min(FOLD_SWEEP),
        "folds_swept_high": max(FOLD_SWEEP),
        "clean_certifiable_in_range": certifiable,
    }
    return {"name": "fold_power", "metrics": metrics, "rows": rows}


if __name__ == "__main__":
    result = run(load_policy())
    write_result(result["name"], result)
    print_table("fold power sweep (band margin is the smallest resolvable leak)", result["rows"])
    print("\nmetrics:", result["metrics"])
