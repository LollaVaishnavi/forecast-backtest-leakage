"""Is a gap a leak or a drift? Calibrate the null, then measure the detector.

The threshold is set on one block of clean series and the false-alarm rate is measured
on a disjoint block, so the threshold is never scored on the series that set it. A
measured false-alarm rate of zero is reported at its resolution floor, one over the eval
size, because zero over forty is "below one in forty", not zero.
"""

from __future__ import annotations

import math

from backtest_leakage.config import load_policy
from backtest_leakage.gap import (
    calibrate_null,
    measure_operating_characteristics,
)
from backtest_leakage.generator import SeriesParams
from harness import print_table, write_result


def run(policy) -> dict:
    params = SeriesParams()
    n = policy.null.n_series
    null = calibrate_null(policy, params, seeds=range(0, n))
    oc = measure_operating_characteristics(
        policy, params, null,
        clean_eval=range(n, 2 * n),
        leak_eval=range(2 * n, 3 * n),
    )

    priced = oc.priced_false_alarm()
    # Clean series whose gap alone would exceed the threshold: the confound, in numbers.
    max_clean_gap = max(null.gaps)

    # To show a false-alarm rate below a target with a measured zero, you need at least
    # 1/target samples. State the sample a one-percent claim would require.
    samples_for_one_pct = math.ceil(1.0 / 0.01)

    rows = [
        {"quantity": "null threshold (95th pct clean gap)", "value": round(null.threshold, 2)},
        {"quantity": "median clean gap", "value": round(null.median, 2)},
        {"quantity": "largest clean gap (no leak)", "value": round(max_clean_gap, 2)},
        {"quantity": "detection rate on leaks", "value": round(oc.detection_rate, 3)},
        {"quantity": "false-alarm rate (measured)", "value": round(oc.false_alarm_rate, 3)},
        {"quantity": "false-alarm rate (priced at floor)", "value": round(priced, 3)},
    ]
    metrics = {
        "n_series": n,
        "null_threshold": round(null.threshold, 2),
        "null_median": round(null.median, 2),
        "band_margin": round(null.band_margin, 2),
        "max_clean_gap": round(max_clean_gap, 2),
        "detection_rate": round(oc.detection_rate, 3),
        "detection_pct": round(100 * oc.detection_rate),
        "false_alarm_rate": round(oc.false_alarm_rate, 3),
        "false_alarm_priced": round(priced, 3),
        "false_alarm_floor": round(oc.false_alarm_floor, 3),
        "n_clean_eval": oc.n_clean_eval,
        "samples_for_one_pct_false_alarm": samples_for_one_pct,
    }
    return {"name": "null_calibration", "metrics": metrics, "rows": rows}


if __name__ == "__main__":
    result = run(load_policy())
    write_result(result["name"], result)
    print_table("null calibration and detector operating characteristics", result["rows"])
    print("\nmetrics:", result["metrics"])
