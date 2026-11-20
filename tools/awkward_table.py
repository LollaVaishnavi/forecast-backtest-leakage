"""Write the uncomfortable numbers to a markdown table.

In CI this is redirected to the job summary, because that is where a human actually reads
a result, rather than scrolling a log. The table leads with the figures the author would
rather not publish: the leak out-scoring the clean pipeline, and the tool's own inability
to certify a clean bill of health.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "docs" / "metrics.json"


def render() -> str:
    m = json.loads(METRICS.read_text())["metrics"]
    rows = [
        ("k-fold rates the leaked pipeline", f"{m['kfold_leak_rmse']} RMSE"),
        ("k-fold rates the clean pipeline", f"{m['kfold_clean_rmse']} RMSE (worse)"),
        ("leaked pipeline's true error", f"{m['wf_leak_rmse']} RMSE"),
        ("k-fold understates the leak by", f"{m['kfold_understatement_pct']}%"),
        ("largest clean gap with no leak", f"{m['max_clean_gap']} RMSE"),
        ("smallest resolvable leak (band margin)", f"never below {m['smallest_band_margin']} RMSE"),
        ("can it certify a clean pipeline", "no, at every fold count tried"),
    ]
    lines = [
        "## Leakage gap: the numbers that matter",
        "",
        "| quantity | value |",
        "| --- | --- |",
    ]
    lines += [f"| {label} | {value} |" for label, value in rows]
    lines.append("")
    lines.append(
        f"Suite: {m['tests_total']} tests at {m['coverage_line_pct']}% line coverage. "
        "Every figure re-measured by `tools/collect_metrics.py` and guarded by "
        "`tools/check_numbers.py`."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(render())
