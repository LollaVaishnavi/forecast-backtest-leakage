"""Render a backtest result as a self-contained HTML report.

The report exists so the evidence screenshot photographs something a reviewer would
actually read: the two schemes side by side, the gap between them, and the verdict in a
badge whose colour states the answer. It is a pure function of the result so the same
inputs always produce the same bytes, and every CSS class it emits has a rule in the
stylesheet, which a test asserts so a verdict can never render invisibly.
"""

from __future__ import annotations

import html

from .config import Policy
from .gap import NullBand, SeriesEvaluation

_VERDICT_TEXT = {
    "clean": "NO LEAK DETECTED",
    "leak": "LEAK DETECTED",
    "inconclusive": "INCONCLUSIVE",
}

_STYLE = """
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; color: #1a1a1a;
       background: #ffffff; margin: 0; padding: 32px; }
h1 { font-size: 22px; margin: 0 0 4px 0; }
.sub { color: #555; font-size: 13px; margin: 0 0 20px 0; }
.badge { display: inline-block; padding: 8px 16px; border-radius: 6px; color: #ffffff;
         font-weight: 700; font-size: 15px; letter-spacing: 0.5px; }
.badge-leak { background: #b00020; }
.badge-clean { background: #1b7f3b; }
.badge-inconclusive { background: #b26a00; }
table { border-collapse: collapse; margin: 18px 0; font-size: 14px; }
th, td { border: 1px solid #ccc; padding: 7px 12px; text-align: right; }
th { background: #f2f2f2; text-align: left; }
td.label { text-align: left; }
.honest { background: #eef6ff; }
.gapline { font-size: 16px; margin: 14px 0; }
.num { font-weight: 700; }
.note { color: #555; font-size: 13px; max-width: 640px; line-height: 1.5; }
"""


def _row(label: str, result, highlight: bool) -> str:
    cls = ' class="honest"' if highlight else ""
    label_html = html.escape(label)
    return (
        f"<tr{cls}><td class=\"label\">{label_html}</td>"
        f"<td>{result.rmse:.2f}</td>"
        f"<td>{result.fold_rmse_std:.2f}</td>"
        f"<td>{result.n_test_rows}</td></tr>"
    )


def build_report_html(
    evaluation: SeriesEvaluation,
    null: NullBand,
    verdict: str,
    *,
    seed: int,
    pipeline: str,
    policy: Policy,
) -> str:
    """Return the full HTML document for one series evaluation."""
    badge_text = _VERDICT_TEXT[verdict]
    rows = "".join(
        [
            _row("shuffled k-fold (the naive estimate)", evaluation.kfold, False),
            _row("walk-forward, fold-fit (honest)", evaluation.walk_forward, True),
            _row("walk-forward, global features", evaluation.walk_forward_global, False),
            _row("blocked k-fold (future in training)", evaluation.blocked_kfold, False),
        ]
    )
    gap = evaluation.gap
    comparison = "above" if gap > null.threshold else "within"
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>backtest-leakage report: seed {seed}, {html.escape(pipeline)} pipeline</title>
<style>{_STYLE}</style></head>
<body>
<h1>Leakage gap report</h1>
<p class="sub">seed {seed}, {html.escape(pipeline)} pipeline, {policy.backtest.n_folds} folds,
scheme {html.escape(policy.backtest.scheme)}</p>
<span class="badge badge-{verdict}">{badge_text}</span>
<table>
<thead><tr><th>scheme</th><th>RMSE</th><th>fold RMSE std</th><th>test rows</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<p class="gapline">Leakage gap (honest minus k-fold):
<span class="num">{gap:.2f}</span> RMSE, which is {comparison} the null threshold of
<span class="num">{null.threshold:.2f}</span>.</p>
<p class="note">The null threshold is the {policy.null.threshold_percentile:.0f}th percentile
of the gap measured on {len(null.gaps)} clean series drawn from the same process, so it is
the gap a drift alone can open. The honest row is the only estimate that trains strictly on
the past; the k-fold row is what a shuffled split would have reported.</p>
</body></html>
"""
