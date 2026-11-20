"""Properties of the HTML report: every verdict renders, every class has a rule."""

from __future__ import annotations

import re

from backtest_leakage.config import load_policy
from backtest_leakage.gap import NullBand, evaluate_series
from backtest_leakage.generator import make_series
from backtest_leakage.report import build_report_html


def _evaluation():
    policy = load_policy()
    y = make_series(2)
    return evaluate_series(y, policy, leaky=True), policy


def test_every_css_class_used_in_the_markup_has_a_rule():
    # The collision that renders a verdict invisible is a class used without a rule, so
    # assert there is a rule for every class the markup emits.
    evaluation, policy = _evaluation()
    null = NullBand(threshold=3.0, median=0.5, gaps=(0.1, 0.5))
    html = build_report_html(evaluation, null, "leak", seed=2, pipeline="leaky", policy=policy)
    used = set(re.findall(r'class="([^"]+)"', html))
    classes = set()
    for group in used:
        classes.update(group.split())
    for cls in classes:
        assert f".{cls}" in html, f"class {cls} has no stylesheet rule"


def test_each_verdict_has_its_own_badge_and_wording():
    evaluation, policy = _evaluation()
    null = NullBand(threshold=3.0, median=0.5, gaps=(0.1, 0.5))
    seen = set()
    for verdict, text in [
        ("leak", "LEAK DETECTED"),
        ("clean", "NO LEAK DETECTED"),
        ("inconclusive", "INCONCLUSIVE"),
    ]:
        html = build_report_html(
            evaluation, null, verdict, seed=2, pipeline="leaky", policy=policy
        )
        assert f"badge-{verdict}" in html
        assert text in html
        seen.add(text)
    assert len(seen) == 3


def test_the_report_states_whether_the_gap_cleared_the_threshold():
    evaluation, policy = _evaluation()
    high = NullBand(threshold=100.0, median=0.5, gaps=(0.1,))
    low = NullBand(threshold=0.1, median=0.05, gaps=(0.01,))
    assert "within the null threshold" in build_report_html(
        evaluation, high, "inconclusive", seed=2, pipeline="leaky", policy=policy
    )
    assert "above the null threshold" in build_report_html(
        evaluation, low, "leak", seed=2, pipeline="leaky", policy=policy
    )
