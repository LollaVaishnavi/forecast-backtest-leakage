"""Fail the build when a document quotes a figure the code no longer produces.

Two checks. The forward check holds every metric to its anchor phrase: the exact wording
the prose must use with the value substituted in, matched after whitespace is collapsed so
a phrase that wraps across a line still matches. The reverse check scans the prose for any
statistic (a decimal, or an integer with a percent sign) that matches neither a measured
metric nor a configured policy value, because such a number is one nothing re-measures.
Code spans, fenced blocks and HTML attributes are excluded from the reverse check, since
an example invocation is allowed to contain an illustrative number.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backtest_leakage.config import load_policy  # noqa: E402

METRICS = ROOT / "docs" / "metrics.json"


def _fmt(value) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value == int(value) else repr(round(value, 4))
    return str(value)


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _strip_for_reverse(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)  # fenced blocks
    text = re.sub(r"`[^`]*`", " ", text)  # inline code
    text = re.sub(r"<[^>]+>", " ", text)  # html tags and their attributes
    return text


def _config_values() -> set[str]:
    policy = load_policy()
    values = {
        policy.backtest.n_folds,
        policy.backtest.embargo,
        policy.features.rolling_window,
        policy.features.anchor_halfwidth,
        policy.model.ridge_alpha,
        policy.null.n_series,
        policy.null.threshold_percentile,
        policy.verdict.min_detectable_gap,
        *policy.features.lags,
    }
    return {_fmt(v) for v in values}


def forward_check(payload: dict, docs: dict[str, str]) -> list[str]:
    """Every metric must appear, via one of its anchors, in a checked document."""
    failures = []
    metrics = payload["metrics"]
    anchors = payload["anchors"]
    normalised = {name: _normalise(text) for name, text in docs.items()}

    for key, value in metrics.items():
        templates = anchors.get(key)
        if not templates:
            failures.append(f"metric '{key}' has no anchor phrase")
            continue
        needles = [_normalise(t.format(_fmt(value))) for t in templates]
        if not any(any(n in doc for doc in normalised.values()) for n in needles):
            failures.append(
                f"metric '{key}'={_fmt(value)} not found via any anchor {templates}"
            )
    return failures


def reverse_check(payload: dict, readme: str) -> list[str]:
    """Flag a statistic in the prose that matches no metric and no policy value."""
    allowed = {_fmt(v) for v in payload["metrics"].values()} | _config_values()
    prose = _strip_for_reverse(readme)
    findings = []
    # A decimal, or an integer directly followed by a percent sign.
    for match in re.finditer(r"(?<![\w.])(\d+\.\d+|\d+(?=%))", prose):
        token = match.group(1)
        if token not in allowed:
            findings.append(f"unguarded statistic '{token}' in README matches no metric")
    return findings


def main() -> int:
    if not METRICS.exists():
        print("check_numbers: docs/metrics.json missing; run collect_metrics", file=sys.stderr)
        return 1
    payload = json.loads(METRICS.read_text())

    docs = {}
    for rel in payload["checked_documents"]:
        path = ROOT / rel
        if not path.exists():
            print(f"check_numbers: checked document missing: {rel}", file=sys.stderr)
            return 1
        docs[rel] = path.read_text(encoding="utf-8")

    failures = forward_check(payload, docs)
    findings = reverse_check(payload, docs["README.md"])

    for f in failures:
        print(f"FAIL (forward): {f}", file=sys.stderr)
    for f in findings:
        print(f"FAIL (reverse): {f}", file=sys.stderr)

    if failures or findings:
        print(f"\ncheck_numbers: {len(failures)} forward, {len(findings)} reverse failures",
              file=sys.stderr)
        return 1
    print(f"check_numbers: all {len(payload['metrics'])} metrics anchored, no stray statistics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
