"""Shared plumbing for the experiments.

Each experiment is a function that takes a policy and returns a dict of named metrics
plus a small table of rows. The same function backs both the standalone script, which
writes the JSON and prints the table, and the metrics collector, which re-measures every
figure the documents quote. Keeping one implementation means a figure in the README and
the figure printed by its experiment can never drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "experiments"


def write_result(name: str, result: dict) -> Path:
    """Write an experiment result to docs/experiments/<name>.json and return the path."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def print_table(title: str, rows: list[dict]) -> None:
    """Print a plain aligned table, the thing a reader sees when running the experiment."""
    print(title)
    if not rows:
        print("  (no rows)")
        return
    headers = list(rows[0].keys())
    widths = {h: max(len(h), *(len(str(r[h])) for r in rows)) for h in headers}
    line = "  ".join(h.ljust(widths[h]) for h in headers)
    print("  " + line)
    for r in rows:
        print("  " + "  ".join(str(r[h]).ljust(widths[h]) for h in headers))
