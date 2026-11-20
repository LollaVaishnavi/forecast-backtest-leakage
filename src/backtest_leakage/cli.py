"""Command line entry point: evaluate one series and return a verdict by exit code.

The exit code is the product, not the printout: a caller in a CI pipeline reads it to
decide whether a model may ship. The codes separate the two answers a naive gate
conflates. Zero means the tool ran and found no leak it could resolve; three means it
found a leak; two means it could not rule a leak out at this fold count, which is a
different thing from a clean bill of health; one means the tool itself failed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_policy
from .gap import (
    EXIT_FOR_VERDICT,
    calibrate_null,
    classify,
    evaluate_series,
)
from .generator import SeriesParams, make_series
from .report import build_report_html

EXIT_USAGE = 1


class _Parser(argparse.ArgumentParser):
    """An argument parser that exits with the usage code, not argparse's default 2.

    The default would collide with the inconclusive verdict's exit code, so a caller
    reading the code could not tell a bad flag from an honest "cannot say". Remapping it
    keeps every exit code meaning one thing.
    """

    def error(self, message: str):
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def _build_parser() -> _Parser:
    parser = _Parser(prog="btleak", description="Measure the leakage gap a k-fold score hides.")
    parser.add_argument("--seed", type=int, default=0, help="seed of the fixture series")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--leaky", dest="leaky", action="store_true",
                       help="include the leaky level-anchor feature (default)")
    group.add_argument("--clean", dest="leaky", action="store_false",
                       help="strictly causal features only")
    parser.set_defaults(leaky=True)
    parser.add_argument("--policy", type=str, default=None, help="path to a policy.toml")
    parser.add_argument("--html", type=str, default=None, help="write the HTML report here")
    parser.add_argument("--json", type=str, default=None, help="write machine-readable output here")
    return parser


def _result_payload(seed, pipeline, evaluation, null, verdict) -> dict:
    return {
        "seed": seed,
        "pipeline": pipeline,
        "schemes": {
            "shuffled_kfold_rmse": round(evaluation.kfold.rmse, 4),
            "walk_forward_rmse": round(evaluation.walk_forward.rmse, 4),
            "walk_forward_global_rmse": round(evaluation.walk_forward_global.rmse, 4),
            "blocked_kfold_rmse": round(evaluation.blocked_kfold.rmse, 4),
        },
        "gap": round(evaluation.gap, 4),
        "feature_fit_gap": round(evaluation.feature_fit_gap, 4),
        "membership_gap": round(evaluation.membership_gap, 4),
        "null_threshold": round(null.threshold, 4),
        "null_median": round(null.median, 4),
        "band_margin": round(null.band_margin, 4),
        "verdict": verdict,
        "exit_code": EXIT_FOR_VERDICT[verdict],
    }


def _format_text(payload: dict) -> str:
    s = payload["schemes"]
    lines = [
        f"backtest-leakage: seed={payload['seed']} pipeline={payload['pipeline']}",
        "",
        f"  shuffled k-fold (naive)        RMSE {s['shuffled_kfold_rmse']:.2f}",
        f"  walk-forward, fold-fit (honest) RMSE {s['walk_forward_rmse']:.2f}",
        f"  walk-forward, global features   RMSE {s['walk_forward_global_rmse']:.2f}",
        f"  blocked k-fold (future in train) RMSE {s['blocked_kfold_rmse']:.2f}",
        "",
        f"  leakage gap (honest - kfold):   {payload['gap']:.2f}",
        f"  null threshold (95th pct clean): {payload['null_threshold']:.2f}"
        f"  (median {payload['null_median']:.2f}, margin {payload['band_margin']:.2f})",
        "",
        f"  verdict: {payload['verdict'].upper()}  (exit {payload['exit_code']})",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, evaluate the series, write any outputs, return the exit code."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        policy = load_policy(args.policy)
    except (FileNotFoundError, ValueError) as exc:
        print(f"btleak: policy error: {exc}", file=sys.stderr)
        return EXIT_USAGE

    params = SeriesParams()
    pipeline = "leaky" if args.leaky else "clean"

    y = make_series(args.seed, params)
    evaluation = evaluate_series(y, policy, leaky=args.leaky)
    null = calibrate_null(policy, params)
    verdict = classify(evaluation.gap, null, policy)
    payload = _result_payload(args.seed, pipeline, evaluation, null, verdict)

    print(_format_text(payload))

    if args.html:
        html_text = build_report_html(
            evaluation, null, verdict, seed=args.seed, pipeline=pipeline, policy=policy
        )
        Path(args.html).write_text(html_text, encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return EXIT_FOR_VERDICT[verdict]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
