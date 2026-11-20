"""Re-measure every published figure and write docs/metrics.json.

Nothing in the documents is typed by hand: this script runs the experiments, reads the
test and coverage reports the CI test step produced, derives the published values at the
exact rounding the prose uses, and records the anchor phrase that guards each one. The
number checker then holds the prose to these values. Run the test suite as its own step
and pass --skip-tests so this reads the reports rather than running pytest inside a
metrics script, which would bury a red test in a collector traceback.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "src"))

import exp_fold_power  # noqa: E402
import exp_gap_decomposition  # noqa: E402
import exp_kfold_blindness  # noqa: E402
import exp_null_calibration  # noqa: E402
from backtest_leakage.config import load_policy  # noqa: E402
from backtest_leakage.gap import evaluate_series  # noqa: E402
from backtest_leakage.generator import SeriesParams, make_series  # noqa: E402

REPORTS = ROOT / "reports"
JUNIT = REPORTS / "junit.xml"
COVERAGE = ROOT / "coverage.json"
METRICS_OUT = ROOT / "docs" / "metrics.json"


def _ensure_reports(skip_tests: bool) -> None:
    if skip_tests:
        if not JUNIT.exists() or not COVERAGE.exists():
            raise SystemExit(
                "--skip-tests needs reports/junit.xml and coverage.json; run the test step first"
            )
        return
    REPORTS.mkdir(exist_ok=True)
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q",
         f"--junitxml={JUNIT}",
         "--cov=backtest_leakage", "--cov-report=json:" + str(COVERAGE)],
        cwd=ROOT, check=True,
    )


def _tests_total() -> int:
    root = ET.parse(JUNIT).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    return int(suite.attrib["tests"])


def _coverage_pct():
    data = json.loads(COVERAGE.read_text())
    pct = data["totals"]["percent_covered"]
    # Report a whole-number coverage as an int so the prose can read "100% line coverage".
    return int(round(pct)) if abs(pct - round(pct)) < 1e-9 else round(pct, 1)


def collect(skip_tests: bool) -> dict:
    _ensure_reports(skip_tests)
    policy = load_policy()

    blind = exp_kfold_blindness.run(policy)["metrics"]
    nullc = exp_null_calibration.run(policy)["metrics"]
    decomp = exp_gap_decomposition.run(policy)["metrics"]
    folds = exp_fold_power.run(policy)["metrics"]

    # The post-fix honest error on the war-story series, machine-checked so the number
    # the README tells about the bug cannot itself go stale.
    seed0_leaky = evaluate_series(make_series(0, SeriesParams()), policy, leaky=True)
    honest_leaky_seed0 = round(seed0_leaky.walk_forward.rmse, 1)

    metrics = {
        "tests_total": _tests_total(),
        "coverage_line_pct": _coverage_pct(),
        "n_series": nullc["n_series"],
        "n_clean_eval": nullc["n_clean_eval"],
        # k-fold blindness
        "kfold_clean_rmse": round(blind["kfold_clean_rmse"], 1),
        "kfold_leak_rmse": round(blind["kfold_leak_rmse"], 1),
        "wf_clean_rmse": round(blind["wf_clean_rmse"], 1),
        "wf_leak_rmse": round(blind["wf_leak_rmse"], 1),
        "kfold_separation_pct": blind["kfold_separation_pct"],
        "wf_degradation_pct": blind["wf_degradation_pct"],
        "kfold_understatement_pct": blind["kfold_understatement_pct"],
        # null calibration
        "null_threshold": round(nullc["null_threshold"], 1),
        "max_clean_gap": round(nullc["max_clean_gap"], 1),
        "detection_pct": nullc["detection_pct"],
        # gap decomposition
        "feature_fit_share_pct": decomp["feature_fit_share_pct"],
        "membership_share_pct": decomp["membership_share_pct"],
        # fold power
        "smallest_band_margin": round(folds["smallest_band_margin_in_sweep"], 1),
        "margin_to_detectable_ratio": folds["margin_to_detectable_ratio"],
        "folds_swept_high": folds["folds_swept_high"],
        "honest_leaky_seed0_rmse": honest_leaky_seed0,
    }

    anchors = {
        "tests_total": ["{} tests"],
        "coverage_line_pct": ["{}% line coverage"],
        "n_series": ["{} clean series", "{}-series"],
        "n_clean_eval": ["one in {}", "{} held-out"],
        "kfold_clean_rmse": ["clean pipeline's {}", "{} on the clean"],
        "kfold_leak_rmse": ["leaked pipeline at {}", "rates it {}"],
        "wf_clean_rmse": ["honest {}", "clean pipeline honestly scores {}"],
        "wf_leak_rmse": ["true error is {}", "honest error of {}"],
        "kfold_separation_pct": ["within {}%", "{}% of each other"],
        "wf_degradation_pct": ["{}% worse", "{}% higher"],
        "kfold_understatement_pct": ["{}% below", "by {}%"],
        "null_threshold": ["threshold of {}", "{} RMSE threshold"],
        "max_clean_gap": ["as large as {}", "a gap of {}"],
        "detection_pct": ["{}% of the seeded leaks", "catches {}%"],
        "feature_fit_share_pct": ["{}% feature-fit", "{}% survives"],
        "membership_share_pct": ["{}% membership", "{}% from letting"],
        "smallest_band_margin": ["never below {}", "narrowest {}"],
        "margin_to_detectable_ratio": ["{} times the smallest", "{}x the smallest"],
        "folds_swept_high": ["up to {} folds", "{} folds"],
        "honest_leaky_seed0_rmse": ["about {} RMSE", "to {} RMSE"],
    }

    checked_documents = [
        "README.md",
        "docs/adr/ADR-001-walk-forward-as-the-honest-estimator.md",
        "docs/adr/ADR-002-fold-fit-features.md",
        "docs/adr/ADR-003-gap-statistic-and-null-calibration.md",
        "docs/adr/ADR-004-a-deliberately-dull-model.md",
        "docs/adr/ADR-005-synthetic-deterministic-fixture.md",
    ]

    return {
        "metrics": metrics,
        "anchors": anchors,
        "checked_documents": checked_documents,
        "note": "every value here is produced by running the suite and the four experiments",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Re-measure and write docs/metrics.json")
    parser.add_argument("--skip-tests", action="store_true",
                        help="read existing reports rather than running pytest")
    args = parser.parse_args(argv)

    payload = collect(args.skip_tests)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {METRICS_OUT.relative_to(ROOT)} with {len(payload['metrics'])} metrics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
