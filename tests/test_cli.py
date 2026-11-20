"""Properties of the CLI: the exit code is the product, and it means one thing."""

from __future__ import annotations

import json

import pytest

from backtest_leakage.cli import main
from backtest_leakage.gap import EXIT_FOR_VERDICT, VERDICT_INCONCLUSIVE, VERDICT_LEAK


def test_the_leaky_pipeline_exits_with_the_leak_code(tmp_path, capsys):
    html = tmp_path / "report.html"
    out = tmp_path / "out.json"
    code = main(["--seed", "2", "--leaky", "--html", str(html), "--json", str(out)])
    assert code == EXIT_FOR_VERDICT[VERDICT_LEAK]
    payload = json.loads(out.read_text())
    assert payload["verdict"] == VERDICT_LEAK
    assert payload["exit_code"] == code
    assert html.read_text().startswith("<!DOCTYPE html>")
    assert "verdict: LEAK" in capsys.readouterr().out


def test_the_clean_pipeline_does_not_exit_with_the_leak_code(capsys):
    code = main(["--seed", "2", "--clean"])
    # On ten folds the clean pipeline cannot be certified, so it reports inconclusive,
    # which is a different exit code from both a leak and a clean bill of health.
    assert code == EXIT_FOR_VERDICT[VERDICT_INCONCLUSIVE]
    assert "verdict: INCONCLUSIVE" in capsys.readouterr().out


def test_a_bad_policy_path_returns_the_usage_code_not_a_traceback(capsys):
    code = main(["--seed", "0", "--policy", "/no/such/policy.toml"])
    assert code == 1
    assert "policy error" in capsys.readouterr().err


def test_an_unknown_flag_exits_with_the_usage_code(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--not-a-flag"])
    assert excinfo.value.code == 1


def test_asking_for_leaky_and_clean_at_once_is_a_usage_error():
    with pytest.raises(SystemExit) as excinfo:
        main(["--leaky", "--clean"])
    assert excinfo.value.code == 1
