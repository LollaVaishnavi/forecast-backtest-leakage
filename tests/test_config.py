"""Properties of policy loading and the validation refused at the boundary."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from backtest_leakage.config import load_policy

_GOOD = """
[backtest]
n_folds = 10
embargo = 7
scheme = "expanding"

[features]
rolling_window = 7
lags = [1, 7, 14]
anchor_halfwidth = 10
fold_fit = true

[model]
ridge_alpha = 1.0

[null]
n_series = 40
threshold_percentile = 95.0

[verdict]
min_detectable_gap = 0.05
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "policy.toml"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return path


def test_the_default_policy_file_loads_and_is_internally_consistent():
    pol = load_policy()
    assert pol.backtest.n_folds >= 3
    assert pol.features.anchor_halfwidth >= 1


def test_loading_a_well_formed_policy_returns_every_section(tmp_path):
    pol = load_policy(_write(tmp_path, _GOOD))
    assert pol.backtest.scheme == "expanding"
    assert pol.features.lags == (1, 7, 14)
    assert pol.model.ridge_alpha == 1.0
    assert pol.null.n_series == 40
    assert pol.verdict.min_detectable_gap == 0.05


def test_the_resolution_floor_is_one_over_the_null_sample_size(tmp_path):
    pol = load_policy(_write(tmp_path, _GOOD))
    assert pol.resolution_floor == pytest.approx(1.0 / 40)


def test_a_missing_policy_file_is_reported_as_such(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_policy(tmp_path / "absent.toml")


def test_a_missing_key_names_the_section_and_the_key(tmp_path):
    body = _GOOD.replace("n_folds = 10\n", "")
    with pytest.raises(ValueError, match="n_folds"):
        load_policy(_write(tmp_path, body))


@pytest.mark.parametrize(
    "replacement, needle",
    [
        ("n_folds = 10", "n_folds = 2"),
        ("embargo = 7", "embargo = -1"),
        ('scheme = "expanding"', 'scheme = "random"'),
        ("rolling_window = 7", "rolling_window = 0"),
        ("lags = [1, 7, 14]", "lags = [0, 7]"),
        ("anchor_halfwidth = 10", "anchor_halfwidth = 0"),
        ("threshold_percentile = 95.0", "threshold_percentile = 40.0"),
        ("n_series = 40", "n_series = 1"),
    ],
)
def test_an_impossible_value_is_refused_with_a_message(tmp_path, replacement, needle):
    body = _GOOD.replace(replacement, needle)
    with pytest.raises(ValueError):
        load_policy(_write(tmp_path, body))
