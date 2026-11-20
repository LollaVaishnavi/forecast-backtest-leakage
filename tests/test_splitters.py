"""Properties of the three split schemes, especially what each lets into training."""

from __future__ import annotations

import numpy as np

from backtest_leakage.splitters import (
    block_size,
    blocked_folds,
    blocked_kfold_splits,
    shuffled_kfold_splits,
    walk_forward_splits,
)


def _usable(n=160, warmup=14):
    return np.arange(warmup, n, dtype=np.int64)


def test_blocked_folds_partition_the_usable_rows_in_order():
    usable = _usable()
    blocks = blocked_folds(usable, 10)
    assert np.array_equal(np.concatenate(blocks), usable)
    assert all(b[0] < b[-1] for b in blocks)


def test_block_size_is_the_smallest_block():
    usable = _usable()
    assert block_size(usable, 10) == min(b.size for b in blocked_folds(usable, 10))


def test_walk_forward_never_trains_on_the_future():
    usable = _usable()
    for train, test in walk_forward_splits(usable, 10, embargo=7, scheme="expanding"):
        assert train.max() < test.min()


def test_walk_forward_respects_the_embargo():
    usable = _usable()
    embargo = 7
    for train, test in walk_forward_splits(usable, 10, embargo=embargo, scheme="expanding"):
        assert train.max() < test.min() - embargo


def test_a_large_embargo_drops_the_unservable_early_steps():
    usable = _usable()
    # An embargo wider than the early history leaves the first scored step with no
    # training rows, so that step is skipped rather than scored on nothing.
    splits = walk_forward_splits(usable, 10, embargo=40, scheme="expanding")
    assert len(splits) < 9
    for train, _ in splits:
        assert train.size > 0


def test_sliding_trains_on_fewer_rows_than_expanding_late_in_the_series():
    usable = _usable(n=400)
    exp = walk_forward_splits(usable, 10, embargo=7, scheme="expanding")
    sli = walk_forward_splits(usable, 10, embargo=7, scheme="sliding")
    assert sli[-1][0].size < exp[-1][0].size


def test_blocked_kfold_trains_on_the_future_but_keeps_test_blocks_contiguous():
    usable = _usable()
    splits = blocked_kfold_splits(usable, 10)
    assert len(splits) == 10
    # At least one fold's training set contains rows after its test block.
    assert any(train.max() > test.max() for train, test in splits)
    # Every test block is a contiguous run of days.
    for _, test in splits:
        assert np.array_equal(test, np.arange(test.min(), test.max() + 1))


def test_shuffled_kfold_is_reproducible_and_covers_every_usable_row_once():
    usable = _usable()
    a = shuffled_kfold_splits(usable, 10, seed=160)
    b = shuffled_kfold_splits(usable, 10, seed=160)
    for (_ta, sa), (_tb, sb) in zip(a, b, strict=True):
        assert np.array_equal(sa, sb)
    all_test = np.sort(np.concatenate([test for _, test in a]))
    assert np.array_equal(all_test, usable)


def test_shuffled_kfold_mixes_time_so_a_fold_is_not_contiguous():
    usable = _usable()
    splits = shuffled_kfold_splits(usable, 10, seed=160)
    # A shuffled test fold spans most of the timeline rather than a single block.
    _, test = splits[0]
    assert test.max() - test.min() > usable.size // 2
