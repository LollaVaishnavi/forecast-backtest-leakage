"""Three ways to cut a time series into train and test folds.

All three cut the usable rows into the same contiguous, time-ordered blocks, so a fold
is the same slice of calendar in each. The only thing that changes between them is which
rows are allowed into the training set:

- walk-forward lets in only rows before the test block (honest);
- blocked k-fold lets in every other block, future included, keeping the test blocks
  contiguous so it differs from walk-forward in exactly one thing, the direction of
  time;
- shuffled k-fold is the textbook practice: rows are permuted into folds, so a training
  set is a random sample from the whole timeline.

Keeping the block boundaries identical is what lets the gap between schemes be read as a
property of the protocol rather than of the slicing.
"""

from __future__ import annotations

import numpy as np

Split = tuple[np.ndarray, np.ndarray]


def blocked_folds(usable: np.ndarray, n_folds: int) -> list[np.ndarray]:
    """Split the usable rows into ``n_folds`` contiguous, time-ordered blocks."""
    return [np.asarray(block, dtype=np.int64) for block in np.array_split(usable, n_folds)]


def block_size(usable: np.ndarray, n_folds: int) -> int:
    """Rows in the smallest block, the unit the test fold is measured in."""
    return min(block.size for block in blocked_folds(usable, n_folds))


def walk_forward_splits(
    usable: np.ndarray, n_folds: int, embargo: int, scheme: str
) -> list[Split]:
    """Honest splits: test on each block after the first, train only on its past.

    The first block is never a test fold because it has no past to train on, so an
    ``n_folds`` series yields ``n_folds - 1`` steps. ``embargo`` rows immediately before
    a test block are dropped from training, so a trailing-window feature cannot straddle
    the boundary and carry the last training rows into the test region. Under
    ``sliding`` the training set is capped at the most recent blocks rather than all
    history, which helps when distant history is from a different regime.
    """
    blocks = blocked_folds(usable, n_folds)
    splits: list[Split] = []
    max_train_blocks = max(1, n_folds // 2)

    for f in range(1, n_folds):
        test = blocks[f]
        test_start = int(test[0])
        start = max(0, f - max_train_blocks) if scheme == "sliding" else 0
        past_blocks = blocks[start:f]
        past = (
            np.concatenate(past_blocks) if past_blocks else np.empty(0, dtype=np.int64)
        )
        train = past[past < test_start - embargo]
        if train.size == 0:
            # Too small a history after the embargo; this step cannot be scored.
            continue
        splits.append((train.astype(np.int64), test.astype(np.int64)))
    return splits


def blocked_kfold_splits(usable: np.ndarray, n_folds: int) -> list[Split]:
    """Contiguous test blocks, but training on every other block, future included.

    This differs from :func:`walk_forward_splits` in one thing only: a test block's
    training set contains the blocks that come after it in time. Comparing the two
    isolates the effect of letting the future into training from the effect of how the
    test rows are chosen.
    """
    blocks = blocked_folds(usable, n_folds)
    splits: list[Split] = []
    for f in range(n_folds):
        test = blocks[f]
        train = np.concatenate([blocks[g] for g in range(n_folds) if g != f])
        splits.append((train.astype(np.int64), test.astype(np.int64)))
    return splits


def shuffled_kfold_splits(usable: np.ndarray, n_folds: int, seed: int) -> list[Split]:
    """The textbook practice: permute rows, then cut into ``n_folds`` folds.

    A training set is now a random sample spread across the whole timeline, so for any
    test row its temporal neighbours are almost all available for fitting. That is the
    mechanism by which a shuffled split quietly supplies the future. The permutation is
    seeded so the splits are reproducible.
    """
    rng = np.random.default_rng(seed)
    permuted = usable.copy()
    rng.shuffle(permuted)
    folds = [np.asarray(f, dtype=np.int64) for f in np.array_split(permuted, n_folds)]
    splits: list[Split] = []
    for f in range(n_folds):
        test = folds[f]
        train = np.concatenate([folds[g] for g in range(n_folds) if g != f])
        splits.append((np.sort(train).astype(np.int64), np.sort(test).astype(np.int64)))
    return splits
