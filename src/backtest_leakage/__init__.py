"""backtest_leakage: measure the leakage gap a k-fold score hides on a time series.

The package exists to make one structural fact measurable: random k-fold cross
validation cannot detect look-ahead leakage, because shuffling puts future rows in
the training set of every fold and so rewards a leak instead of exposing it. A
time-ordered walk-forward backtest keeps the ordering that would expose it. The
modules here run a forecasting pipeline through both schemes, decompose the gap
between them into its leak channels, and calibrate that gap against a
non-stationarity null so a drift is not mistaken for a leak.
"""

__version__ = "0.1.0"
