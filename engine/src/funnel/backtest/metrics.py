"""Performance metrics computed on a daily strategy return series.

Every function takes a return series produced by
``funnel.backtest.engine.strategy_returns`` (or a stitched combination of
such series) and/or the underlying position series used to produce it.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

_ZERO_STD_ATOL = 1e-12
"""Absolute tolerance for treating standard deviation as zero. A perfectly
constant return series can still produce a tiny nonzero ``std(ddof=1)`` due
to floating-point rounding (e.g. ~1e-18 for a series of identical 0.01
values); without this tolerance such a series would report an enormous,
meaningless Sharpe ratio instead of the documented 0.0."""


def sharpe(returns: pd.Series) -> float:
    """Annualized Sharpe ratio: mean / std * sqrt(252).

    Returns ``0.0`` when the sample has fewer than two observations or a
    standard deviation that is zero, NaN, or indistinguishable from zero up
    to floating-point noise (see ``_ZERO_STD_ATOL``) — an undefined ratio is
    reported as "no edge" rather than raising or propagating NaN/inf into
    the funnel.
    """
    clean = returns.dropna()
    if len(clean) < 2:
        return 0.0
    std = clean.std(ddof=1)
    if np.isnan(std) or std <= _ZERO_STD_ATOL:
        return 0.0
    return float(clean.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def _equity_curve(returns: pd.Series) -> pd.Series:
    """Compounded equity curve starting at 1.0, from a daily return series."""
    clean = returns.dropna()
    return (1.0 + clean).cumprod()


def max_drawdown(returns: pd.Series) -> float:
    """Most negative peak-to-trough drawdown on the compounded equity curve.

    Returns a value <= 0.0 (0.0 for an empty series, since there is no
    drawdown to measure). E.g. -0.20 means a 20% peak-to-trough decline.
    """
    equity = _equity_curve(returns)
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def drawdown_duration(returns: pd.Series) -> int:
    """Longest underwater stretch, in trading days (bars below the prior peak).

    A day is "underwater" if the equity curve is strictly below its
    running peak. Returns 0 for an empty series or a series that never
    dips below its running peak.
    """
    equity = _equity_curve(returns)
    if equity.empty:
        return 0
    running_max = equity.cummax()
    underwater = equity < running_max

    longest = 0
    current = 0
    for is_underwater in underwater:
        if is_underwater:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def trade_count(positions: pd.Series) -> int:
    """Number of position changes (nonzero diffs), ignoring the warmup NaN."""
    diffs = positions.diff().dropna()
    return int((diffs != 0.0).sum())


def cagr(returns: pd.Series) -> float:
    """Compound annual growth rate from a daily return series.

    Returns 0.0 for an empty series. Uses the actual number of return
    observations (not calendar days) at 252 trading days/year to annualize.
    """
    equity = _equity_curve(returns)
    if equity.empty:
        return 0.0
    n_days = len(equity)
    final_value = float(equity.iloc[-1])
    if final_value <= 0.0:
        return -1.0
    years = n_days / TRADING_DAYS_PER_YEAR
    if years <= 0.0:
        return 0.0
    return float(final_value ** (1.0 / years) - 1.0)


def win_rate(returns: pd.Series) -> float:
    """Fraction of nonzero-return days that are positive.

    Zero-return days (e.g. a flat position earning nothing) are excluded
    from the denominator: they are neither wins nor losses, and including
    them would understate the win rate of a strategy that is frequently
    flat. Returns 0.0 if there are no nonzero-return days.
    """
    clean = returns.dropna()
    nonzero = clean[clean != 0.0]
    if nonzero.empty:
        return 0.0
    return float((nonzero > 0.0).sum() / len(nonzero))
