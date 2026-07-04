"""Golden tests for backtest metrics: hand-computed values on tiny fixed series."""

import numpy as np
import pandas as pd
import pytest

from funnel.backtest.metrics import (
    cagr,
    drawdown_duration,
    max_drawdown,
    sharpe,
    trade_count,
    win_rate,
)


def test_sharpe_alternating_returns() -> None:
    returns = pd.Series([0.01, -0.01, 0.01, -0.01, 0.01, -0.01])
    expected_mean = returns.mean()
    expected_std = returns.std(ddof=1)
    expected = expected_mean / expected_std * np.sqrt(252)
    assert sharpe(returns) == pytest.approx(expected)


def test_sharpe_monotone_positive_returns() -> None:
    returns = pd.Series([0.01] * 10)
    # constant returns -> zero std -> defined as 0.0, not inf/NaN.
    assert sharpe(returns) == 0.0


def test_sharpe_zero_variance_returns_zero() -> None:
    returns = pd.Series([0.0, 0.0, 0.0])
    assert sharpe(returns) == 0.0


def test_sharpe_empty_series_returns_zero() -> None:
    assert sharpe(pd.Series([], dtype="float64")) == 0.0


def test_sharpe_single_observation_returns_zero() -> None:
    assert sharpe(pd.Series([0.05])) == 0.0


def test_sharpe_all_nan_returns_zero() -> None:
    assert sharpe(pd.Series([np.nan, np.nan])) == 0.0


def test_max_drawdown_known_path() -> None:
    # Equity: 1.0 -> 1.10 -> 0.99 -> 1.188 (30% up from trough of 0.99... let's hand-verify)
    returns = pd.Series([0.10, -0.10, 0.20])
    equity = (1.0 + returns).cumprod()
    running_max = equity.cummax()
    expected = float((equity / running_max - 1.0).min())
    assert max_drawdown(returns) == pytest.approx(expected)
    assert max_drawdown(returns) <= 0.0


def test_max_drawdown_monotone_up_is_zero() -> None:
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    assert max_drawdown(returns) == pytest.approx(0.0)


def test_max_drawdown_empty_series_is_zero() -> None:
    assert max_drawdown(pd.Series([], dtype="float64")) == 0.0


def test_drawdown_duration_known_path() -> None:
    # Peak at day0 (equity=1.0), underwater days1-3, new peak at day4.
    returns = pd.Series([0.0, -0.05, -0.01, 0.02, 0.10])
    assert drawdown_duration(returns) == 3


def test_drawdown_duration_never_underwater() -> None:
    returns = pd.Series([0.01, 0.01, 0.01])
    assert drawdown_duration(returns) == 0


def test_drawdown_duration_empty_series() -> None:
    assert drawdown_duration(pd.Series([], dtype="float64")) == 0


def test_trade_count_counts_nonzero_diffs() -> None:
    positions = pd.Series([0.0, 1.0, 1.0, -1.0, 0.0])
    # diffs: NaN, 1, 0, -2, 1 -> 3 nonzero changes
    assert trade_count(positions) == 3


def test_trade_count_no_changes() -> None:
    positions = pd.Series([1.0, 1.0, 1.0])
    assert trade_count(positions) == 0


def test_trade_count_empty_series() -> None:
    assert trade_count(pd.Series([], dtype="float64")) == 0


def test_cagr_empty_series_is_zero() -> None:
    assert cagr(pd.Series([], dtype="float64")) == 0.0


def test_cagr_positive_known_value() -> None:
    # 252 days of 0.01 daily return compounds to (1.01)^252 over 1 year.
    returns = pd.Series([0.01] * 252)
    result = cagr(returns)
    expected = (1.01**252) ** (1.0 / 1.0) - 1.0
    assert result == pytest.approx(expected)


def test_win_rate_excludes_zero_return_days() -> None:
    returns = pd.Series([0.01, -0.01, 0.0, 0.02, 0.0, -0.03])
    # nonzero days: 0.01, -0.01, 0.02, -0.03 -> 2 positive of 4
    assert win_rate(returns) == pytest.approx(0.5)


def test_win_rate_all_zero_returns_zero() -> None:
    assert win_rate(pd.Series([0.0, 0.0, 0.0])) == 0.0


def test_win_rate_empty_series_is_zero() -> None:
    assert win_rate(pd.Series([], dtype="float64")) == 0.0
