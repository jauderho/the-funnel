"""Sanity checks that strategies actually respond to their intended regime."""

import pandas as pd

from funnel.strategies import meanrev, trend


def test_ma_crossover_ends_long_on_uptrend(trending_ohlcv: pd.DataFrame) -> None:
    positions = trend.ma_crossover(trending_ohlcv, fast=10, slow=50)
    assert positions.iloc[-1] == 1.0


def test_time_series_momentum_ends_long_on_uptrend(trending_ohlcv: pd.DataFrame) -> None:
    positions = trend.time_series_momentum(trending_ohlcv, lookback=90)
    assert positions.iloc[-1] == 1.0


def test_rsi_revert_takes_both_directions_on_mean_reverting(
    mean_reverting_ohlcv: pd.DataFrame,
) -> None:
    positions = meanrev.rsi_revert(mean_reverting_ohlcv, window=14)
    assert (positions == 1.0).any()
    assert (positions == -1.0).any()


def test_ma_crossover_warmup_is_flat(trending_ohlcv: pd.DataFrame) -> None:
    positions = trend.ma_crossover(trending_ohlcv, fast=10, slow=50)
    # Before the slow MA has enough history, the signal must be flat.
    assert (positions.iloc[:49] == 0.0).all()


def test_rsi_revert_warmup_is_flat(mean_reverting_ohlcv: pd.DataFrame) -> None:
    positions = meanrev.rsi_revert(mean_reverting_ohlcv, window=14)
    assert (positions.iloc[:14] == 0.0).all()
