"""Candlestick / chart-pattern strategies (4 families).

Each function follows the strategy contract in ``funnel.strategies.base``:
causal-only, position in {-1, 0, 1}, warmup rows at 0.0.
"""

import numpy as np
import pandas as pd

from funnel.strategies import indicators as ind
from funnel.strategies.base import clean_positions


def engulfing(df: pd.DataFrame) -> pd.Series:
    """Bullish/bearish engulfing candle: current body fully contains the prior body."""
    prev_open = df["open"].shift(1)
    prev_close = df["close"].shift(1)
    bullish = (
        (prev_close < prev_open)
        & (df["close"] > df["open"])
        & (df["close"] >= prev_open)
        & (df["open"] <= prev_close)
    )
    bearish = (
        (prev_close > prev_open)
        & (df["close"] < df["open"])
        & (df["open"] >= prev_close)
        & (df["close"] <= prev_open)
    )
    raw = pd.Series(np.nan, index=df.index)
    raw = raw.mask(bullish, 1.0)
    raw = raw.mask(bearish, -1.0)
    return clean_positions(raw.ffill(), df.index)


def three_bar_reversal(df: pd.DataFrame) -> pd.Series:
    """Three consecutive down (up) closes followed by a strong reversal close."""
    close = df["close"]
    down_streak = (close.shift(1) < close.shift(2)) & (close.shift(2) < close.shift(3))
    up_streak = (close.shift(1) > close.shift(2)) & (close.shift(2) > close.shift(3))
    bullish_reversal = down_streak & (close > close.shift(1))
    bearish_reversal = up_streak & (close < close.shift(1))
    raw = pd.Series(np.nan, index=df.index)
    raw = raw.mask(bullish_reversal, 1.0)
    raw = raw.mask(bearish_reversal, -1.0)
    return clean_positions(raw.ffill(), df.index)


def higher_highs_higher_lows(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Long on a run of rolling higher-highs & higher-lows, short on the mirror."""
    high_rising = df["high"] > df["high"].shift(1)
    low_rising = df["low"] > df["low"].shift(1)
    high_falling = df["high"] < df["high"].shift(1)
    low_falling = df["low"] < df["low"].shift(1)
    uptrend = (high_rising & low_rising).rolling(window).sum() == window
    downtrend = (high_falling & low_falling).rolling(window).sum() == window
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(~uptrend, 1.0)
    raw = raw.where(~downtrend, -1.0)
    return clean_positions(raw, df.index)


def pivot_bounce(df: pd.DataFrame, window: int = 5, threshold: float = 0.003) -> pd.Series:
    """Long near a rolling pivot support level, short near rolling resistance."""
    _, support, resistance = ind.pivot_points(df, window)
    near_support = (df["low"] - support).abs() / support <= threshold
    near_resistance = (df["high"] - resistance).abs() / resistance <= threshold
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(~near_support, 1.0)
    raw = raw.where(~near_resistance, -1.0)
    return clean_positions(raw, df.index)
