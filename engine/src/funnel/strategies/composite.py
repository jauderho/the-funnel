"""Composite strategies combining multiple signals (3 families).

Each function follows the strategy contract in ``funnel.strategies.base``:
causal-only, position in {-1, 0, 1}, warmup rows at 0.0.
"""

import numpy as np
import pandas as pd

from funnel.strategies import indicators as ind
from funnel.strategies.base import clean_positions


def macd_rsi_confirm(
    df: pd.DataFrame,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    rsi_window: int = 14,
    rsi_midline: float = 50.0,
) -> pd.Series:
    """Long when MACD is bullish AND RSI confirms above its midline; symmetric short."""
    macd_line = ind.ema(df["close"], macd_fast) - ind.ema(df["close"], macd_slow)
    signal_line = ind.ema(macd_line, macd_signal)
    rsi_line = ind.rsi(df["close"], rsi_window)

    long_cond = (macd_line > signal_line) & (rsi_line > rsi_midline)
    short_cond = (macd_line < signal_line) & (rsi_line < rsi_midline)
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(~long_cond, 1.0)
    raw = raw.where(~short_cond, -1.0)
    return clean_positions(raw, df.index)


def triple_screen(
    df: pd.DataFrame,
    weekly_proxy_window: int = 5,
    long_ema: int = 26,
    oscillator_window: int = 14,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.Series:
    """Elder's Triple Screen, both screens from daily data.

    Screen 1 (tide): a weekly-proxy trend built by taking a rolling
    ``weekly_proxy_window``-day resample of daily closes (a rolling mean
    standing in for a weekly bar) and comparing it to its own long EMA —
    this sets the allowed trade direction. Screen 2 (wave): a daily
    stochastic oscillator times entries against that trend, trading
    oversold pullbacks in an uptrend and overbought rallies in a downtrend.
    """
    weekly_proxy = df["close"].rolling(weekly_proxy_window).mean()
    weekly_trend = ind.ema(weekly_proxy, long_ema)
    tide_up = weekly_proxy > weekly_trend
    tide_down = weekly_proxy < weekly_trend

    percent_k, _ = ind.stochastic(df, oscillator_window)

    raw = pd.Series(0.0, index=df.index)
    long_cond = tide_up & (percent_k < oversold)
    short_cond = tide_down & (percent_k > overbought)
    raw = raw.where(~long_cond, 1.0)
    raw = raw.where(~short_cond, -1.0)
    return clean_positions(raw, df.index)


def chandelier(
    df: pd.DataFrame, entry_window: int = 20, atr_window: int = 22, mult: float = 3.0
) -> pd.Series:
    """Trend entry (N-day breakout) with a chandelier ATR trailing-stop exit."""
    prior_high = df["close"].shift(1).rolling(entry_window).max()
    prior_low = df["close"].shift(1).rolling(entry_window).min()
    atr_vals = ind.atr(df, atr_window)

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    entry_high = prior_high.to_numpy(dtype=np.float64)
    entry_low = prior_low.to_numpy(dtype=np.float64)
    atr_arr = atr_vals.to_numpy(dtype=np.float64)
    n = len(df)

    position = np.zeros(n, dtype=np.float64)
    highest_since_entry = np.nan
    lowest_since_entry = np.nan
    state = 0.0

    for i in range(n):
        if np.isnan(entry_high[i]) or np.isnan(atr_arr[i]):
            position[i] = 0.0
            continue

        if state == 0.0:
            if close[i] >= entry_high[i]:
                state = 1.0
                highest_since_entry = high[i]
            elif close[i] <= entry_low[i]:
                state = -1.0
                lowest_since_entry = low[i]
        elif state == 1.0:
            highest_since_entry = max(highest_since_entry, high[i])
            chandelier_stop = highest_since_entry - mult * atr_arr[i]
            if close[i] < chandelier_stop:
                state = 0.0
        elif state == -1.0:
            lowest_since_entry = min(lowest_since_entry, low[i])
            chandelier_stop = lowest_since_entry + mult * atr_arr[i]
            if close[i] > chandelier_stop:
                state = 0.0

        position[i] = state

    raw = pd.Series(position, index=df.index)
    return clean_positions(raw, df.index)
