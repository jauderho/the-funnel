"""Mean-reversion / oscillator strategies (12 families).

Each function follows the strategy contract in ``funnel.strategies.base``:
causal-only, position in {-1, 0, 1}, warmup rows at 0.0.
"""

import numpy as np
import pandas as pd

from funnel.strategies import indicators as ind
from funnel.strategies.base import clean_positions


def rsi_revert(
    df: pd.DataFrame, window: int = 14, oversold: float = 30.0, overbought: float = 70.0
) -> pd.Series:
    """Long when RSI is oversold, short when overbought, else hold prior state."""
    rsi_line = ind.rsi(df["close"], window)
    raw = pd.Series(np.nan, index=df.index)
    raw = raw.mask(rsi_line < oversold, 1.0)
    raw = raw.mask(rsi_line > overbought, -1.0)
    return clean_positions(raw.ffill(), df.index)


def bollinger_revert(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Long below the lower Bollinger band, short above the upper band, else hold."""
    _, upper, lower = ind.bollinger_bands(df["close"], window, n_std)
    raw = pd.Series(np.nan, index=df.index)
    raw = raw.mask(df["close"] < lower, 1.0)
    raw = raw.mask(df["close"] > upper, -1.0)
    return clean_positions(raw.ffill(), df.index)


def zscore_revert(df: pd.DataFrame, window: int = 20, threshold: float = 1.5) -> pd.Series:
    """Long when the rolling z-score is very negative, short when very positive."""
    z = ind.rolling_zscore(df["close"], window)
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(z > -threshold, 1.0)
    raw = raw.where(z < threshold, -1.0)
    return clean_positions(raw, df.index)


def stochastic_revert(
    df: pd.DataFrame,
    k_window: int = 14,
    d_window: int = 3,
    oversold: float = 20.0,
    overbought: float = 80.0,
) -> pd.Series:
    """Long when stochastic %K is oversold, short when overbought."""
    percent_k, _ = ind.stochastic(df, k_window, d_window)
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(percent_k > oversold, 1.0)
    raw = raw.where(percent_k < overbought, -1.0)
    return clean_positions(raw, df.index)


def cci_revert(df: pd.DataFrame, window: int = 20, threshold: float = 100.0) -> pd.Series:
    """Long when CCI is deeply negative, short when deeply positive."""
    cci_line = ind.cci(df, window)
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(cci_line > -threshold, 1.0)
    raw = raw.where(cci_line < threshold, -1.0)
    return clean_positions(raw, df.index)


def williams_r_revert(
    df: pd.DataFrame, window: int = 14, oversold: float = -80.0, overbought: float = -20.0
) -> pd.Series:
    """Long when Williams %R shows oversold, short when overbought."""
    wr = ind.williams_r(df, window)
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(wr > oversold, 1.0)
    raw = raw.where(wr < overbought, -1.0)
    return clean_positions(raw, df.index)


def keltner_revert(
    df: pd.DataFrame, ema_window: int = 20, atr_window: int = 10, mult: float = 2.0
) -> pd.Series:
    """Long below the lower Keltner channel, short above the upper channel."""
    _, upper, lower = ind.keltner_channels(df, ema_window, atr_window, mult)
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(df["close"] >= lower, 1.0)
    raw = raw.where(df["close"] <= upper, -1.0)
    return clean_positions(raw, df.index)


def vwap_revert(df: pd.DataFrame, window: int = 20, threshold: float = 0.01) -> pd.Series:
    """Long when price is meaningfully below rolling VWAP, short when above."""
    vwap = ind.rolling_vwap(df, window)
    deviation = (df["close"] - vwap) / vwap
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(deviation > -threshold, 1.0)
    raw = raw.where(deviation < threshold, -1.0)
    return clean_positions(raw, df.index)


def percent_b_revert(
    df: pd.DataFrame, window: int = 20, n_std: float = 2.0, low: float = 0.05, high: float = 0.95
) -> pd.Series:
    """Long when %B (position within Bollinger bands) is near 0, short near 1."""
    _, upper, lower = ind.bollinger_bands(df["close"], window, n_std)
    band_width = (upper - lower).replace(0.0, np.nan)
    percent_b = (df["close"] - lower) / band_width
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(percent_b > low, 1.0)
    raw = raw.where(percent_b < high, -1.0)
    return clean_positions(raw, df.index)


def connors_rsi_revert(
    df: pd.DataFrame,
    rsi_window: int = 3,
    streak_window: int = 2,
    rank_window: int = 100,
    oversold: float = 10.0,
    overbought: float = 90.0,
) -> pd.Series:
    """Long when Connors RSI is deeply oversold, short when deeply overbought."""
    crsi = ind.connors_rsi(df["close"], rsi_window, streak_window, rank_window)
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(crsi > oversold, 1.0)
    raw = raw.where(crsi < overbought, -1.0)
    return clean_positions(raw, df.index)


def ultimate_oscillator_revert(
    df: pd.DataFrame,
    window1: int = 7,
    window2: int = 14,
    window3: int = 28,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.Series:
    """Long when Ultimate Oscillator is oversold, short when overbought."""
    uo = ind.ultimate_oscillator(df, window1, window2, window3)
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(uo > oversold, 1.0)
    raw = raw.where(uo < overbought, -1.0)
    return clean_positions(raw, df.index)


def gap_fade(df: pd.DataFrame, threshold: float = 0.02) -> pd.Series:
    """Fade overnight gaps: short a large gap up, long a large gap down."""
    gap = df["open"] / df["close"].shift(1) - 1.0
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(gap < threshold, -1.0)
    raw = raw.where(gap > -threshold, 1.0)
    return clean_positions(raw, df.index)
