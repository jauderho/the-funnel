"""Volume-based strategies (6 families).

Each function follows the strategy contract in ``funnel.strategies.base``:
causal-only, position in {-1, 0, 1}, warmup rows at 0.0.
"""

import numpy as np
import pandas as pd

from funnel.strategies import indicators as ind
from funnel.strategies.base import clean_positions


def obv_trend(df: pd.DataFrame, ema_window: int = 20) -> pd.Series:
    """Long when On-Balance Volume is above its EMA, short otherwise."""
    obv_line = ind.obv(df)
    obv_ema = ind.ema(obv_line, ema_window)
    raw = pd.Series(np.sign(obv_line - obv_ema), index=df.index)
    return clean_positions(raw, df.index)


def chaikin_money_flow_trend(
    df: pd.DataFrame, window: int = 20, threshold: float = 0.0
) -> pd.Series:
    """Long when Chaikin Money Flow is positive, short when negative."""
    cmf = ind.chaikin_money_flow(df, window)
    raw = pd.Series(np.sign(cmf - threshold), index=df.index)
    return clean_positions(raw, df.index)


def money_flow_index_trend(
    df: pd.DataFrame, window: int = 14, oversold: float = 20.0, overbought: float = 80.0
) -> pd.Series:
    """Long when Money Flow Index shows oversold, short when overbought."""
    mfi = ind.money_flow_index(df, window)
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(mfi > oversold, 1.0)
    raw = raw.where(mfi < overbought, -1.0)
    return clean_positions(raw, df.index)


def volume_surge(df: pd.DataFrame, window: int = 20, mult: float = 2.0) -> pd.Series:
    """Long a surge (volume >> its rolling average) on an up bar, short on a down bar."""
    avg_volume = df["volume"].rolling(window).mean()
    surge = df["volume"] > mult * avg_volume
    bar_return = df["close"].pct_change()
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(~(surge & (bar_return > 0.0)), 1.0)
    raw = raw.where(~(surge & (bar_return < 0.0)), -1.0)
    return clean_positions(raw, df.index)


def force_index_trend(df: pd.DataFrame, window: int = 13) -> pd.Series:
    """Long when the Force Index is positive, short when negative."""
    fi = ind.force_index(df, window)
    raw = pd.Series(np.sign(fi), index=df.index)
    return clean_positions(raw, df.index)


def chaikin_oscillator(df: pd.DataFrame, fast: int = 3, slow: int = 10) -> pd.Series:
    """Chaikin Oscillator: EMA(fast) - EMA(slow) of the Accumulation/Distribution Line."""
    denom = (df["high"] - df["low"]).replace(0.0, np.nan)
    mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / denom
    adl = (mf_multiplier.fillna(0.0) * df["volume"]).cumsum()
    oscillator = ind.ema(adl, fast) - ind.ema(adl, slow)
    raw = pd.Series(np.sign(oscillator), index=df.index)
    return clean_positions(raw, df.index)
