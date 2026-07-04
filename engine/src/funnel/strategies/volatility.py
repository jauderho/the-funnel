"""Volatility-based strategies (3 families).

Each function follows the strategy contract in ``funnel.strategies.base``:
causal-only, position in {-1, 0, 1}, warmup rows at 0.0.
"""

import numpy as np
import pandas as pd

from funnel.strategies import indicators as ind
from funnel.strategies.base import clean_positions


def atr_breakout(df: pd.DataFrame, atr_window: int = 14, mult: float = 1.5) -> pd.Series:
    """Long a close-over-close move beyond an ATR multiple, short the mirror move."""
    atr_vals = ind.atr(df, atr_window)
    move = df["close"] - df["close"].shift(1)
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(move <= mult * atr_vals, 1.0)
    raw = raw.where(move >= -mult * atr_vals, -1.0)
    return clean_positions(raw, df.index)


def volatility_breakout(df: pd.DataFrame, window: int = 20, mult: float = 1.0) -> pd.Series:
    """Long/short when the bar's range expands beyond its rolling average range."""
    bar_range = df["high"] - df["low"]
    avg_range = bar_range.rolling(window).mean()
    expansion = bar_range > mult * avg_range
    bar_return = df["close"].pct_change()
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(~(expansion & (bar_return > 0.0)), 1.0)
    raw = raw.where(~(expansion & (bar_return < 0.0)), -1.0)
    return clean_positions(raw, df.index)


def squeeze_breakout(
    df: pd.DataFrame,
    bb_window: int = 20,
    bb_std: float = 2.0,
    kc_ema_window: int = 20,
    kc_atr_window: int = 10,
    kc_mult: float = 1.5,
) -> pd.Series:
    """TTM-style squeeze: when Bollinger bands sit inside Keltner channels (low
    volatility), trade the direction of the breakout once the squeeze fires."""
    _, bb_upper, bb_lower = ind.bollinger_bands(df["close"], bb_window, bb_std)
    _, kc_upper, kc_lower = ind.keltner_channels(df, kc_ema_window, kc_atr_window, kc_mult)
    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)
    squeeze_fired = squeeze_on.shift(1).fillna(False) & ~squeeze_on
    momentum = df["close"] - df["close"].shift(bb_window)

    raw = pd.Series(np.nan, index=df.index)
    raw = raw.mask(squeeze_fired & (momentum > 0.0), 1.0)
    raw = raw.mask(squeeze_fired & (momentum < 0.0), -1.0)
    return clean_positions(raw.ffill(), df.index)
