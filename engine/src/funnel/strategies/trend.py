"""Trend-following strategies (19 families).

Each function follows the strategy contract in ``funnel.strategies.base``:
causal-only, position in {-1, 0, 1}, warmup rows at 0.0.
"""

import numpy as np
import pandas as pd

from funnel.strategies import indicators as ind
from funnel.strategies.base import clean_positions


def ma_crossover(df: pd.DataFrame, fast: int = 10, slow: int = 50) -> pd.Series:
    """Long when the fast SMA is above the slow SMA, short otherwise."""
    fast_ma = ind.sma(df["close"], fast)
    slow_ma = ind.sma(df["close"], slow)
    raw = pd.Series(np.sign(fast_ma - slow_ma), index=df.index)
    return clean_positions(raw, df.index)


def time_series_momentum(df: pd.DataFrame, lookback: int = 90) -> pd.Series:
    """Long/short by the sign of the trailing N-day total return."""
    trailing_return = df["close"] / df["close"].shift(lookback) - 1.0
    raw = pd.Series(np.sign(trailing_return), index=df.index)
    return clean_positions(raw, df.index)


def roc_momentum(df: pd.DataFrame, lookback: int = 20, threshold: float = 0.0) -> pd.Series:
    """Long/short by rate-of-change vs a threshold band around zero."""
    roc = df["close"].pct_change(lookback) * 100.0
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(roc.abs() <= threshold, np.sign(roc))
    return clean_positions(raw, df.index)


def macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Long when MACD line is above its signal line, short otherwise."""
    macd_line = ind.ema(df["close"], fast) - ind.ema(df["close"], slow)
    signal_line = ind.ema(macd_line, signal)
    raw = pd.Series(np.sign(macd_line - signal_line), index=df.index)
    return clean_positions(raw, df.index)


def donchian_breakout(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Long on a new N-day high close, short on a new N-day low close."""
    prior_high = df["close"].shift(1).rolling(window).max()
    prior_low = df["close"].shift(1).rolling(window).min()
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(df["close"] <= prior_high, 1.0)
    raw = raw.where(df["close"] >= prior_low, -1.0)
    return clean_positions(raw.ffill(), df.index)


def bollinger_breakout(df: pd.DataFrame, window: int = 20, n_std: float = 2.0) -> pd.Series:
    """Long above the upper Bollinger band, short below the lower band, hold between."""
    _, upper, lower = ind.bollinger_bands(df["close"], window, n_std)
    raw = pd.Series(np.nan, index=df.index)
    raw = raw.where(df["close"] < upper, 1.0)
    raw = raw.where(df["close"] > lower, -1.0)
    return clean_positions(raw.ffill(), df.index)


def supertrend(df: pd.DataFrame, atr_window: int = 10, mult: float = 3.0) -> pd.Series:
    """Supertrend: ATR-band trailing stop that flips direction on breach."""
    atr_vals = ind.atr(df, atr_window).to_numpy(dtype=np.float64)
    hl2 = ((df["high"] + df["low"]) / 2.0).to_numpy(dtype=np.float64)
    close = df["close"].to_numpy(dtype=np.float64)
    n = len(df)

    upper_band = hl2 + mult * atr_vals
    lower_band = hl2 - mult * atr_vals
    direction = np.zeros(n, dtype=np.float64)

    first_valid = int(np.argmax(~np.isnan(atr_vals))) if np.any(~np.isnan(atr_vals)) else n
    for i in range(first_valid, n):
        if i == first_valid:
            direction[i] = 1.0
            continue
        if close[i - 1] > upper_band[i - 1]:
            lower_band[i] = max(lower_band[i], lower_band[i - 1])
        if close[i - 1] < lower_band[i - 1]:
            upper_band[i] = min(upper_band[i], upper_band[i - 1])

        if direction[i - 1] == 1.0 and close[i] < lower_band[i - 1]:
            direction[i] = -1.0
        elif direction[i - 1] == -1.0 and close[i] > upper_band[i - 1]:
            direction[i] = 1.0
        else:
            direction[i] = direction[i - 1]

    raw = pd.Series(direction, index=df.index)
    return clean_positions(raw, df.index)


def parabolic_sar(
    df: pd.DataFrame, af_start: float = 0.02, af_step: float = 0.02, af_max: float = 0.2
) -> pd.Series:
    """Parabolic SAR: trailing stop-and-reverse trend follower."""
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    n = len(df)

    direction = np.zeros(n, dtype=np.float64)
    if n < 2:
        return clean_positions(pd.Series(direction, index=df.index), df.index)

    sar = np.zeros(n, dtype=np.float64)
    trend_up = True
    af = af_start
    ep = high[0]
    sar[0] = low[0]
    direction[0] = 1.0

    for i in range(1, n):
        prev_sar = sar[i - 1]
        candidate = prev_sar + af * (ep - prev_sar)
        if trend_up:
            candidate = min(candidate, low[i - 1], low[max(i - 2, 0)])
            if low[i] < candidate:
                trend_up = False
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                sar[i] = candidate
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            candidate = max(candidate, high[i - 1], high[max(i - 2, 0)])
            if high[i] > candidate:
                trend_up = True
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                sar[i] = candidate
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)
        direction[i] = 1.0 if trend_up else -1.0

    raw = pd.Series(direction, index=df.index)
    return clean_positions(raw, df.index)


def adx_trend(df: pd.DataFrame, window: int = 14, threshold: float = 20.0) -> pd.Series:
    """Directional-movement trend: long/short by +DI vs -DI, gated by ADX strength."""
    plus_di, minus_di, adx = ind.directional_movement(df, window)
    raw = pd.Series(0.0, index=df.index)
    trending = adx >= threshold
    raw = raw.where(~(trending & (plus_di > minus_di)), 1.0)
    raw = raw.where(~(trending & (minus_di > plus_di)), -1.0)
    return clean_positions(raw, df.index)


def ichimoku(df: pd.DataFrame, conversion: int = 9, base: int = 26, span_b: int = 52) -> pd.Series:
    """Ichimoku: long when price is above the (unshifted) cloud, short when below."""
    conv_line = (df["high"].rolling(conversion).max() + df["low"].rolling(conversion).min()) / 2.0
    base_line = (df["high"].rolling(base).max() + df["low"].rolling(base).min()) / 2.0
    span_a = (conv_line + base_line) / 2.0
    span_b_line = (df["high"].rolling(span_b).max() + df["low"].rolling(span_b).min()) / 2.0
    cloud_top = pd.concat([span_a, span_b_line], axis=1).max(axis=1)
    cloud_bottom = pd.concat([span_a, span_b_line], axis=1).min(axis=1)
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(df["close"] <= cloud_top, 1.0)
    raw = raw.where(df["close"] >= cloud_bottom, -1.0)
    return clean_positions(raw, df.index)


def linreg_slope_trend(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Long/short by the sign of the rolling linear-regression slope."""
    slope = ind.linreg_slope(df["close"], window)
    raw = pd.Series(np.sign(slope), index=df.index)
    return clean_positions(raw, df.index)


def aroon_trend(df: pd.DataFrame, window: int = 25, threshold: float = 70.0) -> pd.Series:
    """Long when Aroon-up dominates and is strong, short when Aroon-down dominates."""
    aroon_up, aroon_down = ind.aroon(df, window)
    raw = pd.Series(0.0, index=df.index)
    raw = raw.where(~((aroon_up > aroon_down) & (aroon_up >= threshold)), 1.0)
    raw = raw.where(~((aroon_down > aroon_up) & (aroon_down >= threshold)), -1.0)
    return clean_positions(raw, df.index)


def vortex_trend(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Long when +VI is above -VI, short otherwise."""
    vi_plus, vi_minus = ind.vortex(df, window)
    raw = pd.Series(np.sign(vi_plus - vi_minus), index=df.index)
    return clean_positions(raw, df.index)


def trix_trend(df: pd.DataFrame, window: int = 15, signal: int = 9) -> pd.Series:
    """Long/short by TRIX crossing its own signal (EMA-smoothed) line."""
    trix_line = ind.trix(df["close"], window)
    signal_line = ind.ema(trix_line, signal)
    raw = pd.Series(np.sign(trix_line - signal_line), index=df.index)
    return clean_positions(raw, df.index)


def hull_ma_trend(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Long when price is above the Hull moving average, short otherwise."""
    hma = ind.hull_ma(df["close"], window)
    raw = pd.Series(np.sign(df["close"] - hma), index=df.index)
    return clean_positions(raw, df.index)


def kama_trend(df: pd.DataFrame, window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Long when price is above KAMA, short otherwise."""
    kama_line = ind.kama(df["close"], window, fast, slow)
    raw = pd.Series(np.sign(df["close"] - kama_line), index=df.index)
    return clean_positions(raw, df.index)


def turtle(
    df: pd.DataFrame,
    entry_window: int = 20,
    exit_window: int = 10,
    short_entry_window: int = 55,
    short_exit_window: int = 20,
) -> pd.Series:
    """Turtle system: 20/55-day breakout entries, 10/20-day breakout exits."""
    prior_high_entry = df["close"].shift(1).rolling(entry_window).max()
    prior_low_exit = df["close"].shift(1).rolling(exit_window).min()
    prior_low_entry = df["close"].shift(1).rolling(short_entry_window).min()
    prior_high_exit = df["close"].shift(1).rolling(short_exit_window).max()

    close = df["close"].to_numpy(dtype=np.float64)
    hi_entry = prior_high_entry.to_numpy(dtype=np.float64)
    lo_exit = prior_low_exit.to_numpy(dtype=np.float64)
    lo_entry = prior_low_entry.to_numpy(dtype=np.float64)
    hi_exit = prior_high_exit.to_numpy(dtype=np.float64)
    n = len(df)
    position = np.zeros(n, dtype=np.float64)

    state = 0.0
    for i in range(n):
        if np.isnan(hi_entry[i]) or np.isnan(lo_entry[i]):
            position[i] = 0.0
            continue
        if state == 0.0:
            if close[i] >= hi_entry[i]:
                state = 1.0
            elif close[i] <= lo_entry[i]:
                state = -1.0
        elif state == 1.0 and not np.isnan(lo_exit[i]) and close[i] <= lo_exit[i]:
            state = 0.0
        elif state == -1.0 and not np.isnan(hi_exit[i]) and close[i] >= hi_exit[i]:
            state = 0.0
        position[i] = state

    raw = pd.Series(position, index=df.index)
    return clean_positions(raw, df.index)


def dual_momentum(df: pd.DataFrame, lookback: int = 60, long_lookback: int = 240) -> pd.Series:
    """Dual momentum: long only if trailing return is positive (absolute) AND
    stronger than the asset's own longer-lookback trailing return (relative)."""
    short_return = df["close"] / df["close"].shift(lookback) - 1.0
    long_return = df["close"] / df["close"].shift(long_lookback) - 1.0
    raw = pd.Series(0.0, index=df.index)
    condition = (short_return > 0.0) & (short_return > long_return)
    raw = raw.where(~condition, 1.0)
    return clean_positions(raw, df.index)


def elder_ray(df: pd.DataFrame, ema_window: int = 13) -> pd.Series:
    """Elder Ray: long when bull power positive & bear power rising, and vice versa."""
    ema_line = ind.ema(df["close"], ema_window)
    bull_power = df["high"] - ema_line
    bear_power = df["low"] - ema_line
    raw = pd.Series(0.0, index=df.index)
    long_cond = (bull_power > 0.0) & (bear_power > bear_power.shift(1))
    short_cond = (bear_power < 0.0) & (bull_power < bull_power.shift(1))
    raw = raw.where(~long_cond, 1.0)
    raw = raw.where(~short_cond, -1.0)
    return clean_positions(raw, df.index)
