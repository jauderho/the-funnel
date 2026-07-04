"""Shared technical indicators, implemented by hand with pandas/numpy.

Every function here is causal: each output at row ``t`` depends only on
rows ``<= t`` of its inputs. No indicator uses ``center=True``, negative
``shift``, or whole-series statistics. See ``funnel.strategies.base`` for
the no-look-ahead contract these feed into.
"""

import numpy as np
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=span, adjust=False).mean()


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range: max of (high-low, |high-prev_close|, |low-prev_close|)."""
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average true range via Wilder-style rolling mean of true range."""
    return true_range(df).rolling(window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative strength index (Wilder smoothing via EWM)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    return result.where(avg_loss != 0.0, 100.0)


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling mean) / rolling std."""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def rolling_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    """Rolling (not anchored/global) volume-weighted average price."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical * df["volume"]
    return pv.rolling(window).sum() / df["volume"].rolling(window).sum()


def bollinger_bands(
    series: pd.Series, window: int = 20, n_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger bands: (middle, upper, lower)."""
    middle = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = middle + n_std * std
    lower = middle - n_std * std
    return middle, upper, lower


def keltner_channels(
    df: pd.DataFrame, ema_window: int = 20, atr_window: int = 10, mult: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Keltner channels: (middle EMA, upper, lower), width via ATR multiple."""
    middle = ema(df["close"], ema_window)
    band = mult * atr(df, atr_window)
    return middle, middle + band, middle - band


def directional_movement(
    df: pd.DataFrame, window: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder's +DI, -DI, and ADX."""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0), index=df.index
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0), index=df.index
    )
    tr = true_range(df)
    atr_smoothed = tr.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    plus_di = (
        100.0
        * plus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
        / atr_smoothed.replace(0.0, np.nan)
    )
    minus_di = (
        100.0
        * minus_dm.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
        / atr_smoothed.replace(0.0, np.nan)
    )
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    return plus_di, minus_di, adx


def stochastic(
    df: pd.DataFrame, k_window: int = 14, d_window: int = 3
) -> tuple[pd.Series, pd.Series]:
    """Stochastic oscillator: (%K, %D)."""
    lowest_low = df["low"].rolling(k_window).min()
    highest_high = df["high"].rolling(k_window).max()
    denom = (highest_high - lowest_low).replace(0.0, np.nan)
    percent_k = 100.0 * (df["close"] - lowest_low) / denom
    percent_d = percent_k.rolling(d_window).mean()
    return percent_k, percent_d


def cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    sma_typical = typical.rolling(window).mean()
    mean_dev = (typical - sma_typical).abs().rolling(window).mean()
    return (typical - sma_typical) / (0.015 * mean_dev.replace(0.0, np.nan))


def williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Williams %R."""
    highest_high = df["high"].rolling(window).max()
    lowest_low = df["low"].rolling(window).min()
    denom = (highest_high - lowest_low).replace(0.0, np.nan)
    return -100.0 * (highest_high - df["close"]) / denom


def money_flow_index(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Money Flow Index."""
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    raw_flow = typical * df["volume"]
    up = typical.diff() > 0.0
    down = typical.diff() < 0.0
    pos_flow = raw_flow.where(up, 0.0).rolling(window).sum()
    neg_flow = raw_flow.where(down, 0.0).rolling(window).sum()
    money_ratio = pos_flow / neg_flow.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + money_ratio))


def chaikin_money_flow(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    denom = (df["high"] - df["low"]).replace(0.0, np.nan)
    mf_multiplier = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / denom
    mf_volume = mf_multiplier * df["volume"]
    return mf_volume.rolling(window).sum() / df["volume"].rolling(window).sum()


def obv(df: pd.DataFrame) -> pd.Series:
    """On-balance volume."""
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def force_index(df: pd.DataFrame, window: int = 13) -> pd.Series:
    """Force Index: price change x volume, smoothed by an EMA."""
    raw = df["close"].diff() * df["volume"]
    return ema(raw, window)


def aroon(df: pd.DataFrame, window: int = 25) -> tuple[pd.Series, pd.Series]:
    """Aroon up/down: how recently the window high/low occurred."""

    def _periods_since_max(x: np.ndarray) -> float:
        return float(len(x) - 1 - np.argmax(x))

    def _periods_since_min(x: np.ndarray) -> float:
        return float(len(x) - 1 - np.argmin(x))

    since_high = df["high"].rolling(window + 1).apply(_periods_since_max, raw=True)
    since_low = df["low"].rolling(window + 1).apply(_periods_since_min, raw=True)
    aroon_up = 100.0 * (window - since_high) / window
    aroon_down = 100.0 * (window - since_low) / window
    return aroon_up, aroon_down


def vortex(df: pd.DataFrame, window: int = 14) -> tuple[pd.Series, pd.Series]:
    """Vortex indicator: (+VI, -VI)."""
    prev_close = df["close"].shift(1)
    prev_low = df["low"].shift(1)
    prev_high = df["high"].shift(1)
    vm_plus = (df["high"] - prev_low).abs()
    vm_minus = (df["low"] - prev_high).abs()
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    tr_sum = tr.rolling(window).sum().replace(0.0, np.nan)
    vi_plus = vm_plus.rolling(window).sum() / tr_sum
    vi_minus = vm_minus.rolling(window).sum() / tr_sum
    return vi_plus, vi_minus


def trix(series: pd.Series, window: int = 15) -> pd.Series:
    """TRIX: rate of change of a triple-smoothed EMA."""
    triple_ema = ema(ema(ema(series, window), window), window)
    return triple_ema.pct_change() * 100.0


def hull_ma(series: pd.Series, window: int = 20) -> pd.Series:
    """Hull moving average: reduces lag vs a simple/weighted MA."""
    half = max(int(window / 2), 1)
    sqrt_window = max(int(np.sqrt(window)), 1)
    wma_half = _wma(series, half)
    wma_full = _wma(series, window)
    raw = 2.0 * wma_half - wma_full
    return _wma(raw, sqrt_window)


def _wma(series: pd.Series, window: int) -> pd.Series:
    """Linearly weighted moving average (most recent bar weighted highest)."""
    weights = np.arange(1, window + 1, dtype=np.float64)

    def _weighted(x: np.ndarray) -> float:
        return float(np.dot(x, weights) / weights.sum())

    return series.rolling(window).apply(_weighted, raw=True)


def kama(series: pd.Series, window: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman's Adaptive Moving Average."""
    change = (series - series.shift(window)).abs()
    volatility = series.diff().abs().rolling(window).sum().replace(0.0, np.nan)
    efficiency_ratio = (change / volatility).fillna(0.0)
    fast_sc = 2.0 / (fast + 1)
    slow_sc = 2.0 / (slow + 1)
    smoothing_constant = (efficiency_ratio * (fast_sc - slow_sc) + slow_sc) ** 2

    values = series.to_numpy(dtype=np.float64)
    sc = smoothing_constant.to_numpy(dtype=np.float64)
    result = np.full(len(values), np.nan, dtype=np.float64)
    first_valid = window
    if len(values) <= first_valid:
        return pd.Series(result, index=series.index)
    result[first_valid] = values[first_valid]
    for i in range(first_valid + 1, len(values)):
        prev = result[i - 1]
        if np.isnan(prev):
            result[i] = values[i]
        else:
            result[i] = prev + sc[i] * (values[i] - prev)
    return pd.Series(result, index=series.index)


def linreg_slope(series: pd.Series, window: int) -> pd.Series:
    """Rolling linear-regression slope (per-bar, ordinary least squares)."""
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        return float(((y - y.mean()) * (x - x_mean)).sum() / x_var)

    return series.rolling(window).apply(_slope, raw=True)


def ultimate_oscillator(
    df: pd.DataFrame, window1: int = 7, window2: int = 14, window3: int = 28
) -> pd.Series:
    """Ultimate Oscillator: weighted blend of three buying-pressure lookbacks."""
    prev_close = df["close"].shift(1)
    bp = df["close"] - pd.concat([df["low"], prev_close], axis=1).min(axis=1)
    tr = pd.concat([df["high"], prev_close], axis=1).max(axis=1) - pd.concat(
        [df["low"], prev_close], axis=1
    ).min(axis=1)
    tr = tr.replace(0.0, np.nan)
    avg1 = bp.rolling(window1).sum() / tr.rolling(window1).sum()
    avg2 = bp.rolling(window2).sum() / tr.rolling(window2).sum()
    avg3 = bp.rolling(window3).sum() / tr.rolling(window3).sum()
    return 100.0 * (4.0 * avg1 + 2.0 * avg2 + avg3) / 7.0


def connors_rsi(
    series: pd.Series, rsi_window: int = 3, streak_window: int = 2, rank_window: int = 100
) -> pd.Series:
    """Connors RSI: blend of short RSI, streak-length RSI, and percent-rank of return."""
    price_rsi = rsi(series, rsi_window)

    diff = series.diff()
    streak_values = np.zeros(len(series), dtype=np.float64)
    diff_values = diff.to_numpy(dtype=np.float64)
    for i in range(1, len(diff_values)):
        d = diff_values[i]
        prev = streak_values[i - 1]
        if d > 0.0:
            streak_values[i] = prev + 1.0 if prev > 0.0 else 1.0
        elif d < 0.0:
            streak_values[i] = prev - 1.0 if prev < 0.0 else -1.0
        else:
            streak_values[i] = 0.0
    streak = pd.Series(streak_values, index=series.index)
    streak_rsi = rsi(streak, streak_window)

    pct_change = series.pct_change()

    def _percent_rank(x: np.ndarray) -> float:
        last = x[-1]
        return float(100.0 * (x < last).sum() / (len(x) - 1)) if len(x) > 1 else 0.0

    percent_rank = pct_change.rolling(rank_window).apply(_percent_rank, raw=True)

    return (price_rsi + streak_rsi + percent_rank) / 3.0


def pivot_points(df: pd.DataFrame, window: int = 5) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Rolling pivot level and support/resistance (classic floor-trader formula).

    Uses the prior ``window``-bar high/low/close (shifted by 1 to avoid
    including the current bar) to define a rolling pivot, support, and
    resistance level.
    """
    prior_high = df["high"].shift(1).rolling(window).max()
    prior_low = df["low"].shift(1).rolling(window).min()
    prior_close = df["close"].shift(1)
    pivot = (prior_high + prior_low + prior_close) / 3.0
    support = 2.0 * pivot - prior_high
    resistance = 2.0 * pivot - prior_low
    return pivot, support, resistance
