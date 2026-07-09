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
    """Aroon up/down: how recently the window high/low occurred.

    Vectorized via ``sliding_window_view`` + ``argmax``/``argmin`` instead
    of ``rolling().apply()`` (PERF-2): "periods since the window's high/low"
    is exactly ``len(window) - 1 - argmax/argmin(window)``, the original
    callback's own formula, computed once over the whole array instead of
    once per Python-level callback invocation. Unlike a pure linear
    combination (e.g. ``linreg_slope``), ``argmax``/``argmin`` do not
    naturally propagate NaN the way pandas' ``min_periods == window``
    default would, so windows containing any NaN are explicitly masked to
    NaN to match.
    """
    win = window + 1
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)
    n = len(high)
    since_high = np.full(n, np.nan, dtype=np.float64)
    since_low = np.full(n, np.nan, dtype=np.float64)

    if n >= win:
        high_windows = np.lib.stride_tricks.sliding_window_view(high, win)
        low_windows = np.lib.stride_tricks.sliding_window_view(low, win)
        valid_high = ~np.isnan(high_windows).any(axis=1)
        valid_low = ~np.isnan(low_windows).any(axis=1)
        raw_since_high = (win - 1) - np.argmax(high_windows, axis=-1)
        raw_since_low = (win - 1) - np.argmin(low_windows, axis=-1)
        since_high[win - 1 :] = np.where(valid_high, raw_since_high, np.nan)
        since_low[win - 1 :] = np.where(valid_low, raw_since_low, np.nan)

    aroon_up = 100.0 * (window - since_high) / window
    aroon_down = 100.0 * (window - since_low) / window
    return (
        pd.Series(aroon_up, index=df.index),
        pd.Series(aroon_down, index=df.index),
    )


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
    """Linearly weighted moving average (most recent bar weighted highest).

    Vectorized via ``sliding_window_view`` + matrix multiply instead of
    ``rolling().apply()`` (PERF-2), feeding ``hull_ma``. All weights are
    strictly positive, so a NaN anywhere in a window always propagates to a
    NaN output, matching ``rolling(window).apply``'s default
    ``min_periods == window`` behavior.
    """
    weights = np.arange(1, window + 1, dtype=np.float64)
    weight_sum = weights.sum()

    values = series.to_numpy(dtype=np.float64)
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return pd.Series(out, index=series.index)

    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    out[window - 1 :] = windows @ weights / weight_sum
    return pd.Series(out, index=series.index)


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
    """Rolling linear-regression slope (per-bar, ordinary least squares).

    Vectorized via ``numpy.lib.stride_tricks.sliding_window_view`` instead
    of ``rolling().apply()`` (PERF-2: measured 88-205x faster in isolation,
    1e-9 numeric parity). Algebraically identical to the OLS slope formula:
    ``sum((y - ybar) * (x - xbar)) == sum((x - xbar) * y)`` since the
    ``x - xbar`` terms sum to zero, so ``ybar`` need not be computed at all.
    A window containing any NaN naturally propagates to a NaN output (NaN
    times any weight, including zero, is NaN), matching
    ``rolling(window).apply``'s default ``min_periods == window`` behavior
    (a window is only ever evaluated when it is entirely non-null).
    """
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    weights = x - x_mean

    values = series.to_numpy(dtype=np.float64)
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if n < window:
        return pd.Series(out, index=series.index)

    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    out[window - 1 :] = windows @ weights / x_var
    return pd.Series(out, index=series.index)


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


def _streak(diff: pd.Series) -> pd.Series:
    """Signed length of the current same-direction consecutive-change streak.

    Vectorized via a group-by-run-id trick instead of a per-row Python loop
    (PERF-2): consecutive same-sign (nonzero) diffs increment a running
    count in that sign's direction; a sign change (including to/from zero)
    resets the count to +-1. A zero (or NaN, treated as zero -- matching the
    original loop's ``else`` branch) diff is always streak 0. Row 0 is
    always streak 0 (there is no prior diff to compare against), matching
    the original loop which never touches index 0.
    """
    d = diff.to_numpy(dtype=np.float64)
    n = len(d)
    sign = np.zeros(n, dtype=np.float64)
    if n > 1:
        sign[1:] = np.nan_to_num(np.sign(d[1:]), nan=0.0)
    change = np.ones(n, dtype=bool)
    if n > 1:
        change[1:] = sign[1:] != sign[:-1]
    group_id = np.cumsum(change)
    run_pos = pd.Series(np.ones(n)).groupby(group_id).cumsum().to_numpy()
    streak = np.where(sign == 0.0, 0.0, run_pos * sign)
    streak[0] = 0.0
    return pd.Series(streak, index=diff.index)


def _rolling_percent_rank(series: pd.Series, window: int) -> pd.Series:
    """Rolling percent-rank of a window's last value among all values in that window.

    Vectorized via ``sliding_window_view`` instead of ``rolling().apply()``
    (PERF-2): counts, for each window, how many of its values are strictly
    less than the window's own last value (the last value never counts
    against itself, since ``x < x`` is always False). Windows containing
    any NaN are explicitly masked to NaN (comparisons against/with NaN are
    always False, so they would otherwise silently undercount rather than
    propagate), matching ``rolling(window).apply``'s default
    ``min_periods == window`` behavior.
    """
    values = series.to_numpy(dtype=np.float64)
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float64)
    if window <= 1:
        # Matches the original callback's explicit `len(x) > 1` guard: a
        # single-point window is always rank 0.0, unless that point itself
        # is NaN (pandas' min_periods=1 then withholds the NaN result
        # without calling the callback at all).
        out = np.where(np.isnan(values), np.nan, 0.0)
        return pd.Series(out, index=series.index)
    if n < window:
        return pd.Series(out, index=series.index)

    windows = np.lib.stride_tricks.sliding_window_view(values, window)
    last = windows[:, -1:]
    valid = ~np.isnan(windows).any(axis=1)
    count_less = (windows < last).sum(axis=1)
    result = 100.0 * count_less / (window - 1)
    out[window - 1 :] = np.where(valid, result, np.nan)
    return pd.Series(out, index=series.index)


def connors_rsi(
    series: pd.Series, rsi_window: int = 3, streak_window: int = 2, rank_window: int = 100
) -> pd.Series:
    """Connors RSI: blend of short RSI, streak-length RSI, and percent-rank of return."""
    price_rsi = rsi(series, rsi_window)

    streak = _streak(series.diff())
    streak_rsi = rsi(streak, streak_window)

    pct_change = series.pct_change()
    percent_rank = _rolling_percent_rank(pct_change, rank_window)

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
