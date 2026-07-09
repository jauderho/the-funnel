"""PERF-2 numeric-parity tests for the four vectorized indicator families in
``funnel.strategies.indicators``: ``linreg_slope``, ``aroon``, ``hull_ma``
(via its ``_wma`` building block), and ``connors_rsi`` (its streak +
percent-rank parts).

Each family's *original* ``rolling().apply()``-based implementation is
copied verbatim below as a reference (ground truth, not memory) and
compared against the real, now-vectorized module function across all three
shared synthetic fixtures (``trending_ohlcv``, ``mean_reverting_ohlcv``,
``flat_ohlcv`` from ``conftest.py``), at ``rtol=1e-9``.
"""

import numpy as np
import pandas as pd
import pytest

from funnel.strategies.indicators import aroon, connors_rsi, hull_ma, linreg_slope, rsi

FIXTURE_NAMES = ("trending_ohlcv", "mean_reverting_ohlcv", "flat_ohlcv")


# ---------------------------------------------------------------------------
# Reference implementations -- verbatim copies of the pre-vectorization code.
# ---------------------------------------------------------------------------


def _linreg_slope_reference(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=np.float64)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _slope(y: np.ndarray) -> float:
        return float(((y - y.mean()) * (x - x_mean)).sum() / x_var)

    return series.rolling(window).apply(_slope, raw=True)


def _aroon_reference(df: pd.DataFrame, window: int = 25) -> tuple[pd.Series, pd.Series]:
    def _periods_since_max(x: np.ndarray) -> float:
        return float(len(x) - 1 - np.argmax(x))

    def _periods_since_min(x: np.ndarray) -> float:
        return float(len(x) - 1 - np.argmin(x))

    since_high = df["high"].rolling(window + 1).apply(_periods_since_max, raw=True)
    since_low = df["low"].rolling(window + 1).apply(_periods_since_min, raw=True)
    aroon_up = 100.0 * (window - since_high) / window
    aroon_down = 100.0 * (window - since_low) / window
    return aroon_up, aroon_down


def _wma_reference(series: pd.Series, window: int) -> pd.Series:
    weights = np.arange(1, window + 1, dtype=np.float64)

    def _weighted(x: np.ndarray) -> float:
        return float(np.dot(x, weights) / weights.sum())

    return series.rolling(window).apply(_weighted, raw=True)


def _hull_ma_reference(series: pd.Series, window: int = 20) -> pd.Series:
    half = max(int(window / 2), 1)
    sqrt_window = max(int(np.sqrt(window)), 1)
    wma_half = _wma_reference(series, half)
    wma_full = _wma_reference(series, window)
    raw = 2.0 * wma_half - wma_full
    return _wma_reference(raw, sqrt_window)


def _connors_rsi_reference(
    series: pd.Series, rsi_window: int = 3, streak_window: int = 2, rank_window: int = 100
) -> pd.Series:
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


# ---------------------------------------------------------------------------
# Parity tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
@pytest.mark.parametrize("window", [5, 10, 20, 50])
def test_linreg_slope_parity(
    request: pytest.FixtureRequest, fixture_name: str, window: int
) -> None:
    df: pd.DataFrame = request.getfixturevalue(fixture_name)
    expected = _linreg_slope_reference(df["close"], window)
    actual = linreg_slope(df["close"], window)
    np.testing.assert_allclose(
        actual.to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True
    )


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
@pytest.mark.parametrize("window", [5, 25, 40])
def test_aroon_parity(request: pytest.FixtureRequest, fixture_name: str, window: int) -> None:
    df: pd.DataFrame = request.getfixturevalue(fixture_name)
    expected_up, expected_down = _aroon_reference(df, window)
    actual_up, actual_down = aroon(df, window)
    np.testing.assert_allclose(
        actual_up.to_numpy(), expected_up.to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True
    )
    np.testing.assert_allclose(
        actual_down.to_numpy(), expected_down.to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True
    )


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
@pytest.mark.parametrize("window", [9, 20, 34])
def test_hull_ma_parity(request: pytest.FixtureRequest, fixture_name: str, window: int) -> None:
    df: pd.DataFrame = request.getfixturevalue(fixture_name)
    expected = _hull_ma_reference(df["close"], window)
    actual = hull_ma(df["close"], window)
    np.testing.assert_allclose(
        actual.to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True
    )


@pytest.mark.parametrize("fixture_name", FIXTURE_NAMES)
@pytest.mark.parametrize(
    "rsi_window,streak_window,rank_window", [(3, 2, 100), (2, 3, 20), (5, 2, 50)]
)
def test_connors_rsi_parity(
    request: pytest.FixtureRequest,
    fixture_name: str,
    rsi_window: int,
    streak_window: int,
    rank_window: int,
) -> None:
    df: pd.DataFrame = request.getfixturevalue(fixture_name)
    expected = _connors_rsi_reference(df["close"], rsi_window, streak_window, rank_window)
    actual = connors_rsi(df["close"], rsi_window, streak_window, rank_window)
    np.testing.assert_allclose(
        actual.to_numpy(), expected.to_numpy(), rtol=1e-9, atol=1e-9, equal_nan=True
    )
