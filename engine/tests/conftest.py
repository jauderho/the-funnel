"""Shared synthetic OHLCV fixtures for engine tests.

All fixtures are deterministic (fixed numpy seed) and network-free.
"""

import numpy as np
import pandas as pd
import pytest

N_ROWS = 600


def _to_ohlcv(index: pd.DatetimeIndex, close: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Build a plausible OHLCV frame around a given close-price path."""
    daily_range = np.abs(rng.normal(loc=0.5, scale=0.2, size=len(close))) + 0.05
    open_ = close + rng.normal(loc=0.0, scale=0.1, size=len(close))
    high = np.maximum(open_, close) + daily_range
    low = np.minimum(open_, close) - daily_range
    volume = np.abs(rng.normal(loc=1_000_000.0, scale=200_000.0, size=len(close)))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    ).astype("float64")


@pytest.fixture
def trending_ohlcv() -> pd.DataFrame:
    """A steadily uptrending series with mild noise."""
    rng = np.random.default_rng(42)
    index = pd.bdate_range("2020-01-01", periods=N_ROWS)
    drift = np.linspace(0.0, 150.0, N_ROWS)
    noise = np.cumsum(rng.normal(loc=0.0, scale=0.5, size=N_ROWS))
    close = 100.0 + drift + noise
    return _to_ohlcv(index, close, rng)


@pytest.fixture
def mean_reverting_ohlcv() -> pd.DataFrame:
    """A sine wave plus noise: oscillates around a fixed level."""
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2020-01-01", periods=N_ROWS)
    t = np.arange(N_ROWS)
    wave = 15.0 * np.sin(2 * np.pi * t / 40.0)
    noise = rng.normal(loc=0.0, scale=1.0, size=N_ROWS)
    close = 100.0 + wave + noise
    return _to_ohlcv(index, close, rng)


@pytest.fixture
def flat_ohlcv() -> pd.DataFrame:
    """A flat series with only tiny noise."""
    rng = np.random.default_rng(99)
    index = pd.bdate_range("2020-01-01", periods=N_ROWS)
    noise = rng.normal(loc=0.0, scale=0.05, size=N_ROWS)
    close = 100.0 + noise
    return _to_ohlcv(index, close, rng)
