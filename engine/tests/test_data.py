"""Tests for the data layer: cache behavior and the OHLCV column contract."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from funnel.data.sources import OHLCV_COLUMNS, CachedSource


class _CountingStubSource:
    """A DataSource stub that counts how many times ``fetch`` is called."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.call_count = 0

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        self.call_count += 1
        return self._df


@pytest.fixture
def stub_df() -> pd.DataFrame:
    index = pd.date_range("2020-01-01", periods=5, freq="B")
    return pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0, 4.0, 5.0],
            "high": [1.5, 2.5, 3.5, 4.5, 5.5],
            "low": [0.5, 1.5, 2.5, 3.5, 4.5],
            "close": [1.2, 2.2, 3.2, 4.2, 5.2],
            "volume": [100.0, 200.0, 300.0, 400.0, 500.0],
        },
        index=index,
    )


def test_cached_source_hits_cache_on_second_fetch(stub_df: pd.DataFrame, tmp_path: Path) -> None:
    stub = _CountingStubSource(stub_df)
    cached = CachedSource(stub, cache_dir=tmp_path)

    start, end = date(2020, 1, 1), date(2020, 1, 10)
    first = cached.fetch("TEST", start, end)
    second = cached.fetch("TEST", start, end)

    assert stub.call_count == 1
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_cached_source_column_contract(stub_df: pd.DataFrame, tmp_path: Path) -> None:
    stub = _CountingStubSource(stub_df)
    cached = CachedSource(stub, cache_dir=tmp_path)

    result = cached.fetch("TEST", date(2020, 1, 1), date(2020, 1, 10))

    assert list(result.columns) == list(OHLCV_COLUMNS)
    assert isinstance(result.index, pd.DatetimeIndex)
    for col in OHLCV_COLUMNS:
        assert result[col].dtype == "float64"


def test_cached_source_different_ranges_are_different_keys(
    stub_df: pd.DataFrame, tmp_path: Path
) -> None:
    stub = _CountingStubSource(stub_df)
    cached = CachedSource(stub, cache_dir=tmp_path)

    cached.fetch("TEST", date(2020, 1, 1), date(2020, 1, 10))
    cached.fetch("TEST", date(2021, 1, 1), date(2021, 1, 10))

    assert stub.call_count == 2
