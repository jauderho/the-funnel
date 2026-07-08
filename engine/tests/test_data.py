"""Tests for the data layer: cache behavior and the OHLCV column contract."""

from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yfinance

import funnel.data.sources as sources_module
from funnel.data.sources import OHLCV_COLUMNS, CachedSource, YFinanceSource


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


# --- YFinanceSource csrf-cookie-strategy fallback -------------------------
#
# yfinance 1.4.1's default "basic" cookie strategy can silently return empty
# frames on networks where ``fc.yahoo.com`` is unreachable. ``YFinanceSource``
# retries once with the "csrf" cookie strategy when the first attempt comes
# back empty. These tests stub out ``yfinance.download`` and the strategy
# switch helper entirely -- no network access.


@pytest.fixture(autouse=True)
def _reset_csrf_flag() -> Iterator[None]:
    """Reset the module-level once-per-process fallback flag between tests."""
    sources_module._csrf_strategy_active = False
    yield
    sources_module._csrf_strategy_active = False


def _raw_yf_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build a frame shaped like a raw (un-normalized) yfinance download."""
    return df.rename(columns=lambda c: str(c).title())


def _empty_raw_yf_frame() -> pd.DataFrame:
    """A frame shaped like yfinance's real "no data" response.

    yfinance still returns the expected OHLCV columns (title-cased) with a
    ``DatetimeIndex``, just zero rows -- not a bare ``pd.DataFrame()``.
    """
    df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df.index = pd.DatetimeIndex([])
    return df


def test_fetch_falls_back_to_csrf_strategy_on_empty_first_attempt(
    stub_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"download": 0, "switch": 0}

    def fake_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls["download"] += 1
        if calls["download"] == 1:
            return _empty_raw_yf_frame()
        return _raw_yf_frame(stub_df)

    def fake_switch() -> bool:
        calls["switch"] += 1
        sources_module._csrf_strategy_active = True
        return True

    monkeypatch.setattr(yfinance, "download", fake_download)
    monkeypatch.setattr(sources_module, "_switch_to_csrf_cookie_strategy", fake_switch)

    result = YFinanceSource().fetch("SPY", date(2020, 1, 1), date(2020, 1, 10))

    assert calls["switch"] == 1
    assert calls["download"] == 2
    assert not result.empty
    assert list(result.columns) == list(OHLCV_COLUMNS)


def test_fetch_no_fallback_when_first_attempt_succeeds(
    stub_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = {"download": 0, "switch": 0}

    def fake_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls["download"] += 1
        return _raw_yf_frame(stub_df)

    def fake_switch() -> bool:
        calls["switch"] += 1
        return True

    monkeypatch.setattr(yfinance, "download", fake_download)
    monkeypatch.setattr(sources_module, "_switch_to_csrf_cookie_strategy", fake_switch)

    result = YFinanceSource().fetch("SPY", date(2020, 1, 1), date(2020, 1, 10))

    assert calls["switch"] == 0
    assert calls["download"] == 1
    assert not result.empty


def test_fetch_returns_empty_when_both_attempts_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"download": 0, "switch": 0}

    def fake_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls["download"] += 1
        return _empty_raw_yf_frame()

    def fake_switch() -> bool:
        calls["switch"] += 1
        sources_module._csrf_strategy_active = True
        return True

    monkeypatch.setattr(yfinance, "download", fake_download)
    monkeypatch.setattr(sources_module, "_switch_to_csrf_cookie_strategy", fake_switch)

    result = YFinanceSource().fetch("SPY", date(2020, 1, 1), date(2020, 1, 10))

    assert calls["switch"] == 1
    assert calls["download"] == 2
    assert result.empty


def test_fetch_handles_strategy_switch_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates a future yfinance release renaming/removing the private API.

    ``_switch_to_csrf_cookie_strategy`` is expected to catch the error itself
    (there is no public replacement to fall back to), log a warning, and
    report failure -- not propagate the exception up through ``fetch``.
    """
    calls = {"download": 0}

    class _RaisingYfData:
        def _set_cookie_strategy(self, strategy: str) -> None:
            raise AttributeError("_set_cookie_strategy was renamed")

    def fake_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls["download"] += 1
        return _empty_raw_yf_frame()

    monkeypatch.setattr(yfinance, "download", fake_download)
    monkeypatch.setattr("yfinance.data.YfData", _RaisingYfData)

    result = YFinanceSource().fetch("SPY", date(2020, 1, 1), date(2020, 1, 10))

    assert calls["download"] == 1
    assert result.empty
    assert sources_module._csrf_strategy_active is False


def test_fetch_does_not_reswitch_once_strategy_already_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources_module._csrf_strategy_active = True
    calls = {"download": 0, "switch": 0}

    def fake_download(*args: Any, **kwargs: Any) -> pd.DataFrame:
        calls["download"] += 1
        return _empty_raw_yf_frame()

    def fake_switch() -> bool:
        calls["switch"] += 1
        return True

    monkeypatch.setattr(yfinance, "download", fake_download)
    monkeypatch.setattr(sources_module, "_switch_to_csrf_cookie_strategy", fake_switch)

    result = YFinanceSource().fetch("SPY", date(2020, 1, 1), date(2020, 1, 10))

    assert calls["switch"] == 0
    assert calls["download"] == 1
    assert result.empty
