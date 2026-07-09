"""Tests for funnel.pipeline._fetch_all: PERF-2 threaded data-stage fetch.

Verifies the thread pool actually overlaps I/O-bound fetches, preserves
input-order dict assembly regardless of completion order, and propagates a
single failing symbol's exception exactly as a sequential loop would.
"""

import threading
import time
from datetime import date

import pandas as pd
import pytest

from funnel.pipeline import _fetch_all


class _RecordingSource:
    """Records concurrently-in-flight calls and returns a per-symbol frame."""

    def __init__(self, latency_s: float = 0.05) -> None:
        self._latency_s = latency_s
        self._lock = threading.Lock()
        self._in_flight = 0
        self.max_concurrent = 0
        self.call_order: list[str] = []

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        with self._lock:
            self._in_flight += 1
            self.max_concurrent = max(self.max_concurrent, self._in_flight)
            self.call_order.append(symbol)
        time.sleep(self._latency_s)
        with self._lock:
            self._in_flight -= 1
        return pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex([pd.Timestamp("2020-01-01")]))


def test_fetch_all_overlaps_calls_concurrently() -> None:
    source = _RecordingSource(latency_s=0.1)
    symbols = [f"SYM{i}" for i in range(6)]

    t0 = time.perf_counter()
    result = _fetch_all(source, symbols, date(2020, 1, 1), date(2020, 12, 31))
    elapsed = time.perf_counter() - t0

    assert set(result.keys()) == set(symbols)
    assert source.max_concurrent > 1
    # 6 symbols x 0.1s sequentially would take >=0.6s; overlapped with
    # max_workers=8 they should all run essentially at once.
    assert elapsed < 0.5


def test_fetch_all_preserves_input_order_regardless_of_completion_order() -> None:
    class _VariableLatencySource:
        def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
            # Reverse-order latency: the first-submitted symbol finishes last.
            delays = {"AAA": 0.08, "BBB": 0.04, "CCC": 0.0}
            time.sleep(delays[symbol])
            return pd.DataFrame({"close": [float(len(symbol))]})

    symbols = ["AAA", "BBB", "CCC"]
    result = _fetch_all(_VariableLatencySource(), symbols, date(2020, 1, 1), date(2020, 12, 31))

    assert list(result.keys()) == symbols


def test_fetch_all_propagates_exception_from_failing_symbol() -> None:
    class _FailingSource:
        def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
            if symbol == "BAD":
                raise ValueError(f"no data for {symbol}")
            return pd.DataFrame({"close": [1.0]})

    with pytest.raises(ValueError, match="no data for BAD"):
        _fetch_all(_FailingSource(), ["GOOD", "BAD"], date(2020, 1, 1), date(2020, 12, 31))
