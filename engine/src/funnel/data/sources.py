"""Data source protocol, yfinance-backed source, and a parquet-backed cache.

The ``DataSource`` contract is the single interface every strategy, the
backtest engine, and the caching layer rely on: a call to
``fetch(symbol, start, end)`` returns a ``pandas.DataFrame`` indexed by a
``DatetimeIndex`` (ascending, no duplicates) with exactly these columns, all
``float64``:

    open, high, low, close, volume

Column names are lowercase. No other columns are guaranteed to be present.
Implementations must drop rows with a missing ``close`` (a bar with no
close price is not usable by any strategy or the backtest engine).
"""

import logging
import os
from datetime import date
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")

# Set once a switch to the "csrf" cookie strategy has succeeded for this
# process. ``yfinance.data.YfData`` is a process-wide singleton, so a
# successful switch benefits every subsequent ``YFinanceSource.fetch`` call
# without needing to switch (or retry) again.
_csrf_strategy_active = False


@runtime_checkable
class DataSource(Protocol):
    """Contract for anything that can fetch OHLCV bars for a symbol."""

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """Return OHLCV bars for ``symbol`` in ``[start, end]``.

        The returned frame is indexed by an ascending ``DatetimeIndex`` and
        has columns ``open, high, low, close, volume`` (float64, lowercase).
        Rows with a missing ``close`` are dropped.
        """
        ...


class YFinanceSource:
    """Fetches daily OHLCV bars from Yahoo Finance via the ``yfinance`` package."""

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        df = _normalize(_download(symbol, start, end))
        if not df.empty:
            return df

        global _csrf_strategy_active
        if not _csrf_strategy_active:
            logger.warning(
                "yfinance returned an empty frame for %s; falling back to the "
                "'csrf' cookie strategy and retrying once",
                symbol,
            )
            _csrf_strategy_active = _switch_to_csrf_cookie_strategy()
            if _csrf_strategy_active:
                df = _normalize(_download(symbol, start, end))
                if not df.empty:
                    logger.info("csrf cookie strategy fallback succeeded for %s", symbol)
        return df


def _download(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Call ``yf.download`` with the options this source always uses."""
    import yfinance as yf

    return yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        multi_level_index=False,
    )


def _switch_to_csrf_cookie_strategy() -> bool:
    """Switch the process-wide ``YfData`` singleton to the "csrf" strategy.

    yfinance 1.4.1's default "basic" cookie strategy bootstraps a cookie from
    ``fc.yahoo.com``. On some networks that host is connection-refused while
    the real data hosts (``query1``/``query2.finance.yahoo.com``) are
    reachable, and yfinance errors out per-ticker instead of falling back,
    so ``yf.download`` silently returns an empty frame. Switching to the
    "csrf" strategy (which uses the consent flow instead of ``fc.yahoo.com``)
    is a confirmed workaround, but ``_set_cookie_strategy`` is a private API
    with no public equivalent in this version, so calls to it are isolated
    here and guarded: if a future yfinance release renames or removes it,
    we log a warning and skip the retry rather than crash.

    Returns ``True`` if the switch succeeded, ``False`` otherwise.
    """
    try:
        from yfinance.data import YfData

        YfData()._set_cookie_strategy("csrf")  # noqa: SLF001
    except Exception:
        logger.warning(
            "failed to switch yfinance to the 'csrf' cookie strategy; skipping retry",
            exc_info=True,
        )
        return False
    return True


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize a raw yfinance frame to the OHLCV contract."""
    df = raw.rename(columns=lambda c: str(c).lower())
    df = df[[c for c in OHLCV_COLUMNS if c in df.columns]]
    df = df.dropna(subset=["close"])
    df = df.astype({c: "float64" for c in df.columns})
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df = df.sort_index()
    return df


def default_cache_dir() -> Path:
    """Resolve the parquet cache directory.

    Honors ``FUNNEL_DATA_DIR`` if set; otherwise defaults to
    ``<repo>/data/cache`` (relative to this file's location in
    ``engine/src/funnel/data/sources.py``).
    """
    override = os.environ.get("FUNNEL_DATA_DIR")
    if override:
        return Path(override)
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "data" / "cache"


class CachedSource:
    """Wraps a ``DataSource`` with an on-disk parquet cache.

    The cache key is ``symbol`` + ``start`` + ``end``; a hit skips the
    wrapped source entirely, a miss fetches once and writes the parquet
    file before returning.
    """

    def __init__(self, source: DataSource, cache_dir: Path | None = None) -> None:
        self._source = source
        self._cache_dir = cache_dir if cache_dir is not None else default_cache_dir()

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        path = self._cache_path(symbol, start, end)
        if path.exists():
            return pd.read_parquet(path)

        df = self._source.fetch(symbol, start, end)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)
        return df

    def _cache_path(self, symbol: str, start: date, end: date) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return self._cache_dir / f"{safe_symbol}_{start.isoformat()}_{end.isoformat()}.parquet"
