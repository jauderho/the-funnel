"""The default asset universe and history-based filtering."""

import logging
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import pandas as pd

from funnel.data.sources import DataSource

logger = logging.getLogger(__name__)


class AssetClass(StrEnum):
    """Broad asset-class buckets, used downstream by the cost model."""

    INDEX_ETF = "index_etf"
    SECTOR_ETF = "sector_etf"
    COMMODITY = "commodity"
    RATES_CREDIT = "rates_credit"
    CRYPTO = "crypto"
    LARGE_CAP = "large_cap"


@dataclass(slots=True, frozen=True)
class AssetSpec:
    """A single universe member: its symbol and asset class."""

    symbol: str
    asset_class: AssetClass


ASSET_UNIVERSE: tuple[AssetSpec, ...] = (
    # Index ETFs
    AssetSpec("SPY", AssetClass.INDEX_ETF),
    AssetSpec("QQQ", AssetClass.INDEX_ETF),
    AssetSpec("IWM", AssetClass.INDEX_ETF),
    AssetSpec("DIA", AssetClass.INDEX_ETF),
    # Sector ETFs
    AssetSpec("XLK", AssetClass.SECTOR_ETF),
    AssetSpec("XLF", AssetClass.SECTOR_ETF),
    AssetSpec("XLE", AssetClass.SECTOR_ETF),
    AssetSpec("XLV", AssetClass.SECTOR_ETF),
    AssetSpec("XLI", AssetClass.SECTOR_ETF),
    AssetSpec("XLP", AssetClass.SECTOR_ETF),
    AssetSpec("XLY", AssetClass.SECTOR_ETF),
    AssetSpec("XLU", AssetClass.SECTOR_ETF),
    AssetSpec("XLB", AssetClass.SECTOR_ETF),
    # Commodities
    AssetSpec("GLD", AssetClass.COMMODITY),
    AssetSpec("SLV", AssetClass.COMMODITY),
    AssetSpec("USO", AssetClass.COMMODITY),
    AssetSpec("DBA", AssetClass.COMMODITY),
    # Rates / credit
    AssetSpec("TLT", AssetClass.RATES_CREDIT),
    AssetSpec("IEF", AssetClass.RATES_CREDIT),
    AssetSpec("HYG", AssetClass.RATES_CREDIT),
    AssetSpec("LQD", AssetClass.RATES_CREDIT),
    # Crypto
    AssetSpec("BTC-USD", AssetClass.CRYPTO),
    AssetSpec("ETH-USD", AssetClass.CRYPTO),
    # Large caps
    AssetSpec("AAPL", AssetClass.LARGE_CAP),
    AssetSpec("MSFT", AssetClass.LARGE_CAP),
    AssetSpec("GOOGL", AssetClass.LARGE_CAP),
    AssetSpec("AMZN", AssetClass.LARGE_CAP),
    AssetSpec("NVDA", AssetClass.LARGE_CAP),
    AssetSpec("JPM", AssetClass.LARGE_CAP),
    AssetSpec("XOM", AssetClass.LARGE_CAP),
    AssetSpec("JNJ", AssetClass.LARGE_CAP),
)

DEFAULT_START = date(2010, 1, 1)
DEFAULT_END = date(2025, 12, 31)

MIN_HISTORY_DAYS = 1000


def filter_universe(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Drop assets with fewer than ``MIN_HISTORY_DAYS`` rows, logging drops."""
    kept: dict[str, pd.DataFrame] = {}
    for symbol, df in data.items():
        if len(df) < MIN_HISTORY_DAYS:
            logger.info(
                "dropping %s: %d rows < MIN_HISTORY_DAYS=%d", symbol, len(df), MIN_HISTORY_DAYS
            )
            continue
        kept[symbol] = df
    return kept


def load_universe(source: DataSource) -> dict[str, pd.DataFrame]:
    """Fetch every asset in ``ASSET_UNIVERSE`` and filter by min history."""
    data = {
        spec.symbol: source.fetch(spec.symbol, DEFAULT_START, DEFAULT_END)
        for spec in ASSET_UNIVERSE
    }
    return filter_universe(data)
