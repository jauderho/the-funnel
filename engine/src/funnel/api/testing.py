"""Test/dev-only helpers for the API layer: a deterministic synthetic data source.

Kept out of ``funnel.data.sources`` deliberately (that module's contract is
production data sourcing only). ``SyntheticSource`` generates plausible,
network-free OHLCV data for every symbol requested, so local UI development
(``FUNNEL_FAKE_DATA=1``) and tests never depend on network access or
yfinance.
"""

from datetime import date

import numpy as np
import pandas as pd


class SyntheticSource:
    """Deterministic, network-free OHLCV generator keyed by symbol name.

    Each symbol gets its own seeded RNG (derived from a hash of the symbol),
    so repeated calls for the same symbol/date range are reproducible and
    different symbols get visibly different price paths.
    """

    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        index = pd.bdate_range(start, end)
        n = len(index)
        seed = abs(hash(symbol)) % (2**32)
        rng = np.random.default_rng(seed)

        drift = rng.normal(loc=0.0003, scale=0.0002)
        daily_returns = rng.normal(loc=drift, scale=0.015, size=n)
        close = 100.0 * np.cumprod(1.0 + daily_returns)

        daily_range = np.abs(rng.normal(loc=0.5, scale=0.2, size=n)) + 0.05
        open_ = close + rng.normal(loc=0.0, scale=0.1, size=n)
        high = np.maximum(open_, close) + daily_range
        low = np.minimum(open_, close) - daily_range
        volume = np.abs(rng.normal(loc=1_000_000.0, scale=200_000.0, size=n))

        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        ).astype("float64")
