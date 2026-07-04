"""The strategy signal contract.

A strategy is any callable with the signature::

    (df: pd.DataFrame, **params) -> pd.Series

``df`` is an OHLCV frame (see ``funnel.data.sources.DataSource``) and the
return value is a **position series** aligned to ``df.index``, with values
restricted to ``{-1.0, 0.0, 1.0}`` (float64): short, flat, long.

CRITICAL NO-LOOK-AHEAD RULE
---------------------------
The position at index ``t`` may use data up to and including bar ``t``
(e.g. ``close_t``). The backtest engine (M2) applies ``position_t`` to the
return from ``t`` to ``t + 1`` — computing ``position_t`` from anything at
``t + 1`` or later is a bug, not an optimization.

Concretely, a strategy must never:

- call ``.shift(-k)`` for any ``k > 0``,
- use a rolling window with ``center=True``,
- use whole-series statistics (a full-sample mean/std/min/max, or any other
  global/normalization computed over the entire frame rather than causally,
  up to the current row).

Only rolling/expanding constructs (computed strictly from data at or before
the current row) are permitted. Warmup rows — where an indicator isn't yet
defined because of insufficient history — must be assigned position 0.0,
not NaN.
"""

from enum import StrEnum
from typing import Protocol, runtime_checkable

import pandas as pd

VALID_POSITIONS = frozenset({-1.0, 0.0, 1.0})


class Category(StrEnum):
    """Strategy family categories."""

    TREND = "trend"
    MEAN_REVERSION = "mean_reversion"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    PATTERN = "pattern"
    COMPOSITE = "composite"


@runtime_checkable
class Strategy(Protocol):
    """Callable signature every strategy function implements."""

    def __call__(self, df: pd.DataFrame, **params: object) -> pd.Series: ...


def clean_positions(raw: pd.Series, index: pd.Index) -> pd.Series:
    """Finalize a raw position series: reindex, fill warmup NaNs with 0.0, cast.

    Shared tail step for strategy implementations: aligns the computed
    signal to ``index`` (in case an implementation's intermediate result
    dropped rows via ``dropna``), fills any remaining NaN (warmup) with
    0.0, and casts to float64.
    """
    return raw.reindex(index).fillna(0.0).astype("float64")
