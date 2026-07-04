"""The regime classification contract and regime-conditioned metrics.

A ``RegimeDetector`` is any object with a ``classify(df) -> pd.Series``
method. ``df`` is an OHLCV frame for the market proxy (see
``funnel.data.sources.DataSource``) and the return value is a **regime
label series** aligned to ``df.index``, with values drawn from ``Regime``.

CRITICAL NO-LOOK-AHEAD RULE (same discipline as ``funnel.strategies.base``)
----------------------------------------------------------------------------
The regime label at index ``t`` may use data up to and including bar ``t``
only. Detectors must not fit once on the full series and label the past
with that fit — see ``regime.hmm`` for the one nuanced case (periodic
expanding-window refits), which documents exactly how it stays causal.

Warmup convention
------------------
Rows before a detector has enough history to produce a real signal are
labeled ``Regime.CHOPPY``, not a third "unknown" value. This is a
deliberate choice: CHOPPY is the conservative, mean-reversion-friendly
default (the lower-risk assumption when the detector has no opinion yet),
and keeping the output strictly binary means every downstream consumer
(regime-conditioned metrics, M6 routing) can treat the label series
uniformly without a third branch to handle.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import pandas as pd

from funnel.backtest.metrics import max_drawdown, sharpe
from funnel.strategies.base import Category


class Regime(StrEnum):
    """Market regime label."""

    TRENDING = "trending"
    CHOPPY = "choppy"


@runtime_checkable
class RegimeDetector(Protocol):
    """Callable signature every regime detector implements."""

    def classify(self, df: pd.DataFrame) -> pd.Series:
        """Return a regime label series aligned to ``df.index``.

        Values are ``Regime`` members only (object dtype — see each
        detector's docstring; categorical is not used so that comparison
        helpers can freely concatenate/compare label series from different
        detectors without dtype reconciliation). No NaNs: every row,
        including warmup, gets a label.
        """
        ...


PREFERRED_REGIME: Mapping[Category, Regime] = {
    Category.TREND: Regime.TRENDING,
    Category.VOLATILITY: Regime.TRENDING,
    Category.COMPOSITE: Regime.TRENDING,
    Category.MEAN_REVERSION: Regime.CHOPPY,
    Category.PATTERN: Regime.CHOPPY,
    Category.VOLUME: Regime.TRENDING,
}
"""Default regime each strategy category is expected to perform best in.

These are starting-point defaults for M6 routing, not hard rules: trend,
volatility-breakout, and composite families are expected to do best when
the market is trending; mean-reversion and pattern families are expected to
do best in choppy/range-bound conditions; volume-based families are grouped
with TRENDING as a default because volume-confirmation techniques (e.g.
OBV/breakout-with-volume) are typically trend-following in construction.
All of these are overridable — they encode a hypothesis to test via
regime-conditioned metrics, not a presumed truth.
"""


@dataclass(slots=True, frozen=True)
class RegimeMetrics:
    """Performance metrics computed on the subset of days in one regime."""

    sharpe: float
    max_drawdown: float
    n_days: int
    total_return: float


def regime_conditioned_metrics(
    returns: pd.Series, regimes: pd.Series
) -> dict[Regime, RegimeMetrics]:
    """Split ``returns`` by regime label and compute per-regime metrics.

    ``regimes`` is reindexed onto ``returns.index`` and forward-filled
    (regime labels are typically produced on a market-proxy calendar that
    may not align exactly with a given strategy's return index) before
    splitting. Days with no regime label available even after
    forward-filling (i.e. before the first known label) are excluded from
    both subsets.

    Returns one entry per ``Regime`` member that has at least one day in
    ``returns`` (a regime with zero days present is omitted rather than
    reported with misleading zeroed-out metrics).
    """
    aligned_regimes = regimes.reindex(returns.index).ffill()

    result: dict[Regime, RegimeMetrics] = {}
    for regime in Regime:
        mask = aligned_regimes == regime
        subset = returns.loc[mask]
        if subset.empty:
            continue
        equity = (1.0 + subset.dropna()).cumprod()
        total_return = float(equity.iloc[-1] - 1.0) if not equity.empty else 0.0
        result[regime] = RegimeMetrics(
            sharpe=sharpe(subset),
            max_drawdown=max_drawdown(subset),
            n_days=int(len(subset)),
            total_return=total_return,
        )
    return result
