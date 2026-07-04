"""Layer 4 — route a strategy's signal on/off by market regime.

Zeros out a position series wherever the current regime does not match the
regime the strategy is expected to perform best in
(``funnel.regime.base.PREFERRED_REGIME``, overridable per call).
"""

import pandas as pd

from funnel.regime.base import Regime


def route_by_regime(
    positions: pd.Series,
    regimes: pd.Series,
    active_regime: Regime,
) -> pd.Series:
    """Zero out ``positions`` wherever the regime label is not ``active_regime``.

    ``regimes`` is reindexed onto ``positions.index`` and forward-filled
    (regime labels are typically produced on a market-proxy calendar that
    may not align exactly with the position series' index — same
    convention as ``funnel.regime.base.regime_conditioned_metrics``). Rows
    with no regime label available even after forward-filling (before the
    first known label) are treated as not matching ``active_regime`` and
    are zeroed.
    """
    aligned_regimes = regimes.reindex(positions.index).ffill()
    mask = aligned_regimes == active_regime
    return positions.where(mask, 0.0).astype("float64")
