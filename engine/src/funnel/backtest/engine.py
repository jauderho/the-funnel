"""Core backtest engine: turn a position series into net daily returns.

Timing convention (no look-ahead, matches ``funnel.strategies.base``): the
strategy contract states "position_t applies to the t -> t+1 return", i.e.
the position decided using data up to bar ``t`` earns the close-to-close
return realized over ``[t, t+1]``. Relabeling the output row by the return's
*end* date ``u = t + 1`` gives, for every output row ``u``:

- gross return: ``position_{u-1} * asset_return_u`` where
  ``asset_return_u = close_u / close_{u-1} - 1`` — the position held into
  ``u`` (decided at ``u - 1``, using only data known by then) times the
  return realized by the close of ``u``.
- cost: a trade is placed at ``u`` to move from ``position_{u-1}`` to
  ``position_u`` (so that ``position_u`` is on by the close of ``u``, ready
  to earn the *next* row's return). Its cost, ``|position_u -
  position_{u-1}| * bps/1e4``, is charged against row ``u`` — the day the
  change takes effect. A flat-to-long (or short) flip is 1 side; a
  long-to-short (or vice versa) flip is 2 sides — ``|delta position|``
  captures both naturally since position is in ``{-1, 0, 1}``.
- Both terms of row ``u`` are therefore ``position_{u-1}`` (gross) and the
  diff ending at ``u`` (cost) — no additional shift needed between them.

The first row has no prior close (``pct_change`` yields NaN) and no prior
position (nothing to diff against for cost), so it is undefined and dropped
rather than zero-filled — zero-filling would silently misrepresent "no data
yet" as "no return", which the funnel's Sharpe/drawdown math must not see.
"""

import pandas as pd

from funnel.config import CostModel
from funnel.data.universe import AssetClass


def cost_bps_for(asset_class: AssetClass, costs: CostModel) -> float:
    """Map an asset class to its per-side transaction cost in basis points."""
    if asset_class is AssetClass.CRYPTO:
        return costs.crypto_bps_per_side
    return costs.default_bps_per_side


def strategy_returns(positions: pd.Series, close: pd.Series, cost_bps_per_side: float) -> pd.Series:
    """Net daily strategy returns from a position series and close prices.

    ``positions`` and ``close`` must share the same index. Returns a series
    aligned to that index with the first row dropped (undefined: no prior
    close for the asset return, no prior position for the cost diff).
    """
    asset_return = close.pct_change()
    gross = positions.shift(1) * asset_return
    cost = positions.diff().abs() * (cost_bps_per_side / 1e4)
    net = gross - cost
    return net.iloc[1:]
