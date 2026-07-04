"""Layer 2 — position sizing: map a raw {-1,0,1} position to a fractional weight.

Every function here takes a raw position series (values in ``{-1.0, 0.0,
1.0}``, the strategy contract from ``funnel.strategies.base``) and returns a
**weighted** position series in ``[-1.0, 1.0]`` — the same sign, scaled by
some risk-based fraction of full exposure. ``funnel.backtest.engine.
strategy_returns`` needs no changes to consume a weighted series: it is pure
multiplication (``positions.shift(1) * asset_return``) and the cost term
already scales naturally with ``|Δweight|`` rather than a fixed per-flip
charge.

CAUSALITY: every sizing function computes its scale factor at row ``t`` using
only data up to and including ``t`` (rolling windows, no negative shift), so
sizing never leaks information the base signal itself does not have. Warmup
rows (insufficient history for the rolling stat) are sized to 0.0, matching
the base-signal warmup convention.
"""

import numpy as np
import pandas as pd

from funnel.strategies.indicators import atr

TRADING_DAYS_PER_YEAR = 252


def volatility_target(
    positions: pd.Series,
    close: pd.Series,
    target_annual_vol: float = 0.15,
    vol_window: int = 21,
    max_leverage: float = 1.0,
) -> pd.Series:
    """Scale ``positions`` so realized asset volatility tracks ``target_annual_vol``.

    ``weight_t = clip(target_annual_vol / realized_vol_t, 0, max_leverage) *
    position_t``, where ``realized_vol_t`` is the annualized rolling
    standard deviation of the asset's daily returns over the trailing
    ``vol_window`` days *up to and including* ``t`` (causal: no centered or
    forward-looking window). Scaling is on the asset's own volatility, not
    the strategy's realized returns, so it is defined even on rows where
    ``position_t`` is flat.

    ``max_leverage`` caps the scale factor at ``1.0`` by default (no
    leverage above full exposure) — a deliberately conservative default per
    the no-large-margin constraint. Rows without enough history for the
    rolling vol (warmup) are sized to 0.0, and rows with zero realized vol
    (e.g. a dead-flat stretch) are also sized to 0.0 rather than dividing by
    zero.
    """
    daily_returns = close.pct_change()
    realized_vol = daily_returns.rolling(vol_window).std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

    scale = target_annual_vol / realized_vol.replace(0.0, np.nan)
    scale = scale.clip(upper=max_leverage).fillna(0.0)

    weighted = (scale * positions).reindex(positions.index).fillna(0.0)
    return weighted.astype("float64")


def atr_size(
    positions: pd.Series,
    df: pd.DataFrame,
    atr_window: int = 14,
    risk_fraction: float = 0.01,
    max_weight: float = 1.0,
) -> pd.Series:
    """Classic ATR risk-based position sizing.

    ``weight_t = min(max_weight, risk_fraction / (atr_t / close_t)) *
    position_t``. ``atr_t / close_t`` is ATR expressed as a fraction of
    price — the expected daily "risk unit" per unit of exposure — so
    ``risk_fraction / (atr_t / close_t)`` is the exposure at which one ATR
    move costs about ``risk_fraction`` of the position: a wider ATR (more
    volatile) shrinks the weight, a narrower ATR grows it, capped at
    ``max_weight``. Both ``atr`` and ``close`` at row ``t`` use only data up
    to and including ``t`` (``funnel.strategies.indicators.atr`` is causal),
    so the sizing is causal. Warmup rows (ATR not yet defined) and rows
    where ``close`` is zero/ATR is undefined are sized to 0.0.
    """
    atr_line = atr(df, atr_window)
    risk_per_unit = (atr_line / df["close"]).replace(0.0, np.nan)

    scale = (risk_fraction / risk_per_unit).clip(upper=max_weight).fillna(0.0)

    weighted = (scale * positions).reindex(positions.index).fillna(0.0)
    return weighted.astype("float64")


def cap_weight(positions: pd.Series, max_weight: float) -> pd.Series:
    """Clip ``positions`` to ``[-max_weight, max_weight]``."""
    return positions.clip(lower=-max_weight, upper=max_weight).astype("float64")
