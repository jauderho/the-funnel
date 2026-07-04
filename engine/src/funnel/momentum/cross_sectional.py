"""Cross-sectional momentum: a standalone research/diagnostic check (PRD Layer 4).

**This module is research-only, not a tradeable strategy.** It ranks assets
against each other and holds both long and short positions (top third long,
bottom third short). The rest of this project enforces a long-only tradeable
track (signals clipped to ``{0, 1}``; see ``funnel.strategies.base``), so a
short-holding portfolio can never be surfaced as a tradeable recommendation.
Every result produced here carries ``research_only=True`` so downstream
reports and the UI cannot present it as anything else.

Mechanics
---------
Every 21 trading days (one rebalance cycle) on the panel's outer-joined
close-price calendar, every asset with a full lookback (+ skip, for 12-1) of
trailing history is ranked by trailing return. The top third is bought
equal-weight (long leg sums to +1.0), the bottom third is sold equal-weight
(short leg sums to -1.0); the middle third is unranked (0.0 weight). Weights
decided at a rebalance's close apply starting the *next* trading day — the
same no-look-ahead convention as ``funnel.backtest.engine`` — and are **held
constant until the next rebalance** (no intra-period drift as returns
accrue). This is a deliberate simplification: real turnover would drift
weights day-to-day as the winners/losers' relative sizes change, but constant
weights keep this diagnostic simple and its costs easy to reason about; it is
still accounted for in the docstring so nobody mistakes it for a drift model.

Transaction costs are charged only on rebalance dates, proportional to
per-asset turnover: ``cost_i = |w_new_i - w_old_i| * bps_i / 1e4``, where
``bps_i`` is the asset's own per-asset-class rate (crypto costs more). A
rebalance that reproduces the same ranking (no turnover) costs nothing.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from funnel.backtest.engine import cost_bps_for
from funnel.backtest.metrics import max_drawdown, sharpe
from funnel.backtest.walkforward import _is_oos_split, _window_bounds
from funnel.config import CostModel, WalkForwardConfig
from funnel.data.universe import AssetClass, filter_universe

REBALANCE_EVERY_DAYS = 21
"""Rebalance cadence, in trading days — one rebalance per ~calendar month."""


@dataclass(slots=True, frozen=True)
class Lookback:
    """One momentum-ranking lookback definition.

    ``trailing_days`` is the length of the return window; ``skip_days`` is
    how many of the most recent trading days are excluded from that window
    (used by "12-1" to avoid short-term reversal contamination: the ranking
    return is measured from ``t - trailing_days - skip_days`` to
    ``t - skip_days``, i.e. it ends ``skip_days`` before the rebalance date).
    """

    label: str
    trailing_days: int
    skip_days: int = 0

    @property
    def history_needed(self) -> int:
        """Total trailing rows of history required to compute this lookback."""
        return self.trailing_days + self.skip_days


LOOKBACKS: tuple[Lookback, ...] = (
    Lookback(label="3m", trailing_days=63, skip_days=0),
    Lookback(label="6m", trailing_days=126, skip_days=0),
    Lookback(label="12-1", trailing_days=252, skip_days=21),
)


@dataclass(slots=True, frozen=True)
class CrossSectionalScore:
    """Walk-forward score for one lookback's cross-sectional portfolio.

    Carries only what ``walk_forward_score`` can compute from a bare return
    series; the lookback label and rebalance count are attached by the
    caller (``run_cross_sectional_check``), which has that context.
    """

    is_sharpe: float
    oos_sharpe: float
    oos_max_drawdown: float


def _build_close_panel(data: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Outer-join every asset's close series onto one wide panel, forward-fill gaps.

    Forward-filling means an asset missing a bar on a given calendar date
    (e.g. a holiday observed on one exchange but not another) carries its
    last known price rather than injecting a spurious return; a leading gap
    before an asset's own history starts is left as NaN so it is correctly
    treated as "no data yet" by the lookback/history checks below.
    """
    closes = {symbol: df["close"] for symbol, df in data.items()}
    panel = pd.DataFrame(closes).sort_index()
    return panel.ffill()


def _rebalance_dates(index: pd.Index, first_valid_i: int) -> list[int]:
    """Positional indices of rebalance dates: every ``REBALANCE_EVERY_DAYS`` rows, starting
    at the first position where at least one asset could plausibly be ranked."""
    return list(range(first_valid_i, len(index), REBALANCE_EVERY_DAYS))


def _trailing_return(panel: pd.DataFrame, i: int, lookback: Lookback) -> pd.Series:
    """Per-asset trailing return ending ``lookback.skip_days`` before position ``i``.

    ``i`` is the rebalance date's positional index. Return is
    ``close[i - skip_days] / close[i - skip_days - trailing_days] - 1``,
    using only rows strictly before/at ``i - skip_days`` — never row ``i``
    itself when ``skip_days > 0`` avoids leaking the most recent month into
    the ranking (the "12-1" construction). An asset without a real (non-NaN,
    non-forward-filled-from-before-its-start) price at both endpoints yields
    NaN and is excluded from ranking that date.
    """
    end_i = i - lookback.skip_days
    start_i = end_i - lookback.trailing_days
    if start_i < 0:
        return pd.Series(np.nan, index=panel.columns)
    end_prices = panel.iloc[end_i]
    start_prices = panel.iloc[start_i]
    return end_prices / start_prices - 1.0


def _target_weights(trailing_return: pd.Series, columns: pd.Index) -> pd.Series:
    """Rank assets with a valid trailing return into top/bottom-third long/short legs.

    Assets with fewer than 3 eligible (non-NaN) candidates on a given date
    produce an all-zero (flat) weight vector — there is no meaningful
    top/bottom third to form with fewer than 3 names.
    """
    weights = pd.Series(0.0, index=columns)
    eligible = trailing_return.dropna()
    n = len(eligible)
    if n < 3:
        return weights

    third = n // 3
    ranked = eligible.sort_values(ascending=False)
    longs = ranked.index[:third]
    shorts = ranked.index[-third:]

    weights.loc[longs] = 1.0 / third
    weights.loc[shorts] = -1.0 / third
    return weights


def cross_sectional_returns(
    data: dict[str, pd.DataFrame],
    lookback: Lookback,
    cost_model: CostModel,
    asset_classes: Mapping[str, AssetClass],
) -> pd.Series:
    """Daily net return series of the top/bottom-third long/short momentum portfolio.

    Builds a close-price panel from ``data`` (already filtered to the
    project's standard universe + ``MIN_HISTORY_DAYS``, per PRD), ranks
    assets every ``REBALANCE_EVERY_DAYS`` trading days by trailing return
    (``lookback``), and holds equal-weight long/short legs (top/bottom
    third) constant until the next rebalance. Weights decided at a
    rebalance's close apply starting the next day (no look-ahead, matching
    ``funnel.backtest.engine``'s ``positions.shift(1) * asset_return``
    convention). Turnover cost is charged on the rebalance date's own row —
    the day the new weights are "achieved" (mirrors engine.py's
    ``positions.diff().abs()`` being charged same-row as the new position,
    not the day after); see the module docstring for the "no intra-period
    drift" simplification.
    """
    filtered = filter_universe(data)
    panel = _build_close_panel(filtered)
    asset_returns = panel.pct_change()

    n_rows = len(panel)
    first_valid_i = lookback.history_needed
    rebalance_positions = [i for i in _rebalance_dates(panel.index, first_valid_i) if i < n_rows]

    # Pass 1: compute the weight vector set at each rebalance date, and the
    # turnover cost incurred getting there from the prior weights.
    weights_by_date: dict[pd.Timestamp, pd.Series] = {}
    cost_by_date: dict[pd.Timestamp, float] = {}
    prev_weights = pd.Series(0.0, index=panel.columns)
    for i in rebalance_positions:
        date = panel.index[i]
        trailing_return = _trailing_return(panel, i, lookback)
        new_weights = _target_weights(trailing_return, panel.columns)
        turnover = (new_weights - prev_weights).abs()
        cost = float(
            sum(
                turnover[symbol] * cost_bps_for(asset_classes[symbol], cost_model) / 1e4
                for symbol in panel.columns
                if symbol in asset_classes
            )
        )
        weights_by_date[date] = new_weights
        cost_by_date[date] = cost
        prev_weights = new_weights

    # Pass 2: forward-fill the weight vector across all days (constant
    # between rebalances — the documented simplification), then apply it to
    # the *next* day's return (no look-ahead).
    weights_df = pd.DataFrame(weights_by_date).T.reindex(panel.index).ffill().fillna(0.0)
    applied_weights = weights_df.shift(1).fillna(0.0)

    gross = (applied_weights * asset_returns.fillna(0.0)).sum(axis=1)
    cost_series = pd.Series(cost_by_date, dtype="float64").reindex(panel.index).fillna(0.0)
    net = gross - cost_series

    # Rows before the first rebalance are always exactly zero (no weights
    # have ever been set, so there is nothing to apply and no cost to
    # charge) and carry no information; the first rebalance's own row
    # (index `first_valid_i`) is the first meaningful row — it already
    # reflects that rebalance's turnover cost — so the series starts there,
    # not one row later.
    return net.iloc[first_valid_i:]


def walk_forward_score(portfolio_returns: pd.Series, wf: WalkForwardConfig) -> CrossSectionalScore:
    """Apply the same 5-window/70-30 walk-forward discipline to a return series.

    Unlike ``funnel.backtest.walkforward.walk_forward_oos`` (which recomputes
    indicators per-window to avoid leaking pre-window history into a
    strategy's warmup), there is no indicator recompute step here: the
    cross-sectional portfolio's return series is already fully formed (it
    was built with no look-ahead in ``cross_sectional_returns``). Slicing
    that already-realized return series into 5 sequential windows and
    taking each window's last 30% as OOS is therefore the directly
    comparable treatment — the IS/OOS split boundaries are identical to the
    single-asset sweep, just applied to a return series instead of an
    indicator-driven position series.
    """
    n_rows = len(portfolio_returns)
    bounds = _window_bounds(n_rows, wf.n_windows)

    is_chunks: list[pd.Series] = []
    oos_chunks: list[pd.Series] = []
    for start, end in bounds:
        _, split, _ = _is_oos_split(start, end, wf.is_fraction)
        is_chunks.append(portfolio_returns.iloc[start:split])
        oos_chunks.append(portfolio_returns.iloc[split:end])

    stitched_is = pd.concat(is_chunks)
    stitched_oos = pd.concat(oos_chunks)

    return CrossSectionalScore(
        is_sharpe=sharpe(stitched_is),
        oos_sharpe=sharpe(stitched_oos),
        oos_max_drawdown=max_drawdown(stitched_oos),
    )


CROSS_SECTIONAL_COLUMNS: tuple[str, ...] = (
    "lookback",
    "is_sharpe",
    "oos_sharpe",
    "oos_max_drawdown",
    "n_rebalances",
    "research_only",
)

_COMPARISON_FAMILIES = ("time_series_momentum", "roc_momentum")


def _single_asset_comparison(single_asset_momentum: pd.DataFrame) -> float:
    """Mean OOS Sharpe of the single-asset momentum families used for comparison.

    ``single_asset_momentum`` is the sweep results DataFrame (columns
    include ``family`` and ``oos_sharpe``); only the rows belonging to
    ``_COMPARISON_FAMILIES`` (the single-asset trend-following momentum
    families) are averaged, so the comparison is apples-to-apples: momentum
    signal vs. momentum signal, single-asset vs. cross-sectional.
    """
    subset = single_asset_momentum.loc[single_asset_momentum["family"].isin(_COMPARISON_FAMILIES)]
    if subset.empty:
        return float("nan")
    return float(subset["oos_sharpe"].mean())


def run_cross_sectional_check(
    data: dict[str, pd.DataFrame],
    wf: WalkForwardConfig,
    cost_model: CostModel,
    asset_classes: Mapping[str, AssetClass],
    single_asset_momentum: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Run every lookback's cross-sectional portfolio through walk-forward scoring.

    One row per lookback (``LOOKBACKS``): lookback label, IS/OOS Sharpe, OOS
    max drawdown, rebalance count, and ``research_only=True`` on every row.
    If ``single_asset_momentum`` (the sweep results DataFrame) is given, an
    extra ``single_asset_mean_oos_sharpe`` column is added — the mean OOS
    Sharpe of the ``time_series_momentum`` and ``roc_momentum`` families —
    repeated on every row so the report can show cross-sectional vs.
    single-asset side by side.
    """
    # Hoisted out of the per-lookback loop: the panel and calendar do not
    # depend on the lookback, only the rebalance-eligibility cutoff does.
    panel_index = _build_close_panel(filter_universe(data)).index

    n_panel_rows = len(panel_index)
    rows: list[dict[str, object]] = []
    for lookback in LOOKBACKS:
        returns = cross_sectional_returns(data, lookback, cost_model, asset_classes)
        rebalance_positions = _rebalance_dates(panel_index, lookback.history_needed)
        n_rebalances = len([i for i in rebalance_positions if i < n_panel_rows])
        score = walk_forward_score(returns, wf)
        rows.append(
            {
                "lookback": lookback.label,
                "is_sharpe": score.is_sharpe,
                "oos_sharpe": score.oos_sharpe,
                "oos_max_drawdown": score.oos_max_drawdown,
                "n_rebalances": n_rebalances,
                "research_only": True,
            }
        )

    df = pd.DataFrame(rows, columns=list(CROSS_SECTIONAL_COLUMNS))

    if single_asset_momentum is not None:
        df["single_asset_mean_oos_sharpe"] = _single_asset_comparison(single_asset_momentum)

    return df


def plain_language_verdict(df: pd.DataFrame) -> str:
    """An honest, non-tuned-to-look-good verdict sentence for the cross-sectional check.

    States whether ranking assets against each other beat single-asset
    momentum (when the comparison column is present), and always names the
    deepest OOS drawdown observed across lookbacks — a strategy with a
    positive Sharpe but a catastrophic drawdown is not quietly omitted.
    """
    if df.empty:
        return "Cross-sectional momentum check produced no results."

    best = df.loc[df["oos_sharpe"].idxmax()]
    worst_dd = float(df["oos_max_drawdown"].min())

    sentences = [
        f"Cross-sectional momentum (research-only, long/short): best lookback "
        f"'{best['lookback']}' scored OOS Sharpe {best['oos_sharpe']:.2f}, "
        f"deepest OOS drawdown across lookbacks was {worst_dd:.1%}."
    ]

    if "single_asset_mean_oos_sharpe" in df.columns:
        single_sharpe = float(df["single_asset_mean_oos_sharpe"].iloc[0])
        if pd.isna(single_sharpe):
            sentences.append(
                "No comparable single-asset momentum survivors were available for comparison."
            )
        elif best["oos_sharpe"] > single_sharpe:
            sentences.append(
                f"This beat the single-asset momentum families' mean OOS Sharpe "
                f"({single_sharpe:.2f})."
            )
        else:
            sentences.append(
                f"This did NOT beat the single-asset momentum families' mean OOS Sharpe "
                f"({single_sharpe:.2f})."
            )

    sentences.append(
        "Reminder: this portfolio holds short positions and is not part of the "
        "tradeable (long-only) track."
    )
    return " ".join(sentences)


def write_cross_sectional(df: pd.DataFrame, path: Path) -> None:
    """Write the cross-sectional check DataFrame to ``path`` as CSV (``cross_sectional.csv``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
