"""Walk-forward out-of-sample validation.

Splits an asset's full history into ``n_windows`` sequential, contiguous,
non-overlapping windows of (near-)equal length. Within each window, the
first ``is_fraction`` of rows are in-sample (IS) and the remainder is
out-of-sample (OOS). The strategy is run *independently* on each window
(see ``_window_positions`` for why), and the OOS tails (and, separately,
the IS heads) are stitched into single combined series across all windows.

This is the mechanic the rest of the funnel depends on: every reported OOS
Sharpe / drawdown / trade count comes from data the strategy's indicators
never warmed up on, computed once per window rather than once on the full
series and then sliced.
"""

from dataclasses import dataclass

import pandas as pd

from funnel.backtest.engine import strategy_returns
from funnel.backtest.metrics import max_drawdown, sharpe, trade_count
from funnel.config import WalkForwardConfig
from funnel.strategies.grid import StrategyConfig

MIN_OOS_ROWS = 30
"""Minimum rows an individual window's OOS segment must have to be counted
as a meaningful independent trial. Below this, per-window Sharpe/drawdown
are too noisy to be worth computing at all."""


class InsufficientHistoryError(ValueError):
    """Raised when an asset's history is too short for a meaningful walk-forward split.

    The sweep runner (M2 §5) catches this and records the pair as skipped
    rather than letting it propagate — a too-short asset is not a failure,
    it is out of scope for validation.
    """


@dataclass(slots=True, frozen=True)
class WalkForwardResult:
    """Stitched in-sample / out-of-sample walk-forward outcome for one (config, asset) pair."""

    is_sharpe: float
    """Sharpe ratio of the stitched in-sample returns (all windows' IS legs concatenated)."""

    oos_sharpe: float
    """Sharpe ratio of the stitched out-of-sample returns (all windows' OOS tails concatenated)."""

    oos_max_drawdown: float
    """Max drawdown (<=0.0) of the stitched OOS returns."""

    oos_trade_count: int
    """Total position changes across all windows' OOS position segments."""

    oos_returns: pd.Series
    """The stitched OOS return series, in time order (needed by M3 robustness/bootstrap)."""

    is_returns: pd.Series
    """The stitched IS return series, in time order."""


def _window_bounds(n_rows: int, n_windows: int) -> list[tuple[int, int]]:
    """Split ``[0, n_rows)`` into ``n_windows`` contiguous, non-overlapping, near-equal ranges.

    Any remainder (when ``n_rows`` doesn't divide evenly) is distributed one
    row at a time to the earliest windows, so window sizes differ by at
    most one row. The ranges are half-open ``[start, end)`` and their union
    is exactly ``[0, n_rows)`` with no gaps or overlaps.
    """
    base_size, remainder = divmod(n_rows, n_windows)
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(n_windows):
        size = base_size + (1 if i < remainder else 0)
        end = start + size
        bounds.append((start, end))
        start = end
    return bounds


def _is_oos_split(start: int, end: int, is_fraction: float) -> tuple[int, int, int]:
    """Split a window's ``[start, end)`` into IS ``[start, split)`` and OOS ``[split, end)``."""
    size = end - start
    is_size = round(size * is_fraction)
    split = start + is_size
    return start, split, end


def _window_positions(df: pd.DataFrame, config: StrategyConfig) -> pd.Series:
    """Compute positions using only the given window's own data.

    Deliberately recomputes the strategy from scratch on ``df`` (the
    window's slice) rather than computing positions once on the full
    series and slicing the window out afterward. The two are not
    equivalent: an indicator computed on the full series has already
    warmed up on data from *before* the window when the window slice is
    taken, which would let pre-window history leak into what the funnel is
    supposed to treat as an independent out-of-sample trial. Recomputing
    per-window forces every window's indicators to warm up strictly within
    that window, exactly as they would if the window were the only history
    the strategy had ever seen — the honest measure of walk-forward
    robustness.
    """
    return config.fn(df, **config.params)


def walk_forward_oos(
    df: pd.DataFrame,
    config: StrategyConfig,
    wf: WalkForwardConfig,
    cost_bps: float,
) -> WalkForwardResult:
    """Run a strategy config through ``wf.n_windows`` walk-forward windows on ``df``.

    Raises ``InsufficientHistoryError`` if any window's OOS segment would
    have fewer than ``MIN_OOS_ROWS`` rows of returns — the sweep runner
    catches this and records the pair as skipped.
    """
    n_rows = len(df)
    bounds = _window_bounds(n_rows, wf.n_windows)

    oos_return_chunks: list[pd.Series] = []
    is_return_chunks: list[pd.Series] = []
    oos_position_chunks: list[pd.Series] = []

    for start, end in bounds:
        _, split, _ = _is_oos_split(start, end, wf.is_fraction)
        # strategy_returns drops only the window's very first row (index
        # `start`), which always falls inside the IS segment since
        # `split > start` whenever `is_fraction > 0`. So the OOS segment's
        # raw row count, `end - split`, equals the OOS *returns* count
        # exactly — no adjustment needed here (unlike the IS segment).
        oos_rows_in_window = end - split
        if oos_rows_in_window < MIN_OOS_ROWS:
            raise InsufficientHistoryError(
                f"window [{start}, {end}) has only {oos_rows_in_window} OOS rows "
                f"(< {MIN_OOS_ROWS} needed)"
            )

        window_df = df.iloc[start:end]
        positions = _window_positions(window_df, config)
        returns = strategy_returns(positions, window_df["close"], cost_bps)

        # `returns` is indexed like `window_df` minus its first row; split
        # it (and the matching position slice) at `split` in that same
        # local index space.
        is_returns = returns.loc[returns.index < window_df.index[split - start]]
        oos_returns = returns.loc[returns.index >= window_df.index[split - start]]
        oos_positions = positions.loc[positions.index >= window_df.index[split - start]]

        is_return_chunks.append(is_returns)
        oos_return_chunks.append(oos_returns)
        oos_position_chunks.append(oos_positions)

    stitched_is_returns = pd.concat(is_return_chunks)
    stitched_oos_returns = pd.concat(oos_return_chunks)

    # Trade count is summed per-window rather than computed on a single
    # concatenated position series: consecutive windows are independent
    # trials, so a position carried at the end of one window's OOS segment
    # must not be diffed against the next window's OOS segment start (that
    # boundary is not a real trade — it is two unrelated windows abutting).
    oos_trades = sum(trade_count(chunk) for chunk in oos_position_chunks)

    return WalkForwardResult(
        is_sharpe=sharpe(stitched_is_returns),
        oos_sharpe=sharpe(stitched_oos_returns),
        oos_max_drawdown=max_drawdown(stitched_oos_returns),
        oos_trade_count=oos_trades,
        oos_returns=stitched_oos_returns,
        is_returns=stitched_is_returns,
    )
