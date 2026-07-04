"""The six-filter survival funnel.

Every filter reads its threshold from a ``FunnelThresholds`` instance —
never a hardcoded literal — so a rendered report can always state the exact
numbers that were applied (the honesty-by-design requirement in PLAN.md).
All six filters are always evaluated, even after an earlier one fails, so
the attrition report can show per-filter pass counts rather than just a
single pass/fail verdict.
"""

from dataclasses import dataclass

from funnel.backtest.walkforward import WalkForwardResult
from funnel.config import FunnelThresholds


@dataclass(slots=True, frozen=True)
class FilterOutcome:
    """Per-filter pass/fail booleans for one walk-forward result."""

    passes_max_dd_floor: bool
    """Filter 1 — ``oos_max_drawdown > thresholds.max_dd_floor``."""

    passes_min_oos_sharpe: bool
    """Filter 2 — ``oos_sharpe > thresholds.min_oos_sharpe``."""

    passes_max_oos_sharpe: bool
    """Filter 3 — ``oos_sharpe < thresholds.max_oos_sharpe``."""

    passes_overfit_gap: bool
    """Filter 4 — the OOS-vs-IS Sharpe gap is not implausibly large."""

    passes_min_trades: bool
    """Filter 5 — ``oos_trade_count >= thresholds.min_trades``."""

    passes_positive_is_sharpe: bool
    """Filter 6 — ``is_sharpe > 0`` (only checked if ``thresholds.require_positive_is_sharpe``)."""

    @property
    def survived(self) -> bool:
        """Whether all six filters passed."""
        return (
            self.passes_max_dd_floor
            and self.passes_min_oos_sharpe
            and self.passes_max_oos_sharpe
            and self.passes_overfit_gap
            and self.passes_min_trades
            and self.passes_positive_is_sharpe
        )


FunnelVerdict = FilterOutcome
"""Alias: a ``FilterOutcome`` *is* the funnel verdict — it carries both the
per-filter booleans and the overall ``survived`` property."""


def apply_funnel(result: WalkForwardResult, thresholds: FunnelThresholds) -> FunnelVerdict:
    """Evaluate all six funnel filters against one walk-forward result.

    Filter 4 (the overfit-signature gap check) rule, exactly as implemented:
    ``oos_sharpe <= is_sharpe * max_oos_is_ratio`` when ``is_sharpe > 0``.
    When ``is_sharpe <= 0`` the ratio is not meaningful (multiplying by a
    non-positive number would either invert the inequality or make an
    arbitrarily bad OOS look "fine" relative to an already-bad IS), so the
    filter fails outright in that case — which is also consistent with
    filter 6 (positive IS Sharpe required) already failing the config.
    """
    passes_max_dd_floor = result.oos_max_drawdown > thresholds.max_dd_floor
    passes_min_oos_sharpe = result.oos_sharpe > thresholds.min_oos_sharpe
    passes_max_oos_sharpe = result.oos_sharpe < thresholds.max_oos_sharpe

    if result.is_sharpe > 0:
        passes_overfit_gap = result.oos_sharpe <= result.is_sharpe * thresholds.max_oos_is_ratio
    else:
        passes_overfit_gap = False

    passes_min_trades = result.oos_trade_count >= thresholds.min_trades

    if thresholds.require_positive_is_sharpe:
        passes_positive_is_sharpe = result.is_sharpe > 0
    else:
        passes_positive_is_sharpe = True

    return FilterOutcome(
        passes_max_dd_floor=passes_max_dd_floor,
        passes_min_oos_sharpe=passes_min_oos_sharpe,
        passes_max_oos_sharpe=passes_max_oos_sharpe,
        passes_overfit_gap=passes_overfit_gap,
        passes_min_trades=passes_min_trades,
        passes_positive_is_sharpe=passes_positive_is_sharpe,
    )
