"""The full (config x asset) sweep: the concrete "test everything" step.

Runs every ``StrategyConfig`` against every asset's OHLCV frame through
walk-forward validation and the six-filter funnel, and assembles one row
per pair into a single DataFrame — the raw material the attrition report
(M2 §6) and later robustness checks (M3) are computed from. Pairs whose
asset history is too short for a meaningful walk-forward split are recorded
as skipped rows (flagged, not silently dropped) so the total count stays
honest.
"""

import logging
from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd

from funnel.backtest.engine import cost_bps_for
from funnel.backtest.funnel import apply_funnel
from funnel.backtest.walkforward import InsufficientHistoryError, walk_forward_oos
from funnel.cancellation import RunCancelledError
from funnel.config import CostModel, FunnelThresholds, WalkForwardConfig
from funnel.data.universe import AssetClass
from funnel.strategies.grid import StrategyConfig, total_backtest_count

logger = logging.getLogger(__name__)

SWEEP_COLUMNS: tuple[str, ...] = (
    "config_name",
    "family",
    "category",
    "params",
    "symbol",
    "is_sharpe",
    "oos_sharpe",
    "oos_max_drawdown",
    "oos_trade_count",
    "passes_max_dd_floor",
    "passes_min_oos_sharpe",
    "passes_max_oos_sharpe",
    "passes_overfit_gap",
    "passes_min_trades",
    "passes_positive_is_sharpe",
    "survived",
    "skipped",
)


def _params_to_str(params: dict[str, object]) -> str:
    """Render a params dict as a compact, deterministic, human-readable string."""
    return ",".join(f"{k}={v}" for k, v in sorted(params.items()))


def _skipped_row(config: StrategyConfig, symbol: str) -> dict[str, object]:
    return {
        "config_name": config.name,
        "family": config.family,
        "category": config.category.value,
        "params": _params_to_str(config.params),
        "symbol": symbol,
        "is_sharpe": float("nan"),
        "oos_sharpe": float("nan"),
        "oos_max_drawdown": float("nan"),
        "oos_trade_count": 0,
        "passes_max_dd_floor": False,
        "passes_min_oos_sharpe": False,
        "passes_max_oos_sharpe": False,
        "passes_overfit_gap": False,
        "passes_min_trades": False,
        "passes_positive_is_sharpe": False,
        "survived": False,
        "skipped": True,
    }


def run_sweep(
    data: Mapping[str, pd.DataFrame],
    configs: list[StrategyConfig],
    asset_classes: Mapping[str, AssetClass],
    wf: WalkForwardConfig,
    thresholds: FunnelThresholds,
    costs: CostModel,
    should_stop: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Run every (config, asset) pair through walk-forward + the six-filter funnel.

    ``data`` maps symbol -> OHLCV frame; ``asset_classes`` maps symbol ->
    ``AssetClass`` (used only to look up the per-asset-class cost rate, so
    it is hoisted out of the config loop below rather than recomputed per
    config).

    ``should_stop``, if given, is checked once per (config, asset) iteration
    — a stage-boundary check alone would leave a multi-minute sweep
    unstoppable, since this function runs to completion before its caller's
    next ``progress`` call. When it returns ``True``, ``RunCancelledError``
    is raised immediately and no rows (nor ``sweep_results.csv``) are
    written for this run; whatever a caller already wrote before invoking
    this function is unaffected.
    """
    symbols = list(data.keys())
    total = total_backtest_count(len(configs), len(symbols))
    logger.info("%d configs x %d assets = %d backtests", len(configs), len(symbols), total)
    print(f"{len(configs)} configs x {len(symbols)} assets = {total} backtests")

    # Hoisted out of the config loop: per-asset cost rate does not depend
    # on the strategy config, only on the asset's class.
    cost_by_symbol = {symbol: cost_bps_for(asset_classes[symbol], costs) for symbol in symbols}

    rows: list[dict[str, object]] = []
    for symbol in symbols:
        df = data[symbol]
        cost_bps = cost_by_symbol[symbol]
        for config in configs:
            if should_stop is not None and should_stop():
                raise RunCancelledError("run_sweep cancelled")
            try:
                result = walk_forward_oos(df, config, wf, cost_bps)
            except InsufficientHistoryError:
                rows.append(_skipped_row(config, symbol))
                continue

            verdict = apply_funnel(result, thresholds)
            rows.append(
                {
                    "config_name": config.name,
                    "family": config.family,
                    "category": config.category.value,
                    "params": _params_to_str(config.params),
                    "symbol": symbol,
                    "is_sharpe": result.is_sharpe,
                    "oos_sharpe": result.oos_sharpe,
                    "oos_max_drawdown": result.oos_max_drawdown,
                    "oos_trade_count": result.oos_trade_count,
                    "passes_max_dd_floor": verdict.passes_max_dd_floor,
                    "passes_min_oos_sharpe": verdict.passes_min_oos_sharpe,
                    "passes_max_oos_sharpe": verdict.passes_max_oos_sharpe,
                    "passes_overfit_gap": verdict.passes_overfit_gap,
                    "passes_min_trades": verdict.passes_min_trades,
                    "passes_positive_is_sharpe": verdict.passes_positive_is_sharpe,
                    "survived": verdict.survived,
                    "skipped": False,
                }
            )

    return pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))


def write_sweep_results(df: pd.DataFrame, path: Path) -> None:
    """Write the sweep DataFrame to ``path`` as CSV (e.g. ``sweep_results.csv``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
