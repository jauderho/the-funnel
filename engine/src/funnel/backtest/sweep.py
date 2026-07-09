"""The full (config x asset) sweep: the concrete "test everything" step.

Runs every ``StrategyConfig`` against every asset's OHLCV frame through
walk-forward validation and the six-filter funnel, and assembles one row
per pair into a single DataFrame — the raw material the attrition report
(M2 §6) and later robustness checks (M3) are computed from. Pairs whose
asset history is too short for a meaningful walk-forward split are recorded
as skipped rows (flagged, not silently dropped) so the total count stays
honest.

PERF-1 parallelism (n_workers)
-------------------------------------------------------------------------
The (config x asset) sweep is embarrassingly parallel: each asset's
backtests are independent of every other asset's. ``run_sweep`` chunks work
**by asset** — one task per symbol, covering every config for that symbol —
so each asset's OHLCV frame crosses a process-pool worker boundary exactly
once rather than once per (config, asset) pair. ``n_workers in {0, 1}``
(and the resolved-to-1 edge case) always takes the original single-process
loop (``_run_sweep_serial``) completely unchanged, which both is the
default-equivalence baseline for tests and remains the only path usable
when ``should_stop`` needs (config, asset)-granularity cancellation
checks. The parallel path (``_run_sweep_parallel``) calls the exact same
``_score_pair`` scoring function per pair as the serial path — the two
paths differ only in *how* work is scheduled, never in the arithmetic
performed, so their outputs are byte-identical for the same inputs.
"""

import logging
import os
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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


def _score_pair(
    config: StrategyConfig,
    symbol: str,
    df: pd.DataFrame,
    wf: WalkForwardConfig,
    thresholds: FunnelThresholds,
    cost_bps: float,
) -> dict[str, object]:
    """Score one (config, symbol) pair: walk-forward + funnel -> one sweep row.

    A pure function of its arguments (no shared/mutable state), called
    identically by the serial in-process loop and by a parallel worker
    process — the single source of truth both paths call, so their numeric
    output is guaranteed identical regardless of which one executes it.
    """
    try:
        result = walk_forward_oos(df, config, wf, cost_bps)
    except InsufficientHistoryError:
        return _skipped_row(config, symbol)

    verdict = apply_funnel(result, thresholds)
    return {
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


def _score_symbol(
    symbol: str,
    df: pd.DataFrame,
    configs: list[StrategyConfig],
    wf: WalkForwardConfig,
    thresholds: FunnelThresholds,
    cost_bps: float,
) -> list[dict[str, object]]:
    """Score every config against one symbol's OHLCV frame.

    Module-level (not a closure or bound method) and built only from
    picklable arguments (a DataFrame, frozen dataclasses, and
    ``StrategyConfig``s whose ``fn`` fields are module-level callables), so
    it can be submitted as-is to a ``ProcessPoolExecutor`` under both macOS
    ``spawn`` and Linux ``fork``. This is the per-asset chunk
    ``_run_sweep_parallel`` dispatches to each worker: ``df`` crosses the
    process boundary exactly once per asset rather than once per
    (config, asset) pair.
    """
    return [_score_pair(config, symbol, df, wf, thresholds, cost_bps) for config in configs]


def _resolve_n_workers(n_workers: int | None, n_symbols: int) -> int:
    """Resolve the requested worker count to an actual process-pool size.

    ``None`` defaults to ``os.process_cpu_count()`` (falling back to 1 if
    that is unavailable). The result is capped at ``n_symbols`` since work
    is chunked per-asset — more workers than assets can never be used.
    """
    resolved = n_workers if n_workers is not None else (os.process_cpu_count() or 1)
    return max(1, min(resolved, max(n_symbols, 1)))


def _run_sweep_serial(
    symbols: list[str],
    data: Mapping[str, pd.DataFrame],
    configs: list[StrategyConfig],
    wf: WalkForwardConfig,
    thresholds: FunnelThresholds,
    cost_by_symbol: Mapping[str, float],
    should_stop: Callable[[], bool] | None,
) -> pd.DataFrame:
    """Original single-process sweep loop: the equivalence baseline.

    ``n_workers`` in ``{0, 1}`` (and the resolved-to-1 edge case) always
    takes this path verbatim. ``should_stop`` is checked once per
    (config, asset) iteration here — the finest-grained cancellation point
    available in this codebase.
    """
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        df = data[symbol]
        cost_bps = cost_by_symbol[symbol]
        for config in configs:
            if should_stop is not None and should_stop():
                raise RunCancelledError("run_sweep cancelled")
            rows.append(_score_pair(config, symbol, df, wf, thresholds, cost_bps))
    return pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))


def _run_sweep_parallel(
    symbols: list[str],
    data: Mapping[str, pd.DataFrame],
    configs: list[StrategyConfig],
    wf: WalkForwardConfig,
    thresholds: FunnelThresholds,
    cost_by_symbol: Mapping[str, float],
    n_workers: int,
    should_stop: Callable[[], bool] | None,
) -> pd.DataFrame:
    """Per-asset process-pool sweep: one task per symbol, scored via ``_score_symbol``.

    Results are assembled in ``symbols`` (task-submission) order regardless
    of completion order, so output is identical to ``_run_sweep_serial`` for
    the same inputs.

    ``should_stop`` is polled once between each batch of completed futures
    (finer-grained per-(config, asset) checking is not possible across a
    process boundary without an expensive shared-cancellation channel — a
    ``threading.Event``-backed closure, which is how ``should_stop`` is
    implemented by the job registry, cannot be pickled into a worker
    process). Each per-asset task takes a small fraction of the total sweep
    time, so this still bounds cancellation latency to a few seconds. On
    cancellation, not-yet-started futures are cancelled and the pool is torn
    down without waiting for already-running tasks to finish
    (``cancel_futures=True``, ``wait=False``), then ``RunCancelledError`` is
    raised. Any other exception (e.g. a worker task raising) is handled the
    same non-blocking way before re-raising.
    """
    executor = ProcessPoolExecutor(max_workers=n_workers)
    rows_by_symbol: dict[str, list[dict[str, object]]] = {}
    try:
        futures = {
            executor.submit(
                _score_symbol, symbol, data[symbol], configs, wf, thresholds, cost_by_symbol[symbol]
            ): symbol
            for symbol in symbols
        }
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                rows_by_symbol[futures[future]] = future.result()
            if should_stop is not None and should_stop():
                raise RunCancelledError("run_sweep cancelled")
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    rows: list[dict[str, object]] = []
    for symbol in symbols:
        rows.extend(rows_by_symbol[symbol])
    return pd.DataFrame(rows, columns=list(SWEEP_COLUMNS))


def run_sweep(
    data: Mapping[str, pd.DataFrame],
    configs: list[StrategyConfig],
    asset_classes: Mapping[str, AssetClass],
    wf: WalkForwardConfig,
    thresholds: FunnelThresholds,
    costs: CostModel,
    should_stop: Callable[[], bool] | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Run every (config, asset) pair through walk-forward + the six-filter funnel.

    ``data`` maps symbol -> OHLCV frame; ``asset_classes`` maps symbol ->
    ``AssetClass`` (used only to look up the per-asset-class cost rate, so
    it is hoisted out of the config loop below rather than recomputed per
    config).

    ``n_workers`` controls sweep parallelism (see the module docstring for
    the chunking/equivalence design): ``None`` (the default) resolves to
    ``os.process_cpu_count()`` (capped at the number of assets) and runs the
    per-asset ``ProcessPoolExecutor`` path; ``0`` or ``1`` always runs the
    original single-process loop unchanged.

    ``should_stop``, if given, is checked once per (config, asset) iteration
    in the serial path (``n_workers`` 0/1), or once per completed per-asset
    task in the parallel path — see ``_run_sweep_parallel`` for why a
    coarser check is unavoidable across a process boundary. When it returns
    ``True``, ``RunCancelledError`` is raised immediately and no rows (nor
    ``sweep_results.csv``) are written for this run; whatever a caller
    already wrote before invoking this function is unaffected.
    """
    symbols = list(data.keys())
    total = total_backtest_count(len(configs), len(symbols))
    logger.info("%d configs x %d assets = %d backtests", len(configs), len(symbols), total)
    print(f"{len(configs)} configs x {len(symbols)} assets = {total} backtests")

    # Hoisted out of the config loop: per-asset cost rate does not depend
    # on the strategy config, only on the asset's class.
    cost_by_symbol = {symbol: cost_bps_for(asset_classes[symbol], costs) for symbol in symbols}

    if n_workers == 0 or n_workers == 1:
        return _run_sweep_serial(
            symbols, data, configs, wf, thresholds, cost_by_symbol, should_stop
        )

    resolved_workers = _resolve_n_workers(n_workers, len(symbols))
    if resolved_workers <= 1:
        return _run_sweep_serial(
            symbols, data, configs, wf, thresholds, cost_by_symbol, should_stop
        )

    return _run_sweep_parallel(
        symbols, data, configs, wf, thresholds, cost_by_symbol, resolved_workers, should_stop
    )


def write_sweep_results(df: pd.DataFrame, path: Path) -> None:
    """Write the sweep DataFrame to ``path`` as CSV (e.g. ``sweep_results.csv``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
