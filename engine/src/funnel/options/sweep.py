"""The full (overlay config x symbol) sweep: overlay-vs-buy-and-hold validation.

Runs every ``OverlayConfig`` against every symbol's OHLCV frame through
``simulate_overlay`` (V2-M2), scores the resulting overlay return series
*and* the underlying's buy-and-hold return series through v1's exact
walk-forward window discipline (``score_overlay``, reusing
``funnel.backtest.walkforward``'s private window-splitting helpers exactly as
``funnel.momentum.cross_sectional.walk_forward_score`` already does for an
already-realized return series), then bootstrap-stresses the overlay's
stitched OOS returns (reusing ``funnel.robustness.bootstrap`` as-is). One row
per (config, symbol) pair is assembled into a single DataFrame — pairs whose
history is too short for a meaningful walk-forward split are recorded as
skipped rows (flagged, not silently dropped), mirroring
``funnel.backtest.sweep``.

HONESTY RULES (PLAN.md, "v2 — Options Overlay Module")
--------------------------------------------------------------------------
- No row is ever filtered out for looking bad: every (config, symbol) pair
  that runs produces a row, survived or not.
- The buy-and-hold comparison columns (``underlying_oos_sharpe``,
  ``underlying_oos_max_drawdown``, ``oos_sharpe_vs_hold``) are always present,
  scored on the identical windows as the overlay, so "vs. hold" is never
  hidden or opt-in.
- ``model_priced`` is a constant ``True`` column on every row (including
  skipped ones): every price in this module is a synthetic BSM model price,
  never a market quote, and every consumer of this CSV must be able to see
  that from the row itself.
- The reported assignment-probability column is named ``mean_model_prob_itm``
  — never bare "assignment probability" — per ``pricing.py``'s labeling rule
  (it is a risk-neutral model P(ITM at expiry), not a real-world forecast).
"""

import logging
import os
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from funnel.backtest.metrics import max_drawdown, sharpe
from funnel.backtest.walkforward import (
    MIN_OOS_ROWS,
    InsufficientHistoryError,
    _is_oos_split,
    _window_bounds,
)
from funnel.cancellation import RunCancelledError
from funnel.config import FunnelThresholds, WalkForwardConfig
from funnel.options.grid import OverlayConfig
from funnel.options.overlays import OverlayCosts, UndefinedRiskError, simulate_overlay
from funnel.options.pricing import VolProxyConfig
from funnel.robustness.bootstrap import bootstrap_stress

logger = logging.getLogger(__name__)

OVERLAY_SWEEP_COLUMNS: tuple[str, ...] = (
    "config_name",
    "structure",
    "symbol",
    "spec_params",
    "overlay_is_sharpe",
    "overlay_oos_sharpe",
    "overlay_oos_max_drawdown",
    "underlying_oos_sharpe",
    "underlying_oos_max_drawdown",
    "oos_sharpe_vs_hold",
    "premium_collected_annualized",
    "mean_model_prob_itm",
    "n_assignments",
    "n_rolls",
    "upside_forgone",
    "bootstrap_sharpe_p5",
    "bootstrap_sharpe_p50",
    "bootstrap_sharpe_p95",
    "bootstrap_worst_case_drawdown",
    "bootstrap_verdict",
    "model_priced",
    "skipped",
)


@dataclass(slots=True, frozen=True)
class OverlayScore:
    """Stitched in-sample / out-of-sample walk-forward score of one return series."""

    is_sharpe: float
    """Sharpe ratio of the stitched in-sample returns (all windows' IS legs)."""

    oos_sharpe: float
    """Sharpe ratio of the stitched out-of-sample returns (all windows' OOS tails)."""

    oos_max_drawdown: float
    """Max drawdown (<=0.0) of the stitched OOS returns."""

    oos_returns: pd.Series
    """The stitched OOS return series, in time order (fed to ``bootstrap_stress``)."""


def score_overlay(returns: pd.Series, wf: WalkForwardConfig) -> OverlayScore:
    """Apply v1's exact walk-forward window discipline to an already-realized
    daily return series.

    Unlike ``funnel.backtest.walkforward.walk_forward_oos`` (which recomputes
    an indicator-driven position series per-window to keep a strategy's
    indicators from warming up on pre-window history), there is no per-window
    recompute here: ``simulate_overlay`` already produced a fully causal,
    already-realized return series (every entry/roll/settlement decision at
    day t uses only ``close``/``vol`` at or before t — see
    ``options/overlays.py``'s module docstring). So, exactly as
    ``funnel.momentum.cross_sectional.walk_forward_score`` does for the
    cross-sectional portfolio's already-realized return series, slicing this
    series into ``wf.n_windows`` sequential windows and taking each window's
    trailing ``(1 - wf.is_fraction)`` tail as OOS is the directly comparable
    treatment — the window boundaries are identical in construction to the
    single-asset strategy sweep, just applied to a return series instead of
    re-running an indicator. Calling this on both ``OverlayResult.returns``
    and ``OverlayResult.underlying_returns`` scores the overlay and its
    buy-and-hold benchmark on window-identical boundaries, making
    ``oos_sharpe_vs_hold`` a fair like-for-like comparison.

    Raises ``InsufficientHistoryError`` if any window's OOS segment has fewer
    than ``MIN_OOS_ROWS`` valid (non-NaN) observations — the overlay return
    series is NaN during the vol-proxy warmup before the first position
    enters, so a short enough series can starve a window of real OOS data
    even though ``len(returns)`` looks adequate. The sweep runner below
    catches this and records the pair as skipped, mirroring
    ``walk_forward_oos``'s own skip criterion.
    """
    n_rows = len(returns)
    bounds = _window_bounds(n_rows, wf.n_windows)

    is_chunks: list[pd.Series] = []
    oos_chunks: list[pd.Series] = []
    for start, end in bounds:
        _, split, _ = _is_oos_split(start, end, wf.is_fraction)
        oos_chunk = returns.iloc[split:end]
        n_valid_oos = int(oos_chunk.notna().sum())
        if n_valid_oos < MIN_OOS_ROWS:
            raise InsufficientHistoryError(
                f"window [{start}, {end}) has only {n_valid_oos} valid OOS rows "
                f"(< {MIN_OOS_ROWS} needed)"
            )
        is_chunks.append(returns.iloc[start:split])
        oos_chunks.append(oos_chunk)

    stitched_is = pd.concat(is_chunks)
    stitched_oos = pd.concat(oos_chunks)

    return OverlayScore(
        is_sharpe=sharpe(stitched_is),
        oos_sharpe=sharpe(stitched_oos),
        oos_max_drawdown=max_drawdown(stitched_oos),
        oos_returns=stitched_oos,
    )


def _spec_params_to_str(config: OverlayConfig) -> str:
    """Render an overlay spec's parameters as a compact, deterministic string."""
    spec = config.spec
    fields = {
        "dte_target": spec.dte_target,
        "selector_mode": spec.strike_selector.mode,
        "selector_value": spec.strike_selector.value,
        "roll_at_dte": spec.roll_at_dte,
        "avoid_assignment": spec.avoid_assignment,
        "assignment_prob_trigger": spec.assignment_prob_trigger,
        "spread_width_pct": spec.spread_width_pct,
        "kind": spec.kind.value,
        "contracts": spec.contracts,
    }
    return ",".join(f"{k}={v}" for k, v in sorted(fields.items()))


def _skipped_row(config: OverlayConfig, symbol: str) -> dict[str, object]:
    return {
        "config_name": config.name,
        "structure": config.spec.structure.value,
        "symbol": symbol,
        "spec_params": _spec_params_to_str(config),
        "overlay_is_sharpe": float("nan"),
        "overlay_oos_sharpe": float("nan"),
        "overlay_oos_max_drawdown": float("nan"),
        "underlying_oos_sharpe": float("nan"),
        "underlying_oos_max_drawdown": float("nan"),
        "oos_sharpe_vs_hold": float("nan"),
        "premium_collected_annualized": float("nan"),
        "mean_model_prob_itm": float("nan"),
        "n_assignments": 0,
        "n_rolls": 0,
        "upside_forgone": float("nan"),
        "bootstrap_sharpe_p5": float("nan"),
        "bootstrap_sharpe_p50": float("nan"),
        "bootstrap_sharpe_p95": float("nan"),
        "bootstrap_worst_case_drawdown": float("nan"),
        "bootstrap_verdict": "skipped",
        "model_priced": True,
        "skipped": True,
    }


def _score_overlay_pair(
    config: OverlayConfig,
    symbol: str,
    df: pd.DataFrame,
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    rate: float,
    n_bootstrap: int,
    seed: int,
    dd_floor: float,
) -> dict[str, object]:
    """Score one (overlay config, symbol) pair: simulate + walk-forward + bootstrap.

    A pure function of its arguments, called identically by the serial
    in-process loop and by a parallel worker process — see
    ``funnel.backtest.sweep._score_pair`` for the identical design rationale.
    """
    try:
        result = simulate_overlay(df, config.spec, vol_config, costs, rate)
        overlay_score = score_overlay(result.returns, wf)
        underlying_score = score_overlay(result.underlying_returns, wf)
    except (UndefinedRiskError, InsufficientHistoryError):  # fmt: skip
        return _skipped_row(config, symbol)

    bootstrap = bootstrap_stress(
        overlay_score.oos_returns, n_bootstrap, seed=seed, dd_floor=dd_floor
    )

    return {
        "config_name": config.name,
        "structure": config.spec.structure.value,
        "symbol": symbol,
        "spec_params": _spec_params_to_str(config),
        "overlay_is_sharpe": overlay_score.is_sharpe,
        "overlay_oos_sharpe": overlay_score.oos_sharpe,
        "overlay_oos_max_drawdown": overlay_score.oos_max_drawdown,
        "underlying_oos_sharpe": underlying_score.oos_sharpe,
        "underlying_oos_max_drawdown": underlying_score.oos_max_drawdown,
        "oos_sharpe_vs_hold": overlay_score.oos_sharpe - underlying_score.oos_sharpe,
        "premium_collected_annualized": result.premium_collected_annualized,
        "mean_model_prob_itm": result.mean_prob_itm_at_entry,
        "n_assignments": len(result.events),
        "n_rolls": result.n_rolls,
        "upside_forgone": result.upside_forgone,
        "bootstrap_sharpe_p5": bootstrap.sharpe_p5,
        "bootstrap_sharpe_p50": bootstrap.sharpe_p50,
        "bootstrap_sharpe_p95": bootstrap.sharpe_p95,
        "bootstrap_worst_case_drawdown": bootstrap.worst_case_drawdown,
        "bootstrap_verdict": bootstrap.verdict,
        "model_priced": True,
        "skipped": False,
    }


def _score_overlay_symbol(
    symbol: str,
    df: pd.DataFrame,
    configs: list[OverlayConfig],
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    rate: float,
    n_bootstrap: int,
    seed: int,
    dd_floor: float,
) -> list[dict[str, object]]:
    """Score every overlay config against one symbol's OHLCV frame.

    Module-level and picklable (see
    ``funnel.backtest.sweep._score_symbol``) — the per-asset chunk
    ``_run_overlay_sweep_parallel`` dispatches to each worker.
    """
    return [
        _score_overlay_pair(
            config, symbol, df, wf, vol_config, costs, rate, n_bootstrap, seed, dd_floor
        )
        for config in configs
    ]


def _resolve_n_workers(n_workers: int | None, n_symbols: int) -> int:
    """Resolve the requested worker count — identical policy to
    ``funnel.backtest.sweep._resolve_n_workers``."""
    resolved = n_workers if n_workers is not None else (os.process_cpu_count() or 1)
    return max(1, min(resolved, max(n_symbols, 1)))


def _run_overlay_sweep_serial(
    all_symbols: list[str],
    data: Mapping[str, pd.DataFrame],
    configs: list[OverlayConfig],
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    rate: float,
    n_bootstrap: int,
    seed: int,
    dd_floor: float,
    should_stop: Callable[[], bool] | None,
) -> pd.DataFrame:
    """Original single-process overlay sweep loop — the equivalence baseline.

    ``n_workers`` in ``{0, 1}`` (and the resolved-to-1 edge case) always
    takes this path verbatim.
    """
    rows: list[dict[str, object]] = []
    for symbol in all_symbols:
        df = data[symbol]
        for config in configs:
            if should_stop is not None and should_stop():
                raise RunCancelledError("run_overlay_sweep cancelled")
            rows.append(
                _score_overlay_pair(
                    config, symbol, df, wf, vol_config, costs, rate, n_bootstrap, seed, dd_floor
                )
            )
    return pd.DataFrame(rows, columns=list(OVERLAY_SWEEP_COLUMNS))


def _run_overlay_sweep_parallel(
    all_symbols: list[str],
    data: Mapping[str, pd.DataFrame],
    configs: list[OverlayConfig],
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    rate: float,
    n_bootstrap: int,
    seed: int,
    dd_floor: float,
    n_workers: int,
    should_stop: Callable[[], bool] | None,
) -> pd.DataFrame:
    """Per-symbol process-pool overlay sweep — mirrors
    ``funnel.backtest.sweep._run_sweep_parallel``: one task per symbol,
    results assembled in submission order, ``should_stop`` polled between
    completed futures, non-blocking teardown on cancellation/error.
    """
    executor = ProcessPoolExecutor(max_workers=n_workers)
    rows_by_symbol: dict[str, list[dict[str, object]]] = {}
    try:
        futures = {
            executor.submit(
                _score_overlay_symbol,
                symbol,
                data[symbol],
                configs,
                wf,
                vol_config,
                costs,
                rate,
                n_bootstrap,
                seed,
                dd_floor,
            ): symbol
            for symbol in all_symbols
        }
        pending = set(futures)
        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                rows_by_symbol[futures[future]] = future.result()
            if should_stop is not None and should_stop():
                raise RunCancelledError("run_overlay_sweep cancelled")
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)

    rows: list[dict[str, object]] = []
    for symbol in all_symbols:
        rows.extend(rows_by_symbol[symbol])
    return pd.DataFrame(rows, columns=list(OVERLAY_SWEEP_COLUMNS))


def run_overlay_sweep(
    data: Mapping[str, pd.DataFrame],
    configs: list[OverlayConfig],
    symbols: list[str] | None,
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    rate: float,
    thresholds: FunnelThresholds,
    n_bootstrap: int = 200,
    seed: int = 42,
    should_stop: Callable[[], bool] | None = None,
    n_workers: int | None = None,
) -> pd.DataFrame:
    """Run every (overlay config, symbol) pair through simulation + walk-forward + bootstrap.

    ``symbols`` defaults to every key in ``data`` when ``None``. ``thresholds``
    supplies the bootstrap's drawdown survivability floor
    (``thresholds.max_dd_floor``, the same threshold the v1 funnel uses) so
    the overlay bootstrap's "fragile" verdict is on the identical floor as
    the rest of the system.

    A pair is recorded as a skipped row (``skipped=True``, all metrics
    ``NaN``) rather than raising when ``simulate_overlay`` rejects the spec
    (``UndefinedRiskError`` — should not occur for configs drawn from
    ``build_overlay_grid``, which are pre-validated at construction, but
    guarded here defensively) or when the symbol's history is too short for
    a meaningful walk-forward split (``InsufficientHistoryError``, raised by
    ``score_overlay``).

    ``n_workers`` mirrors ``funnel.backtest.sweep.run_sweep``'s parallelism
    design (per-asset ``ProcessPoolExecutor`` chunking, ``None`` -> resolves
    to ``os.process_cpu_count()`` capped at the symbol count, ``0``/``1`` ->
    original serial loop unchanged for the equivalence baseline).

    ``should_stop``, if given, is checked once per (config, symbol)
    iteration in the serial path, or once per completed per-symbol task in
    the parallel path — same discipline and rationale as
    ``funnel.backtest.sweep.run_sweep``. When it returns ``True``,
    ``RunCancelledError`` is raised immediately and no rows (nor
    ``overlay_results.csv``) are written for this run.
    """
    all_symbols = list(data.keys()) if symbols is None else symbols
    total = len(configs) * len(all_symbols)
    logger.info(
        "%d overlay configs x %d symbols = %d overlay backtests",
        len(configs),
        len(all_symbols),
        total,
    )
    print(
        f"{len(configs)} overlay configs x {len(all_symbols)} symbols = {total} overlay backtests"
    )

    dd_floor = thresholds.max_dd_floor

    if n_workers == 0 or n_workers == 1:
        return _run_overlay_sweep_serial(
            all_symbols,
            data,
            configs,
            wf,
            vol_config,
            costs,
            rate,
            n_bootstrap,
            seed,
            dd_floor,
            should_stop,
        )

    resolved_workers = _resolve_n_workers(n_workers, len(all_symbols))
    if resolved_workers <= 1:
        return _run_overlay_sweep_serial(
            all_symbols,
            data,
            configs,
            wf,
            vol_config,
            costs,
            rate,
            n_bootstrap,
            seed,
            dd_floor,
            should_stop,
        )

    return _run_overlay_sweep_parallel(
        all_symbols,
        data,
        configs,
        wf,
        vol_config,
        costs,
        rate,
        n_bootstrap,
        seed,
        dd_floor,
        resolved_workers,
        should_stop,
    )


def write_overlay_results(df: pd.DataFrame, path: Path) -> None:
    """Write the overlay sweep DataFrame to ``path`` as CSV (``overlay_results.csv``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
