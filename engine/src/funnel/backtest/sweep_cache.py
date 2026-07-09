"""Threshold-independent sweep-metrics cache (PERF-2).

``run_sweep``'s expensive output per (config, symbol) pair -- ``is_sharpe``,
``oos_sharpe``, ``oos_max_drawdown``, ``oos_trade_count`` (all produced by
``walk_forward_oos``) -- does NOT depend on ``FunnelThresholds``. Only the
six pass/fail verdict columns (``apply_funnel``) depend on thresholds, and
that is pure cheap arithmetic over already-computed scalars. So a
profile/slider change (new thresholds) on an otherwise-identical (data,
grid, walk-forward config, cost model) does not need to re-run a single
walk-forward backtest -- it only needs to re-apply the funnel filters to
cached metrics.

This module caches the metrics-only sweep table (everything in
``SWEEP_COLUMNS`` except the threshold-derived verdict columns) to a parquet
file keyed by a fingerprint of the five things the metrics actually depend
on: the engine version/schema (``funnel.compute_cache``), the input data,
the strategy grid, the walk-forward config, and the cost model.
``run_sweep_cached`` is a drop-in alternative to ``run_sweep`` that returns
the identical full sweep DataFrame (metrics + verdicts) either way -- the
only difference is whether the expensive half was read from disk.

Honesty/staleness: the cache key hashes the *actual* per-symbol OHLCV
content (not just symbol names or shapes) plus a canonical repr of every
config's name/family/params, the walk-forward window/split, the cost
model's fields, and ``funnel.__version__`` + ``COMPUTE_CACHE_SCHEMA``. Any
change to any of these -- new data, a different grid, a different
walk-forward split, different costs, or a code change to sweep/walk-forward
semantics that bumps the schema -- produces a different key and therefore a
guaranteed cache miss (never a stale hit).
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from funnel import __version__ as FUNNEL_VERSION
from funnel.backtest.sweep import SWEEP_COLUMNS, _params_to_str, run_sweep
from funnel.compute_cache import (
    COMPUTE_CACHE_SCHEMA,
    evict_oldest,
    hash_dataframe,
    write_cache_metadata,
)
from funnel.config import CostModel, FunnelThresholds, WalkForwardConfig
from funnel.data.universe import AssetClass
from funnel.strategies.grid import StrategyConfig

logger = logging.getLogger(__name__)

METRIC_COLUMNS: tuple[str, ...] = (
    "config_name",
    "family",
    "category",
    "params",
    "symbol",
    "is_sharpe",
    "oos_sharpe",
    "oos_max_drawdown",
    "oos_trade_count",
    "skipped",
)
"""The threshold-independent subset of SWEEP_COLUMNS -- everything computed
by walk_forward_oos, none of the six apply_funnel verdict columns."""

VERDICT_COLUMNS: tuple[str, ...] = (
    "passes_max_dd_floor",
    "passes_min_oos_sharpe",
    "passes_max_oos_sharpe",
    "passes_overfit_gap",
    "passes_min_trades",
    "passes_positive_is_sharpe",
    "survived",
)


@dataclass(slots=True, frozen=True)
class CacheResult:
    sweep_df: pd.DataFrame
    cache_hit: bool
    fingerprint: str
    metrics_elapsed_s: float
    """Time spent producing the metrics half (walk_forward_oos for every
    pair) -- 0.0 on a cache hit, since that work was skipped entirely."""
    verdict_elapsed_s: float
    """Time spent applying the six threshold-derived filter columns --
    always paid, on both hit and miss, but is cheap vectorized arithmetic."""


def fingerprint_sweep_inputs(
    data: Mapping[str, pd.DataFrame],
    configs: list[StrategyConfig],
    wf: WalkForwardConfig,
    costs: CostModel,
) -> str:
    """Fingerprint of everything the *metrics* half of a sweep depends on.

    Deliberately excludes ``FunnelThresholds`` -- that is the entire point
    (threshold-independence). Includes the cost model (it feeds
    ``cost_bps_for``, which changes the per-pair backtest arithmetic), the
    walk-forward config (window count / split changes what gets computed),
    and an engine-version salt (see ``funnel.compute_cache``) so a code
    change to sweep/walk-forward semantics invalidates every prior entry.
    """
    h = hashlib.sha256()
    h.update(f"schema={COMPUTE_CACHE_SCHEMA}".encode())
    h.update(f"funnel_version={FUNNEL_VERSION}".encode())

    for symbol in sorted(data):
        h.update(symbol.encode())
        h.update(hash_dataframe(data[symbol]).encode())

    for config in sorted(configs, key=lambda c: c.name):
        h.update(config.name.encode())
        h.update(config.family.encode())
        h.update(config.category.value.encode())
        h.update(_params_to_str(config.params).encode())

    h.update(repr(wf).encode())
    h.update(repr(costs).encode())
    return h.hexdigest()[:32]


def _apply_funnel_vectorized(metrics: pd.DataFrame, thresholds: FunnelThresholds) -> pd.DataFrame:
    """Vectorized equivalent of calling ``apply_funnel`` row-by-row.

    Produces byte-identical verdict columns to
    ``funnel.backtest.funnel.apply_funnel`` (same six comparisons, same
    filter-4 is_sharpe<=0 special case), just computed as whole-column numpy
    comparisons instead of a Python loop -- correctness is covered by the
    parity test in ``test_sweep_cache.py`` (asserts full equality against
    the real ``run_sweep`` output on the same inputs).
    """
    out = metrics.copy()
    is_sharpe = out["is_sharpe"].to_numpy()
    oos_sharpe = out["oos_sharpe"].to_numpy()
    oos_dd = out["oos_max_drawdown"].to_numpy()
    oos_trades = out["oos_trade_count"].to_numpy()
    skipped = out["skipped"].to_numpy()

    passes_max_dd_floor = oos_dd > thresholds.max_dd_floor
    passes_min_oos_sharpe = oos_sharpe > thresholds.min_oos_sharpe
    passes_max_oos_sharpe = oos_sharpe < thresholds.max_oos_sharpe

    is_positive = is_sharpe > 0
    passes_overfit_gap = np.where(
        is_positive, oos_sharpe <= is_sharpe * thresholds.max_oos_is_ratio, False
    )
    passes_min_trades = oos_trades >= thresholds.min_trades
    if thresholds.require_positive_is_sharpe:
        passes_positive_is_sharpe = is_positive
    else:
        passes_positive_is_sharpe = np.full(len(out), True)

    survived = (
        passes_max_dd_floor
        & passes_min_oos_sharpe
        & passes_max_oos_sharpe
        & passes_overfit_gap
        & passes_min_trades
        & passes_positive_is_sharpe
    )

    # Skipped rows are never survivors and always fail every filter --
    # matches _skipped_row in sweep.py exactly (all False, skipped=True).
    for arr in (
        passes_max_dd_floor,
        passes_min_oos_sharpe,
        passes_max_oos_sharpe,
        passes_overfit_gap,
        passes_min_trades,
        passes_positive_is_sharpe,
        survived,
    ):
        arr[skipped] = False

    out["passes_max_dd_floor"] = passes_max_dd_floor
    out["passes_min_oos_sharpe"] = passes_min_oos_sharpe
    out["passes_max_oos_sharpe"] = passes_max_oos_sharpe
    out["passes_overfit_gap"] = passes_overfit_gap
    out["passes_min_trades"] = passes_min_trades
    out["passes_positive_is_sharpe"] = passes_positive_is_sharpe
    out["survived"] = survived
    return out[list(SWEEP_COLUMNS)]


def _cache_path(cache_dir: Path, fingerprint: str) -> Path:
    return cache_dir / f"sweep_metrics_{fingerprint}.parquet"


def run_sweep_cached(
    data: Mapping[str, pd.DataFrame],
    configs: list[StrategyConfig],
    asset_classes: Mapping[str, AssetClass],
    wf: WalkForwardConfig,
    thresholds: FunnelThresholds,
    costs: CostModel,
    cache_dir: Path,
    should_stop: Callable[[], bool] | None = None,
    n_workers: int | None = None,
) -> CacheResult:
    """Drop-in alternative to ``run_sweep`` with a threshold-independent
    metrics cache on disk under ``cache_dir``.

    On a cache hit (same data/grid/wf/costs/engine-version fingerprint as a
    prior run, thresholds may differ), skips every ``walk_forward_oos`` call
    entirely and only re-applies the funnel filters. On a miss, runs the
    full sweep via ``run_sweep`` (metrics + verdicts for the given
    thresholds in one pass, since that is cheaper than a second pass) and
    persists the metrics-only half for next time, evicting the oldest entry
    if the cache is at capacity (``funnel.compute_cache.MAX_ENTRIES_PER_KIND``).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = fingerprint_sweep_inputs(data, configs, wf, costs)
    path = _cache_path(cache_dir, fingerprint)

    if path.exists():
        t0 = time.perf_counter()
        metrics = pd.read_parquet(path)
        t1 = time.perf_counter()
        sweep_df = _apply_funnel_vectorized(metrics, thresholds)
        t2 = time.perf_counter()
        logger.info("sweep cache HIT fingerprint=%s (read %.3fs)", fingerprint, t1 - t0)
        return CacheResult(
            sweep_df=sweep_df,
            cache_hit=True,
            fingerprint=fingerprint,
            metrics_elapsed_s=0.0,
            verdict_elapsed_s=t2 - t1,
        )

    t0 = time.perf_counter()
    sweep_df = run_sweep(
        data, configs, asset_classes, wf, thresholds, costs, should_stop, n_workers
    )
    t1 = time.perf_counter()
    metrics = sweep_df[list(METRIC_COLUMNS)]
    metrics.to_parquet(path, index=False)
    write_cache_metadata(
        path.with_suffix(".json"),
        fingerprint,
        {"funnel_version": FUNNEL_VERSION, "n_rows": len(metrics)},
    )
    evict_oldest(cache_dir, "sweep_metrics_*.parquet")
    logger.info("sweep cache MISS fingerprint=%s (computed+wrote %.3fs)", fingerprint, t1 - t0)
    return CacheResult(
        sweep_df=sweep_df,
        cache_hit=False,
        fingerprint=fingerprint,
        metrics_elapsed_s=t1 - t0,
        verdict_elapsed_s=0.0,
    )
