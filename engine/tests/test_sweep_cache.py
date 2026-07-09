"""Tests for funnel.backtest.sweep_cache: threshold-independent sweep-metrics
cache (PERF-2).

Parity: a cache miss followed by a cache hit under a *different* threshold
set must reproduce exactly what a fresh (uncached) ``run_sweep`` call would
have produced for each threshold set. Staleness: any change to the data,
walk-forward config, cost model, grid, or the engine-version/schema salt
must produce a different fingerprint (a guaranteed miss, never a stale hit).
"""

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import funnel.backtest.sweep_cache as sweep_cache_module
from funnel.backtest.sweep import run_sweep
from funnel.backtest.sweep_cache import fingerprint_sweep_inputs, run_sweep_cached
from funnel.config import CostModel, FunnelThresholds, WalkForwardConfig
from funnel.data.universe import AssetClass
from funnel.strategies.base import Category
from funnel.strategies.grid import StrategyConfig
from funnel.strategies.meanrev import rsi_revert
from funnel.strategies.trend import ma_crossover, time_series_momentum


def _make_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=n)
    close = 100.0 + np.cumsum(rng.normal(loc=0.02, scale=1.0, size=n))
    return pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1e6},
        index=index,
    ).astype("float64")


@pytest.fixture
def configs() -> list[StrategyConfig]:
    return [
        StrategyConfig(
            "ma_5_20", "ma_crossover", ma_crossover, {"fast": 5, "slow": 20}, Category.TREND
        ),
        StrategyConfig(
            "tsm_60", "time_series_momentum", time_series_momentum, {"lookback": 60}, Category.TREND
        ),
        StrategyConfig(
            "rsi_14",
            "rsi_revert",
            rsi_revert,
            {"window": 14, "oversold": 30.0, "overbought": 70.0},
            Category.MEAN_REVERSION,
        ),
    ]


@pytest.fixture
def data() -> dict[str, pd.DataFrame]:
    return {"AAA": _make_df(700, seed=1), "BBB": _make_df(700, seed=2)}


@pytest.fixture
def asset_classes() -> dict[str, AssetClass]:
    return {"AAA": AssetClass.LARGE_CAP, "BBB": AssetClass.CRYPTO}


# ---------------------------------------------------------------------------
# Parity: cache miss + hit (under different thresholds) match fresh runs
# ---------------------------------------------------------------------------


def test_cache_miss_then_hit_under_different_thresholds_matches_fresh_runs(
    tmp_path: Path,
    configs: list[StrategyConfig],
    data: dict[str, pd.DataFrame],
    asset_classes: dict[str, AssetClass],
) -> None:
    wf = WalkForwardConfig()
    costs = CostModel()
    cache_dir = tmp_path / "cache"

    thresholds_a = FunnelThresholds()
    # Deliberately as permissive as possible so its survivor set is very
    # unlikely to coincide with the (much stricter) default thresholds_a --
    # this test needs the two threshold sets to actually disagree on at
    # least one row to meaningfully exercise verdict reapplication.
    thresholds_b = FunnelThresholds(
        max_dd_floor=-1.0,
        min_oos_sharpe=-10.0,
        max_oos_sharpe=100.0,
        max_oos_is_ratio=100.0,
        min_trades=0,
        require_positive_is_sharpe=False,
    )

    miss_result = run_sweep_cached(data, configs, asset_classes, wf, thresholds_a, costs, cache_dir)
    assert miss_result.cache_hit is False
    assert miss_result.metrics_elapsed_s > 0.0

    hit_result = run_sweep_cached(data, configs, asset_classes, wf, thresholds_b, costs, cache_dir)
    assert hit_result.cache_hit is True
    assert hit_result.metrics_elapsed_s == 0.0
    assert hit_result.fingerprint == miss_result.fingerprint

    fresh_a = run_sweep(data, configs, asset_classes, wf, thresholds_a, costs)
    fresh_b = run_sweep(data, configs, asset_classes, wf, thresholds_b, costs)

    pd.testing.assert_frame_equal(
        miss_result.sweep_df.reset_index(drop=True), fresh_a.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        hit_result.sweep_df.reset_index(drop=True), fresh_b.reset_index(drop=True)
    )
    # The two threshold sets must actually produce different survivor sets
    # here -- otherwise this test would pass trivially without exercising
    # the verdict-reapplication path at all.
    assert not fresh_a["survived"].equals(fresh_b["survived"])


def test_cache_writes_parquet_and_json_sidecar(
    tmp_path: Path,
    configs: list[StrategyConfig],
    data: dict[str, pd.DataFrame],
    asset_classes: dict[str, AssetClass],
) -> None:
    cache_dir = tmp_path / "cache"
    result = run_sweep_cached(
        data,
        configs,
        asset_classes,
        WalkForwardConfig(),
        FunnelThresholds(),
        CostModel(),
        cache_dir,
    )
    parquet_path = cache_dir / f"sweep_metrics_{result.fingerprint}.parquet"
    json_path = cache_dir / f"sweep_metrics_{result.fingerprint}.json"
    assert parquet_path.is_file()
    assert json_path.is_file()


# ---------------------------------------------------------------------------
# Staleness: every input that affects metrics changes the fingerprint
# ---------------------------------------------------------------------------


def test_fingerprint_structurally_excludes_thresholds() -> None:
    sig = inspect.signature(fingerprint_sweep_inputs)
    assert "thresholds" not in sig.parameters


def test_fingerprint_is_deterministic(
    configs: list[StrategyConfig], data: dict[str, pd.DataFrame]
) -> None:
    wf = WalkForwardConfig()
    costs = CostModel()
    assert fingerprint_sweep_inputs(data, configs, wf, costs) == fingerprint_sweep_inputs(
        data, configs, wf, costs
    )


def test_fingerprint_changes_on_data_perturbation(
    configs: list[StrategyConfig], data: dict[str, pd.DataFrame]
) -> None:
    wf = WalkForwardConfig()
    costs = CostModel()
    base = fingerprint_sweep_inputs(data, configs, wf, costs)

    perturbed = {k: v.copy() for k, v in data.items()}
    perturbed["AAA"].iloc[10, perturbed["AAA"].columns.get_loc("close")] += 0.01
    assert fingerprint_sweep_inputs(perturbed, configs, wf, costs) != base


def test_fingerprint_changes_on_wf_change(
    configs: list[StrategyConfig], data: dict[str, pd.DataFrame]
) -> None:
    costs = CostModel()
    base = fingerprint_sweep_inputs(data, configs, WalkForwardConfig(), costs)
    changed = fingerprint_sweep_inputs(data, configs, WalkForwardConfig(n_windows=4), costs)
    assert base != changed


def test_fingerprint_changes_on_cost_change(
    configs: list[StrategyConfig], data: dict[str, pd.DataFrame]
) -> None:
    wf = WalkForwardConfig()
    base = fingerprint_sweep_inputs(data, configs, wf, CostModel())
    changed = fingerprint_sweep_inputs(data, configs, wf, CostModel(default_bps_per_side=2.0))
    assert base != changed


def test_fingerprint_changes_on_grid_change(
    configs: list[StrategyConfig], data: dict[str, pd.DataFrame]
) -> None:
    wf = WalkForwardConfig()
    costs = CostModel()
    base = fingerprint_sweep_inputs(data, configs, wf, costs)
    changed = fingerprint_sweep_inputs(data, configs[:-1], wf, costs)
    assert base != changed


def test_fingerprint_changes_on_schema_bump(
    monkeypatch: pytest.MonkeyPatch, configs: list[StrategyConfig], data: dict[str, pd.DataFrame]
) -> None:
    wf = WalkForwardConfig()
    costs = CostModel()
    base = fingerprint_sweep_inputs(data, configs, wf, costs)

    monkeypatch.setattr(sweep_cache_module, "COMPUTE_CACHE_SCHEMA", 999)
    bumped = fingerprint_sweep_inputs(data, configs, wf, costs)
    assert bumped != base


def test_fingerprint_changes_on_engine_version_bump(
    monkeypatch: pytest.MonkeyPatch, configs: list[StrategyConfig], data: dict[str, pd.DataFrame]
) -> None:
    wf = WalkForwardConfig()
    costs = CostModel()
    base = fingerprint_sweep_inputs(data, configs, wf, costs)

    monkeypatch.setattr(sweep_cache_module, "FUNNEL_VERSION", "999.0.0")
    bumped = fingerprint_sweep_inputs(data, configs, wf, costs)
    assert bumped != base


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


def test_run_sweep_cached_evicts_oldest_entries_past_cap(
    tmp_path: Path,
    configs: list[StrategyConfig],
    asset_classes: dict[str, AssetClass],
) -> None:
    cache_dir = tmp_path / "cache"
    wf = WalkForwardConfig()
    costs = CostModel()
    thresholds = FunnelThresholds()

    # Write three distinct-fingerprint entries directly (different data each
    # time), then assert eviction to keep=2 by calling evict_oldest with a
    # small cap explicitly (unit-level check of the eviction call itself).
    for seed in (1, 2, 3):
        distinct_data = {"AAA": _make_df(700, seed=seed)}
        distinct_asset_classes = {"AAA": AssetClass.LARGE_CAP}
        run_sweep_cached(
            distinct_data, configs, distinct_asset_classes, wf, thresholds, costs, cache_dir
        )

    from funnel.compute_cache import evict_oldest

    evict_oldest(cache_dir, "sweep_metrics_*.parquet", keep=2)
    remaining = list(cache_dir.glob("sweep_metrics_*.parquet"))
    assert len(remaining) == 2
