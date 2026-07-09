"""Tests for the PERF-2 compute cache wired into ``funnel.pipeline.run_pipeline``.

Uses the same small, deterministic synthetic setup as ``test_pipeline.py``
(kept self-contained rather than imported, matching this test suite's
existing per-file fixture convention) but a shrunk universe subset via
``PipelineConfig.configs`` isn't enough to shrink the *data* stage (that
always fetches the full ``ASSET_UNIVERSE``), so these tests reuse the same
``SyntheticTestSource``/``_small_grid`` shapes as ``test_pipeline.py``.
"""

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from funnel.config import CostModel, FunnelThresholds, WalkForwardConfig
from funnel.data.sources import DataSource
from funnel.pipeline import PipelineConfig, run_pipeline
from funnel.profiles.models import Profile, SliderValues
from funnel.strategies.base import Category
from funnel.strategies.grid import StrategyConfig
from funnel.strategies.meanrev import zscore_revert
from funnel.strategies.trend import ma_crossover

N_ROWS = 1080


def _seed_for(symbol: str) -> int:
    return sum(ord(c) for c in symbol) * 7919 % (2**32)


class _SyntheticSource(DataSource):
    def fetch(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        rng = np.random.default_rng(_seed_for(symbol))
        n = N_ROWS
        index = pd.bdate_range("2018-01-01", periods=n)
        drift = rng.normal(loc=0.0004, scale=0.0002)
        daily_returns = rng.normal(loc=drift, scale=0.012, size=n)
        close = 100.0 * np.cumprod(1.0 + daily_returns)
        daily_range = np.abs(rng.normal(loc=0.5, scale=0.2, size=n)) + 0.05
        open_ = close + rng.normal(loc=0.0, scale=0.1, size=n)
        high = np.maximum(open_, close) + daily_range
        low = np.minimum(open_, close) - daily_range
        volume = np.abs(rng.normal(loc=1_000_000.0, scale=200_000.0, size=n))
        return pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=index,
        ).astype("float64")


def _small_grid() -> list[StrategyConfig]:
    return [
        StrategyConfig(
            name="ma_crossover_10_50",
            family="ma_crossover",
            fn=ma_crossover,
            params={"fast": 10, "slow": 50},
            category=Category.TREND,
        ),
        StrategyConfig(
            name="zscore_revert_20_1.5",
            family="zscore_revert",
            fn=zscore_revert,
            params={"window": 20, "threshold": 1.5},
            category=Category.MEAN_REVERSION,
        ),
    ]


def _profile() -> Profile:
    return Profile(
        name="cache-test-profile",
        sliders=SliderValues(capital=50, risk_tolerance=50, time_horizon=50, drawdown_tolerance=50),
        created_at="2026-07-03",
        preset=False,
    )


def _base_config(cache_dir: Path) -> PipelineConfig:
    return PipelineConfig(
        profile=_profile(),
        wf=WalkForwardConfig(),
        base_thresholds=FunnelThresholds(),
        costs=CostModel(),
        n_bootstrap=5,
        configs=_small_grid(),
        cache_dir=cache_dir,
    )


def test_report_json_has_compute_cache_keys(tmp_path: Path) -> None:
    config = _base_config(tmp_path / "cache")
    result = run_pipeline(config, _SyntheticSource(), tmp_path / "runs", "run-cache-1")
    cc = result.report["compute_cache"]
    assert cc["sweep"] == "miss"
    assert cc["regime"] == "miss"
    assert isinstance(cc["fingerprint_prefix"], str)
    assert cc["fingerprint_prefix"] != ""


def test_threshold_only_rerun_hits_cache_and_matches_fresh_thresholds(tmp_path: Path) -> None:
    source = _SyntheticSource()
    cache_dir = tmp_path / "cache"

    config_a = _base_config(cache_dir)
    result_a = run_pipeline(config_a, source, tmp_path / "runs", "run-a")
    assert result_a.report["compute_cache"]["sweep"] == "miss"
    assert result_a.report["compute_cache"]["regime"] == "miss"

    # Same data/grid/wf/costs, a different threshold set -> cache hit, and
    # the fingerprint (same underlying data) must match run A's.
    config_b = replace(config_a, base_thresholds=FunnelThresholds(min_oos_sharpe=1.0, min_trades=5))
    result_b = run_pipeline(config_b, source, tmp_path / "runs", "run-b")
    assert result_b.report["compute_cache"]["sweep"] == "hit"
    assert result_b.report["compute_cache"]["regime"] == "hit"
    assert (
        result_b.report["compute_cache"]["fingerprint_prefix"]
        == result_a.report["compute_cache"]["fingerprint_prefix"]
    )

    # The hit path must reproduce the same numbers a fresh run under the
    # same (different) thresholds would -- i.e. reapplying verdicts from
    # cached metrics is not just "some cached answer", it's the *correct*
    # answer for config_b's thresholds.
    config_b_fresh = replace(config_b, use_compute_cache=False)
    result_b_fresh = run_pipeline(config_b_fresh, source, tmp_path / "runs-fresh", "run-b-fresh")
    sweep_b = pd.read_csv(result_b.run_dir / "sweep_results.csv")
    sweep_b_fresh = pd.read_csv(result_b_fresh.run_dir / "sweep_results.csv")
    pd.testing.assert_frame_equal(sweep_b, sweep_b_fresh)
    assert cache_dir.exists()


def test_use_compute_cache_false_forces_fresh_and_ignores_stale_cache(tmp_path: Path) -> None:
    """A forced-fresh run (``use_compute_cache=False``) must neither read
    nor be corrupted by a pre-existing (even a deliberately wrong) cache
    entry at the fingerprint its inputs would hash to."""
    source = _SyntheticSource()
    cache_dir = tmp_path / "cache"

    config = _base_config(cache_dir)
    real_result = run_pipeline(config, source, tmp_path / "runs", "run-real")
    fingerprint = real_result.report["compute_cache"]["fingerprint_prefix"]

    # Corrupt the cache entry on disk: any bypassed run must not be able to
    # observe this (a real cache-reading run would visibly break/mismatch).
    cache_files = list(cache_dir.glob("sweep_metrics_*.parquet"))
    assert cache_files
    for path in cache_files:
        df = pd.read_parquet(path)
        df["is_sharpe"] = 999.0
        df.to_parquet(path, index=False)

    bypass_config = replace(config, use_compute_cache=False)
    bypass_result = run_pipeline(bypass_config, source, tmp_path / "runs-bypass", "run-bypass")
    assert bypass_result.report["compute_cache"]["sweep"] == "bypassed"
    assert bypass_result.report["compute_cache"]["regime"] == "bypassed"
    assert bypass_result.report["compute_cache"]["fingerprint_prefix"] == ""

    sweep_bypass = pd.read_csv(bypass_result.run_dir / "sweep_results.csv")
    assert not (sweep_bypass["is_sharpe"] == 999.0).any()

    # Meanwhile a *cached* run at the same fingerprint would have picked up
    # the corruption -- confirms the corruption test setup is meaningful.
    cached_result = run_pipeline(config, source, tmp_path / "runs-cached", "run-cached")
    assert cached_result.report["compute_cache"]["fingerprint_prefix"] == fingerprint
    sweep_cached = pd.read_csv(cached_result.run_dir / "sweep_results.csv")
    assert (sweep_cached["is_sharpe"] == 999.0).any()
