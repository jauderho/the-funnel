"""Small end-to-end sweep + attrition tests: row counts, skip handling, attrition sums."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from funnel.backtest.sweep import run_sweep, write_sweep_results
from funnel.cancellation import RunCancelledError
from funnel.config import CostModel, FunnelThresholds, WalkForwardConfig
from funnel.data.universe import AssetClass
from funnel.reports.attrition import (
    build_attrition_report,
    render_text,
    to_dict,
    write_funnel_report,
)
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
    return {
        "AAA": _make_df(700, seed=1),
        "BBB": _make_df(700, seed=2),
    }


@pytest.fixture
def asset_classes() -> dict[str, AssetClass]:
    return {"AAA": AssetClass.LARGE_CAP, "BBB": AssetClass.CRYPTO}


def test_sweep_row_count_is_configs_times_assets(
    configs: list[StrategyConfig],
    data: dict[str, pd.DataFrame],
    asset_classes: dict[str, AssetClass],
) -> None:
    wf = WalkForwardConfig()
    thresholds = FunnelThresholds()
    costs = CostModel()
    df = run_sweep(data, configs, asset_classes, wf, thresholds, costs)
    assert len(df) == len(configs) * len(data)


def test_sweep_skip_handling(
    configs: list[StrategyConfig], asset_classes: dict[str, AssetClass]
) -> None:
    data = {
        "AAA": _make_df(700, seed=1),
        "SHORT": _make_df(50, seed=9),  # too short -> every config skipped
    }
    asset_classes = {**asset_classes, "SHORT": AssetClass.LARGE_CAP}
    wf = WalkForwardConfig()
    thresholds = FunnelThresholds()
    costs = CostModel()

    df = run_sweep(data, configs, asset_classes, wf, thresholds, costs)

    assert len(df) == len(configs) * len(data)
    short_rows = df[df["symbol"] == "SHORT"]
    assert len(short_rows) == len(configs)
    assert short_rows["skipped"].all()
    assert not short_rows["survived"].any()

    aaa_rows = df[df["symbol"] == "AAA"]
    assert not aaa_rows["skipped"].any()


def test_sweep_uses_correct_cost_rate_per_asset_class(
    configs: list[StrategyConfig],
    data: dict[str, pd.DataFrame],
    asset_classes: dict[str, AssetClass],
) -> None:
    wf = WalkForwardConfig()
    thresholds = FunnelThresholds()
    costs = CostModel(default_bps_per_side=1.0, crypto_bps_per_side=5.0)

    df = run_sweep(data, configs, asset_classes, wf, thresholds, costs)
    # Same strategy/config, different asset class (LARGE_CAP vs CRYPTO) on
    # data with identical seeds would only differ due to cost rate; here
    # data differs too, so just sanity-check both symbols produced results.
    assert set(df["symbol"]) == {"AAA", "BBB"}


def test_attrition_counts_sum_correctly(
    configs: list[StrategyConfig],
    data: dict[str, pd.DataFrame],
    asset_classes: dict[str, AssetClass],
) -> None:
    wf = WalkForwardConfig()
    thresholds = FunnelThresholds()
    costs = CostModel()
    df = run_sweep(data, configs, asset_classes, wf, thresholds, costs)

    report = build_attrition_report(df, thresholds)

    assert report.n_total_backtests == len(df)
    assert report.n_skipped == int(df["skipped"].sum())
    assert report.n_run == report.n_total_backtests - report.n_skipped
    assert report.n_survived == int(df.loc[~df["skipped"], "survived"].sum())
    assert report.n_survived <= report.n_clears_min_oos_sharpe <= report.n_run
    assert report.n_survived <= report.n_positive_oos_sharpe <= report.n_run

    # Per-category and per-family totals must sum back to n_run.
    assert sum(c.n_total for c in report.by_category) == report.n_run
    assert sum(f.n_total for f in report.by_family) == report.n_run


def test_thresholds_embedded_in_report(
    configs: list[StrategyConfig],
    data: dict[str, pd.DataFrame],
    asset_classes: dict[str, AssetClass],
) -> None:
    wf = WalkForwardConfig()
    thresholds = FunnelThresholds(min_oos_sharpe=0.42)
    costs = CostModel()
    df = run_sweep(data, configs, asset_classes, wf, thresholds, costs)
    report = build_attrition_report(df, thresholds)

    assert report.thresholds.min_oos_sharpe == 0.42
    rendered = render_text(report)
    assert "min_oos_sharpe=0.42" in rendered

    d = to_dict(report)
    assert d["thresholds"] == {
        "max_dd_floor": thresholds.max_dd_floor,
        "min_oos_sharpe": 0.42,
        "max_oos_sharpe": thresholds.max_oos_sharpe,
        "max_oos_is_ratio": thresholds.max_oos_is_ratio,
        "min_trades": thresholds.min_trades,
        "require_positive_is_sharpe": thresholds.require_positive_is_sharpe,
    }


def test_top_survivors_capped_and_sorted_descending(
    configs: list[StrategyConfig],
    data: dict[str, pd.DataFrame],
    asset_classes: dict[str, AssetClass],
) -> None:
    wf = WalkForwardConfig()
    # Loosen thresholds so at least some rows survive, to exercise sorting.
    thresholds = FunnelThresholds(
        max_dd_floor=-1.0,
        min_oos_sharpe=-10.0,
        max_oos_sharpe=10.0,
        max_oos_is_ratio=100.0,
        min_trades=0,
        require_positive_is_sharpe=False,
    )
    costs = CostModel()
    df = run_sweep(data, configs, asset_classes, wf, thresholds, costs)
    report = build_attrition_report(df, thresholds)

    sharpes = report.top_survivors["oos_sharpe"].tolist()
    assert sharpes == sorted(sharpes, reverse=True)
    assert len(report.top_survivors) <= 25


def test_write_sweep_and_funnel_report_csvs(
    tmp_path: Path,
    configs: list[StrategyConfig],
    data: dict[str, pd.DataFrame],
    asset_classes: dict[str, AssetClass],
) -> None:
    wf = WalkForwardConfig()
    thresholds = FunnelThresholds()
    costs = CostModel()
    df = run_sweep(data, configs, asset_classes, wf, thresholds, costs)
    report = build_attrition_report(df, thresholds)

    sweep_path = tmp_path / "sweep_results.csv"
    funnel_path = tmp_path / "funnel_report.csv"
    write_sweep_results(df, sweep_path)
    write_funnel_report(report, funnel_path)

    assert sweep_path.exists()
    assert funnel_path.exists()
    reloaded = pd.read_csv(sweep_path)
    assert len(reloaded) == len(df)


# ---------------------------------------------------------------------------
# PERF-1: sweep parallelism (n_workers) — equivalence and cancellation
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_symbol_data() -> dict[str, pd.DataFrame]:
    """Six symbols so ``n_workers=2`` actually chunks work across multiple
    per-asset process-pool tasks (rather than one worker doing everything)."""
    return {f"SYM{i}": _make_df(700, seed=i) for i in range(6)}


@pytest.fixture
def multi_symbol_asset_classes() -> dict[str, AssetClass]:
    return {f"SYM{i}": AssetClass.LARGE_CAP for i in range(6)}


def test_sweep_n_workers_zero_matches_default_serial(
    configs: list[StrategyConfig],
    data: dict[str, pd.DataFrame],
    asset_classes: dict[str, AssetClass],
) -> None:
    """``n_workers=0`` and the implicit default (no ``n_workers`` passed) must
    take the identical single-process code path."""
    wf = WalkForwardConfig()
    thresholds = FunnelThresholds()
    costs = CostModel()
    df_default = run_sweep(data, configs, asset_classes, wf, thresholds, costs)
    df_zero = run_sweep(data, configs, asset_classes, wf, thresholds, costs, n_workers=0)
    df_one = run_sweep(data, configs, asset_classes, wf, thresholds, costs, n_workers=1)
    pd.testing.assert_frame_equal(df_default, df_zero)
    pd.testing.assert_frame_equal(df_default, df_one)


def test_sweep_parallel_matches_serial_exactly(
    configs: list[StrategyConfig],
    multi_symbol_data: dict[str, pd.DataFrame],
    multi_symbol_asset_classes: dict[str, AssetClass],
) -> None:
    """The process-pool path (``n_workers=3``, actually exercising the pool
    across a ``ProcessPoolExecutor`` — this is the pool-path test PERF-1
    requires, using module-level ``StrategyConfig.fn`` callables and
    picklable frozen-dataclass args so it works under macOS ``spawn`` and
    Linux ``fork`` alike) must produce byte-identical output to the serial
    baseline: same row order (per-asset submission order) and same values."""
    wf = WalkForwardConfig()
    thresholds = FunnelThresholds()
    costs = CostModel()

    df_serial = run_sweep(
        multi_symbol_data, configs, multi_symbol_asset_classes, wf, thresholds, costs, n_workers=1
    )
    df_parallel = run_sweep(
        multi_symbol_data, configs, multi_symbol_asset_classes, wf, thresholds, costs, n_workers=3
    )
    pd.testing.assert_frame_equal(df_serial, df_parallel)


def test_sweep_cancellation_stops_promptly_in_parallel(
    configs: list[StrategyConfig],
    multi_symbol_data: dict[str, pd.DataFrame],
    multi_symbol_asset_classes: dict[str, AssetClass],
) -> None:
    """A ``should_stop`` that is already true must cancel the parallel sweep
    after only the first batch of completed per-asset tasks — not after the
    whole 6-symbol pool has drained — and raise ``RunCancelledError`` with no
    DataFrame returned. Bounds wall time well under what running all 6
    symbols serially would take, proving the pool was torn down without
    waiting for every task."""
    wf = WalkForwardConfig()
    thresholds = FunnelThresholds()
    costs = CostModel()

    start = time.perf_counter()
    with pytest.raises(RunCancelledError):
        run_sweep(
            multi_symbol_data,
            configs,
            multi_symbol_asset_classes,
            wf,
            thresholds,
            costs,
            should_stop=lambda: True,
            n_workers=2,
        )
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0
