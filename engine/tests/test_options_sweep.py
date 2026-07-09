"""Tests for funnel.options.grid and funnel.options.sweep (offline, synthetic).

Covers: the overlay config grid (unique names, per-structure counts, all
specs defined-risk), ``score_overlay``'s window math against v1's exact
formula (and that overlay/underlying are scored on identical windows), a
small end-to-end overlay sweep (row counts, mandated columns, honesty
columns), a covered-call-on-a-rising-fixture sanity check (upside forgone,
vs.-hold comparison), skip handling for too-short history, and bootstrap/run
determinism (PLAN.md "v2 — Options Overlay Module", V2-M3).
"""

import time

import numpy as np
import pandas as pd
import pytest

from funnel.backtest.walkforward import _is_oos_split, _window_bounds
from funnel.cancellation import RunCancelledError
from funnel.config import FunnelThresholds, WalkForwardConfig
from funnel.options.grid import OverlayConfig, build_overlay_grid, summarize_overlay_grid
from funnel.options.overlays import OverlayCosts, OverlaySpec, OverlayStructure, StrikeSelector
from funnel.options.pricing import OptionKind, VolProxyConfig
from funnel.options.sweep import (
    OVERLAY_SWEEP_COLUMNS,
    run_overlay_sweep,
    score_overlay,
    write_overlay_results,
)


def _make_df(n: int, seed: int, drift: float = 0.02, scale: float = 1.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=n)
    close = 100.0 + np.cumsum(rng.normal(loc=drift, scale=scale, size=n))
    return pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1e6},
        index=index,
    ).astype("float64")


def _make_rising_df(n: int, seed: int, daily_mult: float = 1.0015) -> pd.DataFrame:
    """A strongly, steadily rising series with mild noise — enough that a
    near-the-money covered call's short leg is very likely to finish ITM
    repeatedly, so upside is provably capped somewhere in the backtest."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=n)
    close = 100.0 * (daily_mult ** np.arange(n)) + np.cumsum(rng.normal(0.0, 0.05, size=n))
    return pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1e6},
        index=index,
    ).astype("float64")


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def test_all_grid_configs_are_defined_risk() -> None:
    """Every spec built by build_overlay_grid() must construct without
    raising UndefinedRiskError — construction itself is the check."""
    configs = build_overlay_grid()
    assert len(configs) > 0
    for config in configs:
        assert isinstance(config.spec, OverlaySpec)


def test_grid_names_are_unique() -> None:
    configs = build_overlay_grid()
    names = [c.name for c in configs]
    assert len(names) == len(set(names))


def test_grid_total_in_expected_range() -> None:
    configs = build_overlay_grid()
    assert 30 <= len(configs) <= 60
    assert len(configs) == 46


def test_grid_per_structure_counts_match_summary() -> None:
    configs = build_overlay_grid()
    summary = summarize_overlay_grid(configs)

    assert summary == {
        "cash_secured_put": 16,
        "covered_call": 16,
        "leaps": 4,
        "vertical_spread": 10,
    }

    from collections import Counter

    counts = Counter(c.spec.structure.value for c in configs)
    assert dict(counts) == summary


def test_grid_covers_all_four_structures() -> None:
    configs = build_overlay_grid()
    structures = {c.spec.structure for c in configs}
    assert structures == {
        OverlayStructure.COVERED_CALL,
        OverlayStructure.CASH_SECURED_PUT,
        OverlayStructure.VERTICAL_SPREAD,
        OverlayStructure.LEAPS,
    }


def test_grid_includes_hold_to_expiry_configs_where_assignment_is_meaningful() -> None:
    """Every other config's default roll_at_dte=5 makes simulate_overlay's
    settlement/assignment path structurally unreachable (the scheduled-roll
    check always fires first). covered_call, cash_secured_put, and
    vertical_spread must each ship at least one roll_at_dte=0
    (hold-to-expiry) variant so a real report can observe n_assignments > 0;
    LEAPS has no short leg, so assignment does not apply there."""
    configs = build_overlay_grid()
    hold_to_expiry = [c for c in configs if c.spec.roll_at_dte == 0]

    hold_structures = {c.spec.structure for c in hold_to_expiry}
    assert hold_structures == {
        OverlayStructure.COVERED_CALL,
        OverlayStructure.CASH_SECURED_PUT,
        OverlayStructure.VERTICAL_SPREAD,
    }
    assert OverlayStructure.LEAPS not in hold_structures

    from collections import Counter

    counts = Counter(c.spec.structure.value for c in hold_to_expiry)
    assert counts["covered_call"] == 4
    assert counts["cash_secured_put"] == 4
    assert counts["vertical_spread"] == 2

    # The avoid=True + hold-to-expiry combination (P(ITM)-triggered rolls
    # only, otherwise runs to settlement) must be present for both
    # covered_call and cash_secured_put.
    avoid_hold = [c for c in hold_to_expiry if c.spec.avoid_assignment]
    avoid_hold_structures = {c.spec.structure for c in avoid_hold}
    assert OverlayStructure.COVERED_CALL in avoid_hold_structures
    assert OverlayStructure.CASH_SECURED_PUT in avoid_hold_structures


# ---------------------------------------------------------------------------
# score_overlay window math
# ---------------------------------------------------------------------------


def test_score_overlay_stitched_oos_length_matches_v1_formula() -> None:
    n = 700
    wf = WalkForwardConfig(n_windows=5, is_fraction=0.7)
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.0005, 0.01, size=n))

    score = score_overlay(returns, wf)

    bounds = _window_bounds(n, wf.n_windows)
    expected_len = 0
    for start, end in bounds:
        _, split, _ = _is_oos_split(start, end, wf.is_fraction)
        expected_len += end - split

    assert len(score.oos_returns) == expected_len


def test_score_overlay_scores_overlay_and_underlying_on_identical_windows() -> None:
    """OverlayResult.underlying_returns is NaN exactly where returns is NaN
    (same valid mask, per simulate_overlay's contract), so scoring both
    series through score_overlay must stitch OOS segments of identical
    length — the comparison windows are the same."""
    n = 700
    wf = WalkForwardConfig(n_windows=5, is_fraction=0.7)
    rng = np.random.default_rng(2)

    warmup = 21
    overlay_returns = pd.Series(np.nan, index=range(n), dtype="float64")
    overlay_returns.iloc[warmup:] = rng.normal(0.0003, 0.01, size=n - warmup)
    underlying_returns = pd.Series(np.nan, index=range(n), dtype="float64")
    underlying_returns.iloc[warmup:] = rng.normal(0.0005, 0.015, size=n - warmup)

    overlay_score = score_overlay(overlay_returns, wf)
    underlying_score = score_overlay(underlying_returns, wf)

    assert len(overlay_score.oos_returns) == len(underlying_score.oos_returns)


def test_score_overlay_raises_on_insufficient_valid_oos_rows() -> None:
    wf = WalkForwardConfig(n_windows=5, is_fraction=0.7)
    # 100 rows split into 5 windows of 20; OOS tail per window is 6 rows,
    # far below MIN_OOS_ROWS (30).
    returns = pd.Series(np.linspace(0.0, 0.01, 100))
    with pytest.raises(Exception):  # noqa: B017 -- InsufficientHistoryError
        score_overlay(returns, wf)


# ---------------------------------------------------------------------------
# Sweep: fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_configs() -> list[OverlayConfig]:
    return [
        OverlayConfig(
            name="covered_call_test",
            spec=OverlaySpec(
                structure=OverlayStructure.COVERED_CALL,
                dte_target=21,
                strike_selector=StrikeSelector(mode="delta", value=0.25),
            ),
            description="test covered call",
        ),
        OverlayConfig(
            name="cash_secured_put_test",
            spec=OverlaySpec(
                structure=OverlayStructure.CASH_SECURED_PUT,
                dte_target=21,
                strike_selector=StrikeSelector(mode="delta", value=-0.25),
            ),
            description="test cash-secured put",
        ),
        OverlayConfig(
            name="vertical_test",
            spec=OverlaySpec(
                structure=OverlayStructure.VERTICAL_SPREAD,
                dte_target=30,
                strike_selector=StrikeSelector(mode="delta", value=-0.20),
                spread_width_pct=0.05,
                kind=OptionKind.PUT,
            ),
            description="test vertical spread",
        ),
    ]


@pytest.fixture
def data() -> dict[str, pd.DataFrame]:
    return {
        "AAA": _make_df(700, seed=1),
        "BBB": _make_df(700, seed=2, drift=-0.01),
    }


@pytest.fixture
def wf() -> WalkForwardConfig:
    return WalkForwardConfig()


@pytest.fixture
def vol_config() -> VolProxyConfig:
    return VolProxyConfig()


@pytest.fixture
def costs() -> OverlayCosts:
    return OverlayCosts()


@pytest.fixture
def thresholds() -> FunnelThresholds:
    return FunnelThresholds()


# ---------------------------------------------------------------------------
# Sweep: row counts, columns, honesty
# ---------------------------------------------------------------------------


def test_sweep_row_count_is_configs_times_symbols(
    small_configs: list[OverlayConfig],
    data: dict[str, pd.DataFrame],
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    thresholds: FunnelThresholds,
) -> None:
    df = run_overlay_sweep(
        data, small_configs, None, wf, vol_config, costs, 0.03, thresholds, n_bootstrap=25, seed=42
    )
    assert len(df) == len(small_configs) * len(data)


def test_sweep_has_mandated_columns(
    small_configs: list[OverlayConfig],
    data: dict[str, pd.DataFrame],
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    thresholds: FunnelThresholds,
) -> None:
    df = run_overlay_sweep(
        data, small_configs, None, wf, vol_config, costs, 0.03, thresholds, n_bootstrap=25, seed=42
    )
    assert list(df.columns) == list(OVERLAY_SWEEP_COLUMNS)
    assert "model_priced" in df.columns
    assert "mean_model_prob_itm" in df.columns
    assert "assignment_probability" not in df.columns
    assert df["model_priced"].all()


def test_sweep_symbols_param_restricts_universe(
    small_configs: list[OverlayConfig],
    data: dict[str, pd.DataFrame],
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    thresholds: FunnelThresholds,
) -> None:
    df = run_overlay_sweep(
        data,
        small_configs,
        ["AAA"],
        wf,
        vol_config,
        costs,
        0.03,
        thresholds,
        n_bootstrap=25,
        seed=42,
    )
    assert set(df["symbol"]) == {"AAA"}
    assert len(df) == len(small_configs)


def test_covered_call_on_rising_fixture_shows_capped_upside(
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    thresholds: FunnelThresholds,
) -> None:
    """A near-the-money covered call held through a strong, steady rally
    must show a strictly positive upside_forgone and (in this scenario) a
    negative oos_sharpe_vs_hold, and both columns must be populated/finite
    for every row — the honesty-by-design contract never hides this."""
    config = OverlayConfig(
        name="covered_call_35d_45dte",
        spec=OverlaySpec(
            structure=OverlayStructure.COVERED_CALL,
            dte_target=45,
            strike_selector=StrikeSelector(mode="delta", value=0.35),
        ),
        description="near-the-money covered call",
    )
    rising = {"RISING": _make_rising_df(700, seed=5, daily_mult=1.0015)}

    df = run_overlay_sweep(
        rising, [config], None, wf, vol_config, costs, 0.03, thresholds, n_bootstrap=25, seed=42
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert not row["skipped"]
    assert np.isfinite(row["upside_forgone"])
    assert np.isfinite(row["oos_sharpe_vs_hold"])
    assert row["upside_forgone"] > 0.0
    assert row["oos_sharpe_vs_hold"] < 0.0


def test_hold_to_expiry_covered_call_reports_assignments(
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    thresholds: FunnelThresholds,
) -> None:
    """Regression test for the post-acceptance finding: every non-hold config
    in the grid rolls at roll_at_dte=5, before simulate_overlay's scheduled-
    roll check can ever reach true expiry, so _check_assignment is
    structurally unreachable and a production run would always report
    n_assignments=0 regardless of how many years of data it covers. A
    hold-to-expiry (roll_at_dte=0) covered call, run on a strongly and
    steadily rising fixture long enough for several 21-DTE expiries, must
    report n_assignments > 0 — proving the sweep's production path can
    actually surface assignment events now that the grid includes
    roll_at_dte=0 variants."""
    hold_to_expiry_configs = [
        c for c in build_overlay_grid() if c.name == "covered_call_d25_dte21_roll0_noavoid_hold"
    ]
    assert len(hold_to_expiry_configs) == 1
    config = hold_to_expiry_configs[0]
    assert config.spec.roll_at_dte == 0

    rising = {"RISING": _make_rising_df(700, seed=5, daily_mult=1.0015)}

    df = run_overlay_sweep(
        rising, [config], None, wf, vol_config, costs, 0.03, thresholds, n_bootstrap=25, seed=42
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert not row["skipped"]
    assert row["n_assignments"] > 0


def test_bootstrap_columns_deterministic_under_fixed_seed(
    small_configs: list[OverlayConfig],
    data: dict[str, pd.DataFrame],
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    thresholds: FunnelThresholds,
) -> None:
    df1 = run_overlay_sweep(
        data, small_configs, None, wf, vol_config, costs, 0.03, thresholds, n_bootstrap=25, seed=42
    )
    df2 = run_overlay_sweep(
        data, small_configs, None, wf, vol_config, costs, 0.03, thresholds, n_bootstrap=25, seed=42
    )
    bootstrap_cols = [
        "bootstrap_sharpe_p5",
        "bootstrap_sharpe_p50",
        "bootstrap_sharpe_p95",
        "bootstrap_worst_case_drawdown",
        "bootstrap_verdict",
    ]
    pd.testing.assert_frame_equal(df1[bootstrap_cols], df2[bootstrap_cols])


def test_full_run_is_deterministic(
    small_configs: list[OverlayConfig],
    data: dict[str, pd.DataFrame],
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    thresholds: FunnelThresholds,
) -> None:
    df1 = run_overlay_sweep(
        data, small_configs, None, wf, vol_config, costs, 0.03, thresholds, n_bootstrap=25, seed=42
    )
    df2 = run_overlay_sweep(
        data, small_configs, None, wf, vol_config, costs, 0.03, thresholds, n_bootstrap=25, seed=42
    )
    pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# Skip handling
# ---------------------------------------------------------------------------


def test_sweep_skip_handling_for_short_history(
    small_configs: list[OverlayConfig],
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    thresholds: FunnelThresholds,
) -> None:
    data = {
        "AAA": _make_df(700, seed=1),
        "SHORT": _make_df(100, seed=9),
    }
    df = run_overlay_sweep(
        data, small_configs, None, wf, vol_config, costs, 0.03, thresholds, n_bootstrap=25, seed=42
    )

    assert len(df) == len(small_configs) * len(data)
    short_rows = df[df["symbol"] == "SHORT"]
    assert len(short_rows) == len(small_configs)
    assert short_rows["skipped"].all()
    assert short_rows["overlay_oos_sharpe"].isna().all()
    assert short_rows["model_priced"].all()

    aaa_rows = df[df["symbol"] == "AAA"]
    assert not aaa_rows["skipped"].any()


def test_write_overlay_results_csv(
    tmp_path,
    small_configs: list[OverlayConfig],
    data: dict[str, pd.DataFrame],
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    thresholds: FunnelThresholds,
) -> None:
    df = run_overlay_sweep(
        data, small_configs, None, wf, vol_config, costs, 0.03, thresholds, n_bootstrap=25, seed=42
    )
    path = tmp_path / "overlay_results.csv"
    write_overlay_results(df, path)

    assert path.exists()
    reloaded = pd.read_csv(path)
    assert len(reloaded) == len(df)
    assert list(reloaded.columns) == list(OVERLAY_SWEEP_COLUMNS)


# ---------------------------------------------------------------------------
# PERF-1: overlay sweep parallelism (n_workers) — equivalence and cancellation
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_symbol_data() -> dict[str, pd.DataFrame]:
    """Six symbols so ``n_workers=2`` actually chunks work across multiple
    per-symbol process-pool tasks."""
    return {f"SYM{i}": _make_df(700, seed=i) for i in range(6)}


def test_overlay_sweep_n_workers_zero_matches_default_serial(
    small_configs: list[OverlayConfig],
    data: dict[str, pd.DataFrame],
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    thresholds: FunnelThresholds,
) -> None:
    df_default = run_overlay_sweep(
        data, small_configs, None, wf, vol_config, costs, 0.03, thresholds, n_bootstrap=25, seed=42
    )
    df_zero = run_overlay_sweep(
        data,
        small_configs,
        None,
        wf,
        vol_config,
        costs,
        0.03,
        thresholds,
        n_bootstrap=25,
        seed=42,
        n_workers=0,
    )
    pd.testing.assert_frame_equal(df_default, df_zero)


def test_overlay_sweep_parallel_matches_serial_exactly(
    small_configs: list[OverlayConfig],
    multi_symbol_data: dict[str, pd.DataFrame],
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    thresholds: FunnelThresholds,
) -> None:
    """Pool-path test (PERF-1): ``n_workers=3`` actually dispatches across a
    ``ProcessPoolExecutor`` (module-level task function, picklable frozen-
    dataclass args — works under macOS ``spawn`` and Linux ``fork`` alike)
    and must match the serial baseline exactly, including bootstrap columns
    (same seed passed to every worker, same as the serial path)."""
    df_serial = run_overlay_sweep(
        multi_symbol_data,
        small_configs,
        None,
        wf,
        vol_config,
        costs,
        0.03,
        thresholds,
        n_bootstrap=25,
        seed=42,
        n_workers=1,
    )
    df_parallel = run_overlay_sweep(
        multi_symbol_data,
        small_configs,
        None,
        wf,
        vol_config,
        costs,
        0.03,
        thresholds,
        n_bootstrap=25,
        seed=42,
        n_workers=3,
    )
    pd.testing.assert_frame_equal(df_serial, df_parallel)


def test_overlay_sweep_cancellation_stops_promptly_in_parallel(
    small_configs: list[OverlayConfig],
    multi_symbol_data: dict[str, pd.DataFrame],
    wf: WalkForwardConfig,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    thresholds: FunnelThresholds,
) -> None:
    """An already-true ``should_stop`` must cancel the parallel overlay sweep
    after only the first batch of completed per-symbol tasks, raising
    ``RunCancelledError`` well before all 6 symbols would finish serially."""
    start = time.perf_counter()
    with pytest.raises(RunCancelledError):
        run_overlay_sweep(
            multi_symbol_data,
            small_configs,
            None,
            wf,
            vol_config,
            costs,
            0.03,
            thresholds,
            n_bootstrap=25,
            seed=42,
            should_stop=lambda: True,
            n_workers=2,
        )
    elapsed = time.perf_counter() - start
    assert elapsed < 30.0
