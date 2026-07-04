"""Tests for the four-layer application stack (PRD §10): sizing, combining, routing, attribution."""

import numpy as np
import pandas as pd
import pytest

from funnel.layers.combine import combine_signals, select_uncorrelated
from funnel.layers.router import route_by_regime
from funnel.layers.sizing import atr_size, cap_weight, volatility_target
from funnel.layers.stack import (
    LayerToggles,
    SizingChoice,
    SizingMethod,
    StackSpec,
    attribution_table,
    run_stack,
)
from funnel.regime.base import Regime
from funnel.strategies.base import Category
from funnel.strategies.grid import StrategyConfig
from funnel.strategies.meanrev import zscore_revert
from funnel.strategies.trend import ma_crossover

N_ROWS = 600


def _to_ohlcv(index: pd.DatetimeIndex, close: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    daily_range = np.abs(rng.normal(loc=0.5, scale=0.2, size=len(close))) + 0.05
    open_ = close + rng.normal(loc=0.0, scale=0.1, size=len(close))
    high = np.maximum(open_, close) + daily_range
    low = np.minimum(open_, close) - daily_range
    volume = np.abs(rng.normal(loc=1_000_000.0, scale=200_000.0, size=len(close)))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    ).astype("float64")


# ---------------------------------------------------------------------------
# Sizing (layer 2)
# ---------------------------------------------------------------------------


@pytest.fixture
def calm_then_volatile_ohlcv() -> pd.DataFrame:
    """First half: low-vol drift. Second half: same drift, much higher vol."""
    rng = np.random.default_rng(11)
    index = pd.bdate_range("2020-01-01", periods=N_ROWS)
    half = N_ROWS // 2
    calm = np.cumsum(rng.normal(loc=0.05, scale=0.1, size=half))
    volatile = np.cumsum(rng.normal(loc=0.05, scale=3.0, size=N_ROWS - half))
    close = 100.0 + np.concatenate([calm, calm[-1] + volatile])
    return _to_ohlcv(index, close, rng)


def test_vol_target_scales_down_in_high_vol_stretch(calm_then_volatile_ohlcv: pd.DataFrame) -> None:
    positions = pd.Series(1.0, index=calm_then_volatile_ohlcv.index)
    weighted = volatility_target(
        positions, calm_then_volatile_ohlcv["close"], target_annual_vol=0.15, vol_window=21
    )

    half = N_ROWS // 2
    calm_weight = weighted.iloc[half - 20 : half].mean()
    volatile_weight = weighted.iloc[-20:].mean()
    assert volatile_weight < calm_weight


def test_vol_target_scales_up_to_cap_in_calm_stretch() -> None:
    rng = np.random.default_rng(5)
    index = pd.bdate_range("2020-01-01", periods=N_ROWS)
    close = 100.0 + np.cumsum(rng.normal(loc=0.0, scale=0.01, size=N_ROWS))  # near-zero vol
    df = _to_ohlcv(index, close, rng)
    positions = pd.Series(1.0, index=df.index)

    weighted = volatility_target(
        positions, df["close"], target_annual_vol=0.15, vol_window=21, max_leverage=1.0
    )
    assert weighted.max() <= 1.0 + 1e-9
    assert weighted.iloc[-1] == pytest.approx(1.0, abs=1e-6)


def test_vol_target_never_exceeds_max_leverage(calm_then_volatile_ohlcv: pd.DataFrame) -> None:
    positions = pd.Series(1.0, index=calm_then_volatile_ohlcv.index)
    weighted = volatility_target(
        positions, calm_then_volatile_ohlcv["close"], max_leverage=0.5, vol_window=21
    )
    assert weighted.abs().max() <= 0.5 + 1e-9


def test_vol_target_causal_truncation_invariant(calm_then_volatile_ohlcv: pd.DataFrame) -> None:
    positions = pd.Series(1.0, index=calm_then_volatile_ohlcv.index)
    full = volatility_target(positions, calm_then_volatile_ohlcv["close"], vol_window=21)

    truncated_df = calm_then_volatile_ohlcv.iloc[:300]
    truncated_positions = positions.iloc[:300]
    truncated = volatility_target(truncated_positions, truncated_df["close"], vol_window=21)

    pd.testing.assert_series_equal(full.iloc[:300], truncated, check_names=False)


def test_vol_target_warmup_is_zero() -> None:
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2020-01-01", periods=50)
    close = 100.0 + np.cumsum(rng.normal(size=50))
    df = _to_ohlcv(index, close, rng)
    positions = pd.Series(1.0, index=df.index)

    weighted = volatility_target(positions, df["close"], vol_window=21)
    assert (weighted.iloc[:21] == 0.0).all()


def test_atr_size_hand_check() -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    # Constant true range of 2.0 (high-low=2, no gaps) and constant close=100.
    df = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "volume": [1e6] * 5,
        },
        index=index,
    )
    positions = pd.Series(1.0, index=index)

    weighted = atr_size(positions, df, atr_window=2, risk_fraction=0.02, max_weight=1.0)

    # ATR (rolling mean of true range, window=2) is defined from row 1
    # onward (2 obs): true_range is constant 2.0 for rows >=1 (row 0's
    # true_range has no prev_close so it's just high-low=2.0 too, but the
    # rolling window needs 2 obs so row 0 is NaN -> weight 0).
    assert weighted.iloc[0] == pytest.approx(0.0)
    # risk_per_unit = atr/close = 2.0/100.0 = 0.02; scale = 0.02/0.02 = 1.0,
    # capped at max_weight=1.0 -> weight = 1.0 * position(1.0) = 1.0.
    for i in range(1, 5):
        assert weighted.iloc[i] == pytest.approx(1.0)


def test_atr_size_respects_max_weight() -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    df = pd.DataFrame(
        {
            "open": [100.0] * 5,
            "high": [100.5] * 5,
            "low": [99.5] * 5,
            "close": [100.0] * 5,
            "volume": [1e6] * 5,
        },
        index=index,
    )
    positions = pd.Series(1.0, index=index)
    # Tiny ATR (1.0) relative to close means risk_fraction/risk_per_unit is
    # huge; max_weight must cap it.
    weighted = atr_size(positions, df, atr_window=2, risk_fraction=0.5, max_weight=1.0)
    assert weighted.iloc[-1] == pytest.approx(1.0)


def test_atr_size_causal_truncation_invariant(calm_then_volatile_ohlcv: pd.DataFrame) -> None:
    positions = pd.Series(1.0, index=calm_then_volatile_ohlcv.index)
    full = atr_size(positions, calm_then_volatile_ohlcv, atr_window=14)

    truncated_df = calm_then_volatile_ohlcv.iloc[:300]
    truncated_positions = positions.iloc[:300]
    truncated = atr_size(truncated_positions, truncated_df, atr_window=14)

    pd.testing.assert_series_equal(full.iloc[:300], truncated, check_names=False)


def test_atr_size_warmup_is_zero() -> None:
    rng = np.random.default_rng(2)
    index = pd.bdate_range("2020-01-01", periods=30)
    close = 100.0 + np.cumsum(rng.normal(size=30))
    df = _to_ohlcv(index, close, rng)
    positions = pd.Series(1.0, index=df.index)

    weighted = atr_size(positions, df, atr_window=14)
    # rolling(window=14) needs 14 observations, so only rows 0..12 (13 rows)
    # are undefined warmup; row 13 (the 14th row) is the first defined ATR.
    assert (weighted.iloc[:13] == 0.0).all()
    assert weighted.iloc[13] != 0.0


def test_cap_weight_clips_both_sides() -> None:
    positions = pd.Series([-2.0, -0.3, 0.0, 0.5, 3.0])
    capped = cap_weight(positions, max_weight=1.0)
    assert capped.tolist() == [-1.0, -0.3, 0.0, 0.5, 1.0]


# ---------------------------------------------------------------------------
# Combine (layer 3)
# ---------------------------------------------------------------------------


def test_combine_opposite_signals_average_to_zero() -> None:
    index = pd.bdate_range("2020-01-01", periods=10)
    a = pd.Series(1.0, index=index)
    b = pd.Series(-1.0, index=index)

    combined = combine_signals({"a": a, "b": b})
    assert (combined == 0.0).all()


def test_combine_alignment_missing_treated_as_zero() -> None:
    index_a = pd.bdate_range("2020-01-01", periods=5)
    index_b = pd.bdate_range("2020-01-04", periods=5)
    a = pd.Series(1.0, index=index_a)
    b = pd.Series(1.0, index=index_b)

    combined = combine_signals({"a": a, "b": b})
    union_index = index_a.union(index_b)
    assert len(combined) == len(union_index)
    # Days only "a" covers: b contributes 0 -> combined = 0.5 * 1.0 = 0.5
    only_a_day = index_a.difference(index_b)[0]
    assert combined.loc[only_a_day] == pytest.approx(0.5)
    # Overlapping days: both contribute 1.0 -> combined = 1.0
    overlap_day = index_a.intersection(index_b)[0]
    assert combined.loc[overlap_day] == pytest.approx(1.0)


def test_combine_honors_explicit_weights() -> None:
    index = pd.bdate_range("2020-01-01", periods=5)
    a = pd.Series(1.0, index=index)
    b = pd.Series(1.0, index=index)

    combined = combine_signals({"a": a, "b": b}, weights={"a": 0.75, "b": 0.25})
    assert combined.tolist() == pytest.approx([1.0] * len(combined))


def test_combine_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        combine_signals({})


def test_select_uncorrelated_picks_best_duplicate_plus_independent() -> None:
    index = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(77)
    base = rng.normal(loc=0.001, scale=0.01, size=len(index))
    # A and B are near-duplicates (B = A plus tiny noise); C is independent.
    a = pd.Series(base, index=index)
    b = pd.Series(base + rng.normal(scale=1e-5, size=len(index)), index=index)
    c = pd.Series(rng.normal(loc=0.0005, scale=0.01, size=len(index)), index=index)

    selected = select_uncorrelated({"a": a, "b": b, "c": c}, max_pairwise_corr=0.7)

    # Exactly one of a/b (the higher-Sharpe one) plus c.
    assert "c" in selected
    assert len(selected) == 2
    assert ("a" in selected) != ("b" in selected)


def test_select_uncorrelated_is_deterministic() -> None:
    index = pd.bdate_range("2020-01-01", periods=300)
    rng = np.random.default_rng(88)
    base = rng.normal(loc=0.001, scale=0.01, size=len(index))
    a = pd.Series(base, index=index)
    b = pd.Series(base + rng.normal(scale=1e-5, size=len(index)), index=index)
    c = pd.Series(rng.normal(loc=0.0005, scale=0.01, size=len(index)), index=index)
    returns = {"a": a, "b": b, "c": c}

    first = select_uncorrelated(returns, max_pairwise_corr=0.7)
    second = select_uncorrelated(returns, max_pairwise_corr=0.7)
    assert first == second


# ---------------------------------------------------------------------------
# Router (layer 4)
# ---------------------------------------------------------------------------


def test_route_by_regime_zeros_non_matching_regime() -> None:
    index = pd.bdate_range("2020-01-01", periods=6)
    positions = pd.Series([1.0, 1.0, -1.0, -1.0, 1.0, 1.0], index=index)
    regimes = pd.Series(
        [
            Regime.TRENDING,
            Regime.TRENDING,
            Regime.CHOPPY,
            Regime.CHOPPY,
            Regime.TRENDING,
            Regime.CHOPPY,
        ],
        index=index,
    )

    routed = route_by_regime(positions, regimes, Regime.TRENDING)
    expected = [1.0, 1.0, 0.0, 0.0, 1.0, 0.0]
    assert routed.tolist() == expected


def test_route_by_regime_ffills_sparse_labels() -> None:
    index = pd.bdate_range("2020-01-01", periods=6)
    positions = pd.Series(1.0, index=index)
    sparse_index = index[[0, 3]]
    regimes = pd.Series([Regime.TRENDING, Regime.CHOPPY], index=sparse_index)

    routed = route_by_regime(positions, regimes, Regime.TRENDING)
    # Rows 0-2 ffill to TRENDING (stay on), rows 3-5 ffill to CHOPPY (zeroed).
    assert routed.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_route_by_regime_zeros_rows_before_first_label() -> None:
    index = pd.bdate_range("2020-01-01", periods=4)
    positions = pd.Series(1.0, index=index)
    regimes = pd.Series([Regime.TRENDING], index=index[[2]])

    routed = route_by_regime(positions, regimes, Regime.TRENDING)
    assert routed.tolist() == [0.0, 0.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# Stack / attribution
# ---------------------------------------------------------------------------


@pytest.fixture
def regime_switching_ohlcv() -> pd.DataFrame:
    """First half: strong clean trend. Second half: bounded, mean-reverting chop."""
    rng = np.random.default_rng(321)
    index = pd.bdate_range("2019-01-01", periods=N_ROWS)
    half = N_ROWS // 2

    trend_part = (
        100.0 + np.linspace(0.0, 60.0, half) + np.cumsum(rng.normal(loc=0.0, scale=0.1, size=half))
    )
    chop_start = trend_part[-1]
    t = np.arange(N_ROWS - half)
    chop_part = (
        chop_start
        + 8.0 * np.sin(2 * np.pi * t / 20.0)
        + rng.normal(loc=0.0, scale=0.3, size=N_ROWS - half)
    )
    close = np.concatenate([trend_part, chop_part])
    return _to_ohlcv(index, close, rng)


@pytest.fixture
def perfect_regime_labels(regime_switching_ohlcv: pd.DataFrame) -> pd.Series:
    """Hand-built regime labels matching the fixture's true trend/chop halves exactly."""
    half = N_ROWS // 2
    labels = [Regime.TRENDING] * half + [Regime.CHOPPY] * (N_ROWS - half)
    return pd.Series(labels, index=regime_switching_ohlcv.index)


@pytest.fixture
def trend_and_meanrev_configs() -> list[StrategyConfig]:
    trend_config = StrategyConfig(
        name="ma_crossover_test",
        family="ma_crossover",
        fn=ma_crossover,
        params={"fast": 10, "slow": 50},
        category=Category.TREND,
    )
    meanrev_config = StrategyConfig(
        name="zscore_revert_test",
        family="zscore_revert",
        fn=zscore_revert,
        params={"window": 20, "threshold": 1.5},
        category=Category.MEAN_REVERSION,
    )
    return [trend_config, meanrev_config]


def test_routing_improves_sharpe_vs_base(
    regime_switching_ohlcv: pd.DataFrame,
    perfect_regime_labels: pd.Series,
    trend_and_meanrev_configs: list[StrategyConfig],
) -> None:
    trend_config = trend_and_meanrev_configs[0]

    base_spec = StackSpec(
        df=regime_switching_ohlcv,
        configs=[trend_config],
        cost_bps=1.0,
        regimes=perfect_regime_labels,
    )
    base_result = run_stack(base_spec, LayerToggles())
    routed_result = run_stack(base_spec, LayerToggles(regime_routing=True))

    # The trend strategy trades through the whole series by default (base);
    # routing switches it off during the choppy half, where it is expected
    # to whipsaw and drag down Sharpe. Routing should improve Sharpe.
    assert routed_result.sharpe > base_result.sharpe


def test_run_stack_uses_first_config_when_combining_off(
    regime_switching_ohlcv: pd.DataFrame,
    trend_and_meanrev_configs: list[StrategyConfig],
) -> None:
    spec = StackSpec(df=regime_switching_ohlcv, configs=trend_and_meanrev_configs, cost_bps=1.0)
    result = run_stack(spec, LayerToggles(combining=False))

    trend_config = trend_and_meanrev_configs[0]
    solo_spec = StackSpec(df=regime_switching_ohlcv, configs=[trend_config], cost_bps=1.0)
    solo_result = run_stack(solo_spec, LayerToggles())

    assert result.sharpe == pytest.approx(solo_result.sharpe)


def test_run_stack_sizing_toggle_changes_result(
    regime_switching_ohlcv: pd.DataFrame,
    trend_and_meanrev_configs: list[StrategyConfig],
) -> None:
    trend_config = trend_and_meanrev_configs[0]
    spec = StackSpec(
        df=regime_switching_ohlcv,
        configs=[trend_config],
        cost_bps=1.0,
        sizing_choice=SizingChoice(method=SizingMethod.VOL_TARGET, vol_window=21),
    )
    unsized = run_stack(spec, LayerToggles(sizing=False))
    sized = run_stack(spec, LayerToggles(sizing=True))

    assert not unsized.returns.equals(sized.returns)


def test_attribution_table_steps_match_independent_run_stack(
    regime_switching_ohlcv: pd.DataFrame,
    perfect_regime_labels: pd.Series,
    trend_and_meanrev_configs: list[StrategyConfig],
) -> None:
    spec = StackSpec(
        df=regime_switching_ohlcv,
        configs=trend_and_meanrev_configs,
        cost_bps=1.0,
        regimes=perfect_regime_labels,
    )
    table = attribution_table(spec)

    assert table["step"].tolist() == ["base", "+sizing", "+routing", "+combining"]

    expected_toggles = {
        "base": LayerToggles(),
        "+sizing": LayerToggles(sizing=True),
        "+routing": LayerToggles(sizing=True, regime_routing=True),
        "+combining": LayerToggles(sizing=True, regime_routing=True, combining=True),
    }
    for _, row in table.iterrows():
        independent = run_stack(spec, expected_toggles[row["step"]])
        assert row["sharpe"] == pytest.approx(independent.sharpe)
        assert row["max_drawdown"] == pytest.approx(independent.max_drawdown)
        assert row["win_rate"] == pytest.approx(independent.win_rate)

    # Deltas are consistent with consecutive rows.
    for i in range(1, len(table)):
        prev_sharpe = table.iloc[i - 1]["sharpe"]
        assert table.iloc[i]["delta_sharpe"] == pytest.approx(table.iloc[i]["sharpe"] - prev_sharpe)


def test_attribution_table_skips_routing_without_regimes(
    regime_switching_ohlcv: pd.DataFrame,
    trend_and_meanrev_configs: list[StrategyConfig],
) -> None:
    spec = StackSpec(df=regime_switching_ohlcv, configs=trend_and_meanrev_configs, cost_bps=1.0)
    table = attribution_table(spec)
    assert table["step"].tolist() == ["base", "+sizing", "+combining"]


def test_attribution_table_skips_combining_with_single_config(
    regime_switching_ohlcv: pd.DataFrame,
    perfect_regime_labels: pd.Series,
    trend_and_meanrev_configs: list[StrategyConfig],
) -> None:
    trend_config = trend_and_meanrev_configs[0]
    spec = StackSpec(
        df=regime_switching_ohlcv,
        configs=[trend_config],
        cost_bps=1.0,
        regimes=perfect_regime_labels,
    )
    table = attribution_table(spec)
    assert table["step"].tolist() == ["base", "+sizing", "+routing"]
