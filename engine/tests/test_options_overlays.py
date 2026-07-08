"""Tests for funnel.options.overlays: defined-risk enforcement, hand-computed
roll/assignment fixtures, upside-forgone honesty, bounded-loss structural
guarantees, avoid-assignment rolling, causality, and cost application
(PLAN.md "v2 — Options Overlay Module", V2-M2).
"""

import numpy as np
import pandas as pd
import pytest

from funnel.options.overlays import (
    AssignmentEvent,
    OverlayCosts,
    OverlaySpec,
    OverlayStructure,
    StrikeSelector,
    UndefinedRiskError,
    is_defined_risk,
    simulate_overlay,
)
from funnel.options.pricing import OptionKind, VolProxyConfig, bs_price

CALL = OptionKind.CALL
PUT = OptionKind.PUT

TRUNCATE_AT = 400


def _frame_from_closes(closes: list[float]) -> pd.DataFrame:
    """A minimal OHLCV frame (open=high=low=close) around a given close path."""
    index = pd.bdate_range("2020-01-01", periods=len(closes))
    close = pd.Series(closes, index=index, dtype="float64")
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": pd.Series(1_000_000.0, index=index),
        }
    )


def _patch_constant_vol(monkeypatch: pytest.MonkeyPatch, value: float) -> None:
    """Replace ``synthetic_iv`` with a constant, warmup-free series so a
    position enters on bar 0 and every mark uses a known, hand-checkable vol.
    """
    import funnel.options.overlays as overlays_module

    def _constant(close: pd.Series, config: VolProxyConfig) -> pd.Series:
        return pd.Series(value, index=close.index)

    monkeypatch.setattr(overlays_module, "synthetic_iv", _constant)


# ---------------------------------------------------------------------------
# Defined-risk enforcement
# ---------------------------------------------------------------------------

VALID_SPECS = [
    OverlaySpec(
        structure=OverlayStructure.COVERED_CALL,
        dte_target=30,
        strike_selector=StrikeSelector(mode="otm_pct", value=0.05),
    ),
    OverlaySpec(
        structure=OverlayStructure.CASH_SECURED_PUT,
        dte_target=30,
        strike_selector=StrikeSelector(mode="otm_pct", value=0.05),
    ),
    OverlaySpec(
        structure=OverlayStructure.VERTICAL_SPREAD,
        dte_target=30,
        strike_selector=StrikeSelector(mode="otm_pct", value=0.05),
        spread_width_pct=0.05,
        kind=PUT,
    ),
    OverlaySpec(
        structure=OverlayStructure.LEAPS,
        dte_target=252,
        strike_selector=StrikeSelector(mode="delta", value=0.75),
    ),
]


@pytest.mark.parametrize("spec", VALID_SPECS, ids=[s.structure.value for s in VALID_SPECS])
def test_valid_specs_construct(spec: OverlaySpec) -> None:
    ok, reason = is_defined_risk(spec)
    assert ok
    assert reason


def test_zero_width_spread_raises_undefined_risk() -> None:
    with pytest.raises(UndefinedRiskError):
        OverlaySpec(
            structure=OverlayStructure.VERTICAL_SPREAD,
            dte_target=30,
            strike_selector=StrikeSelector(mode="otm_pct", value=0.05),
            spread_width_pct=0.0,
            kind=PUT,
        )


def test_negative_width_spread_raises_undefined_risk() -> None:
    with pytest.raises(UndefinedRiskError):
        OverlaySpec(
            structure=OverlayStructure.VERTICAL_SPREAD,
            dte_target=30,
            strike_selector=StrikeSelector(mode="otm_pct", value=0.05),
            spread_width_pct=-0.05,
            kind=PUT,
        )


def test_negative_dte_raises_undefined_risk() -> None:
    with pytest.raises(UndefinedRiskError):
        OverlaySpec(
            structure=OverlayStructure.COVERED_CALL,
            dte_target=-10,
            strike_selector=StrikeSelector(mode="otm_pct", value=0.05),
        )


@pytest.mark.parametrize("value", [1.0, -1.0, 1.5, -1.5])
def test_delta_at_least_one_raises_undefined_risk(value: float) -> None:
    with pytest.raises(UndefinedRiskError):
        OverlaySpec(
            structure=OverlayStructure.COVERED_CALL,
            dte_target=30,
            strike_selector=StrikeSelector(mode="delta", value=value),
        )


def test_simulate_overlay_rejects_bypassed_undefined_risk_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``simulate_overlay`` re-checks ``is_defined_risk`` itself (belt-and-
    suspenders per the module docstring), independent of ``__post_init__``.
    """
    import funnel.options.overlays as overlays_module

    spec = VALID_SPECS[0]
    monkeypatch.setattr(overlays_module, "is_defined_risk", lambda s: (False, "forced rejection"))
    with pytest.raises(UndefinedRiskError, match="forced rejection"):
        simulate_overlay(
            _frame_from_closes([100.0] * 30),
            spec,
            VolProxyConfig(),
            OverlayCosts(),
        )


# ---------------------------------------------------------------------------
# Hand-computed micro-fixture: covered call premium collection, one roll
# (natural expiry), one expiry-ITM assignment event.
# ---------------------------------------------------------------------------


def test_covered_call_hand_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_constant_vol(monkeypatch, 0.20)
    closes = [100.0] * 10 + [110.0, 110.0]
    df = _frame_from_closes(closes)

    spec = OverlaySpec(
        structure=OverlayStructure.COVERED_CALL,
        dte_target=10,
        strike_selector=StrikeSelector(mode="otm_pct", value=0.05),
        roll_at_dte=0,
    )
    costs = OverlayCosts(per_contract_commission=1.0, spread_haircut_pct=0.10)

    result = simulate_overlay(df, spec, VolProxyConfig(), costs, rate=0.0)

    vol, rate, strike = 0.20, 0.0, 105.0
    premium0 = bs_price(100.0, strike, 10 / 252, vol, rate, CALL)
    premium1 = bs_price(100.0, strike, 9 / 252, vol, rate, CALL)
    premium9 = bs_price(100.0, strike, 1 / 252, vol, rate, CALL)
    capital_base = 100.0 * 100.0  # spot_entry * 100 shares * 1 contract

    open_cost = 1.0 + 0.10 * premium0 * 100.0
    close_cost = 1.0 + 0.10 * 5.0 * 100.0  # intrinsic at settlement = 110-105

    day1_expected = (100.0 * (premium0 - premium1) - open_cost) / capital_base
    day10_expected = (100.0 * premium9 - 500.0 + 1000.0 - close_cost) / capital_base
    cycle_sum_expected = (90.0 * premium0 + 448.0) / capital_base

    assert pd.isna(result.returns.iloc[0])
    assert result.returns.iloc[1] == pytest.approx(day1_expected)
    assert result.returns.iloc[10] == pytest.approx(day10_expected)
    assert result.returns.iloc[1:11].sum() == pytest.approx(cycle_sum_expected)

    assert result.n_rolls == 1
    assert len(result.events) == 1
    event = result.events[0]
    assert isinstance(event, AssignmentEvent)
    assert event.structure is OverlayStructure.COVERED_CALL
    assert event.strike == pytest.approx(105.0)
    assert event.spot == pytest.approx(110.0)
    assert event.moneyness == pytest.approx(110.0 / 105.0)
    assert event.date == df.index[10]


# ---------------------------------------------------------------------------
# Covered call caps upside vs. buy-and-hold
# ---------------------------------------------------------------------------


def _covered_call_spec() -> OverlaySpec:
    return OverlaySpec(
        structure=OverlayStructure.COVERED_CALL,
        dte_target=21,
        strike_selector=StrikeSelector(mode="otm_pct", value=0.05),
        roll_at_dte=5,
    )


def test_covered_call_caps_upside_on_rising_fixture(trending_ohlcv: pd.DataFrame) -> None:
    result = simulate_overlay(
        trending_ohlcv, _covered_call_spec(), VolProxyConfig(), OverlayCosts(), rate=0.03
    )
    valid = result.returns.notna()
    overlay_total = float((1.0 + result.returns[valid]).prod() - 1.0)
    bh_total = float((1.0 + result.underlying_returns[valid]).prod() - 1.0)

    assert overlay_total < bh_total
    assert result.upside_forgone > 0.0
    assert result.upside_forgone == pytest.approx(max(bh_total - overlay_total, 0.0))


def test_covered_call_at_least_matches_hold_on_flat_fixture() -> None:
    # A perfectly flat (zero-noise) price series, not the noisy conftest
    # `flat_ohlcv`: with real noise, the stock leg's per-*cycle* rebasing
    # (see the module docstring) vs. buy-and-hold's per-*day* compounding
    # is itself a (tiny, documented) source of drift unrelated to the
    # option overlay. A dead-flat series makes the stock leg contribute
    # exactly 0.0 either way, isolating the "premium yield, no cap cost"
    # claim this test exists to check. Zero costs for the same reason
    # spelled out in the rising-fixture test above.
    index = pd.bdate_range("2020-01-01", periods=300)
    close = pd.Series(100.0, index=index)
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1_000_000.0}
    )
    zero_costs = OverlayCosts(per_contract_commission=0.0, spread_haircut_pct=0.0)
    result = simulate_overlay(df, _covered_call_spec(), VolProxyConfig(), zero_costs, rate=0.03)
    valid = result.returns.notna()
    overlay_total = float((1.0 + result.returns[valid]).prod() - 1.0)
    bh_total = float((1.0 + result.underlying_returns[valid]).prod() - 1.0)

    assert result.n_rolls > 0
    assert len(result.events) == 0
    assert overlay_total >= bh_total


# ---------------------------------------------------------------------------
# Cash-secured put on a crash: loss bounded by the strike*100 cash reserve
# ---------------------------------------------------------------------------


def test_cash_secured_put_loss_bounded_on_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_constant_vol(monkeypatch, 0.20)
    closes = [100.0] * 10 + [20.0, 20.0]
    df = _frame_from_closes(closes)

    spec = OverlaySpec(
        structure=OverlayStructure.CASH_SECURED_PUT,
        dte_target=10,
        strike_selector=StrikeSelector(mode="otm_pct", value=0.05),
        roll_at_dte=0,
    )
    costs = OverlayCosts(per_contract_commission=1.0, spread_haircut_pct=0.10)

    result = simulate_overlay(df, spec, VolProxyConfig(), costs, rate=0.0)

    vol, rate, strike = 0.20, 0.0, 95.0
    premium0 = bs_price(100.0, strike, 10 / 252, vol, rate, PUT)
    capital_base = strike * 100.0

    open_cost = 1.0 + 0.10 * premium0 * 100.0
    close_cost = 1.0 + 0.10 * 75.0 * 100.0  # intrinsic at settlement = 95-20
    cycle_sum_expected = (100.0 * premium0 - 7500.0 - open_cost - close_cost) / capital_base

    assert result.returns.iloc[1:11].sum() == pytest.approx(cycle_sum_expected)
    # Structural bound: a single cycle can never lose more than its own cash
    # reserve (spot > 0 always keeps realized loss strictly inside -100%).
    assert cycle_sum_expected > -1.0
    assert result.summary()["overlay_max_drawdown"] >= -1.0 - 1e-9

    assert len(result.events) == 1
    assert result.events[0].strike == pytest.approx(95.0)
    assert result.events[0].spot == pytest.approx(20.0)
    assert result.events[0].moneyness == pytest.approx(20.0 / 95.0)


# ---------------------------------------------------------------------------
# Credit vertical spread: max loss <= width - credit, on a gap-through-both
# -strikes fixture
# ---------------------------------------------------------------------------


def test_vertical_spread_max_loss_bounded_by_width(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_constant_vol(monkeypatch, 0.20)
    closes = [100.0] * 10 + [50.0, 50.0]
    df = _frame_from_closes(closes)

    spec = OverlaySpec(
        structure=OverlayStructure.VERTICAL_SPREAD,
        dte_target=10,
        strike_selector=StrikeSelector(mode="otm_pct", value=0.05),
        spread_width_pct=0.05,
        kind=PUT,
        roll_at_dte=0,
    )
    costs = OverlayCosts(per_contract_commission=1.0, spread_haircut_pct=0.10)

    result = simulate_overlay(df, spec, VolProxyConfig(), costs, rate=0.0)

    vol, rate = 0.20, 0.0
    short_strike, long_strike = 95.0, 90.0
    width_dollars = 5.0
    premium0_short = bs_price(100.0, short_strike, 10 / 252, vol, rate, PUT)
    premium0_long = bs_price(100.0, long_strike, 10 / 252, vol, rate, PUT)
    credit0 = premium0_short - premium0_long
    capital_base = width_dollars * 100.0

    # Both legs are deep ITM at settlement (spot=50 < both strikes): the
    # option-only telescoped P&L collapses to exactly credit - width,
    # independent of how far through both strikes the crash goes.
    option_pnl = (100.0 * credit0) - (width_dollars * 100.0)
    assert abs(option_pnl) <= width_dollars * 100.0 + 1e-9

    open_cost = (1.0 + 0.10 * premium0_short * 100.0) + (1.0 + 0.10 * premium0_long * 100.0)
    close_cost = (1.0 + 0.10 * 45.0 * 100.0) + (1.0 + 0.10 * 40.0 * 100.0)  # intrinsics at spot=50
    cycle_sum_expected = (option_pnl - open_cost - close_cost) / capital_base

    assert result.returns.iloc[1:11].sum() == pytest.approx(cycle_sum_expected)

    assert len(result.events) == 1
    assert result.events[0].strike == pytest.approx(95.0)
    assert result.events[0].spot == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# LEAPS: loss bounded by the premium paid (its own capital base)
# ---------------------------------------------------------------------------


def test_leaps_loss_bounded_by_premium_paid() -> None:
    n = 60
    closes = list(np.linspace(100.0, 10.0, n))  # monotonic crash, no prior rally
    df = _frame_from_closes(closes)

    spec = OverlaySpec(
        structure=OverlayStructure.LEAPS,
        dte_target=252,
        strike_selector=StrikeSelector(mode="delta", value=0.75),
        roll_at_dte=5,
    )
    result = simulate_overlay(df, spec, VolProxyConfig(), OverlayCosts(), rate=0.0)

    valid = result.returns.notna()
    equity = (1.0 + result.returns[valid]).cumprod()
    assert (equity >= -1e-9).all()

    total_return = float(equity.iloc[-1] - 1.0)
    assert total_return >= -1.0 - 1e-9


# ---------------------------------------------------------------------------
# avoid_assignment rolls earlier than the dte trigger
# ---------------------------------------------------------------------------


def test_avoid_assignment_rolls_earlier_and_fewer_assignments(trending_ohlcv: pd.DataFrame) -> None:
    strike_selector = StrikeSelector(mode="otm_pct", value=0.05)
    spec_avoid = OverlaySpec(
        structure=OverlayStructure.COVERED_CALL,
        dte_target=30,
        strike_selector=strike_selector,
        roll_at_dte=0,  # hold to expiry unless avoid_assignment triggers early
        assignment_prob_trigger=0.5,
        avoid_assignment=True,
    )
    spec_hold = OverlaySpec(
        structure=OverlayStructure.COVERED_CALL,
        dte_target=30,
        strike_selector=strike_selector,
        roll_at_dte=0,
        assignment_prob_trigger=0.5,
        avoid_assignment=False,
    )

    costs = OverlayCosts()
    result_avoid = simulate_overlay(trending_ohlcv, spec_avoid, VolProxyConfig(), costs, rate=0.03)
    result_hold = simulate_overlay(trending_ohlcv, spec_hold, VolProxyConfig(), costs, rate=0.03)

    assert result_avoid.n_rolls > result_hold.n_rolls
    assert len(result_avoid.events) < len(result_hold.events)


# ---------------------------------------------------------------------------
# Truncation invariance (the v1 look-ahead idiom)
# ---------------------------------------------------------------------------

TRUNCATION_SPECS = [
    OverlaySpec(
        structure=OverlayStructure.COVERED_CALL,
        dte_target=21,
        strike_selector=StrikeSelector(mode="otm_pct", value=0.05),
        roll_at_dte=5,
    ),
    OverlaySpec(
        structure=OverlayStructure.CASH_SECURED_PUT,
        dte_target=21,
        strike_selector=StrikeSelector(mode="otm_pct", value=0.05),
        roll_at_dte=5,
    ),
    OverlaySpec(
        structure=OverlayStructure.VERTICAL_SPREAD,
        dte_target=21,
        strike_selector=StrikeSelector(mode="otm_pct", value=0.05),
        spread_width_pct=0.05,
        kind=CALL,
        roll_at_dte=5,
    ),
    OverlaySpec(
        structure=OverlayStructure.LEAPS,
        dte_target=252,
        strike_selector=StrikeSelector(mode="delta", value=0.75),
        roll_at_dte=21,
    ),
]


@pytest.mark.parametrize(
    "spec", TRUNCATION_SPECS, ids=[s.structure.value for s in TRUNCATION_SPECS]
)
def test_no_lookahead(spec: OverlaySpec, trending_ohlcv: pd.DataFrame) -> None:
    costs = OverlayCosts()
    full = simulate_overlay(trending_ohlcv, spec, VolProxyConfig(), costs, rate=0.03)
    truncated_df = trending_ohlcv.iloc[:TRUNCATE_AT]
    truncated = simulate_overlay(truncated_df, spec, VolProxyConfig(), costs, rate=0.03)

    pd.testing.assert_series_equal(
        full.returns.iloc[:TRUNCATE_AT], truncated.returns, check_names=False
    )


# ---------------------------------------------------------------------------
# Costs: monotonic in spread_haircut_pct; zero costs beat nonzero costs
# ---------------------------------------------------------------------------


def _total_return(returns: pd.Series) -> float:
    valid = returns.dropna()
    return float((1.0 + valid).prod() - 1.0)


def test_higher_haircut_strictly_lowers_return(trending_ohlcv: pd.DataFrame) -> None:
    spec = _covered_call_spec()
    vol_config = VolProxyConfig()

    low = simulate_overlay(
        trending_ohlcv, spec, vol_config, OverlayCosts(spread_haircut_pct=0.05), rate=0.03
    )
    high = simulate_overlay(
        trending_ohlcv, spec, vol_config, OverlayCosts(spread_haircut_pct=0.10), rate=0.03
    )

    assert _total_return(high.returns) < _total_return(low.returns)


def test_zero_costs_beat_nonzero_costs(trending_ohlcv: pd.DataFrame) -> None:
    spec = _covered_call_spec()
    vol_config = VolProxyConfig()

    zero = simulate_overlay(
        trending_ohlcv,
        spec,
        vol_config,
        OverlayCosts(per_contract_commission=0.0, spread_haircut_pct=0.0),
        rate=0.03,
    )
    nonzero = simulate_overlay(trending_ohlcv, spec, vol_config, OverlayCosts(), rate=0.03)

    assert _total_return(zero.returns) > _total_return(nonzero.returns)
