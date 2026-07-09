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
from funnel.options.pricing import OptionKind, VolProxyConfig, bs_price, strike_for_delta

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
    capital_base = 100.0 * 100.0  # spot_entry * 100 shares * 1 contract

    open_cost = 1.0 + 0.10 * premium0 * 100.0
    close_cost = 1.0 + 0.10 * 5.0 * 100.0  # intrinsic at settlement = 110-105

    # Independent day-by-day re-derivation of the equity-relative return
    # path (module docstring "Daily return convention"): equity starts at
    # capital_base, and each day's return is that day's net P&L divided by
    # the *prior day's equity*, not the fixed capital_base — the two only
    # coincide on day 1 (equity has not moved from capital_base yet). Was:
    # a single ``(pnl - cost) / capital_base`` per day, whose daily values
    # summed linearly to the cycle P&L; now the per-day quotient changes
    # (tiny drift here) and the cycle total is a *product*, not a sum.
    premiums = [bs_price(closes[i], strike, (10 - i) / 252, vol, rate, CALL) for i in range(11)]
    equity = capital_base
    expected_returns: list[float] = []
    for i in range(1, 11):
        option_pnl = 100.0 * (premiums[i - 1] - premiums[i])
        stock_pnl = 100.0 * (closes[i] - closes[i - 1])
        day_cost = (open_cost if i == 1 else 0.0) + (close_cost if i == 10 else 0.0)
        net_pnl = option_pnl + stock_pnl - day_cost
        expected_returns.append(net_pnl / equity)
        equity += net_pnl

    day1_expected = (100.0 * (premium0 - premium1) - open_cost) / capital_base
    # day 1: equity == capital_base still, so this matches the new convention too.
    assert day1_expected == pytest.approx(expected_returns[0])
    # day10_expected (old): 0.044900161047454784; new: 0.044816848914772035
    # (a ~0.02% relative drift — equity had crept ~$8 above capital_base by
    # day 10 from days 2-9's small theta gains, so the day-10 quotient's
    # denominator is now that slightly larger equity, not the fixed base).

    assert pd.isna(result.returns.iloc[0])
    for offset, expected in enumerate(expected_returns):
        assert result.returns.iloc[1 + offset] == pytest.approx(expected)

    # Compounding identity (module docstring): capital_base *
    # cumprod(1 + returns) reproduces the true dollar equity path exactly.
    # cycle_sum_expected (old, arithmetic sum): 0.04675910773157299 — this
    # coincidentally still equals the new product-based total (below) to
    # float precision, because with no clamping this cycle the telescoping
    # total P&L / capital_base is convention-independent; only the per-day
    # breakdown above changed.
    cycle_total_expected = float((1.0 + pd.Series(expected_returns)).prod()) - 1.0
    cycle_total_actual = float((1.0 + result.returns.iloc[1:11]).prod()) - 1.0
    assert cycle_total_actual == pytest.approx(cycle_total_expected)
    assert capital_base * (1.0 + cycle_total_actual) == pytest.approx(equity)

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

    # Independent day-by-day re-derivation of the equity-relative return
    # path (see the covered-call fixture above for the convention). The
    # crash lands entirely on day 10, so days 1-9 (flat spot) are near-
    # identical to the old fixed-base numbers; day 10's return now divides
    # by day-9's equity (~capital_base, barely moved) rather than the
    # fixed capital_base — a small drift in this fixture, but this is
    # exactly the mechanism that produces sub -100% daily returns on a
    # more extreme gap (see the new gap-fixture tests below).
    premiums = [bs_price(closes[i], strike, (10 - i) / 252, vol, rate, PUT) for i in range(11)]
    equity = capital_base
    expected_returns: list[float] = []
    for i in range(1, 11):
        option_pnl = -100.0 * premiums[i] - (-100.0 * premiums[i - 1])
        interest = capital_base * (rate / 252.0)  # rate=0.0 here, kept for clarity
        day_cost = (open_cost if i == 1 else 0.0) + (close_cost if i == 10 else 0.0)
        net_pnl = option_pnl + interest - day_cost
        expected_returns.append(net_pnl / equity)
        equity += net_pnl

    for offset, expected in enumerate(expected_returns):
        assert result.returns.iloc[1 + offset] == pytest.approx(expected)

    cycle_total_expected = float((1.0 + pd.Series(expected_returns)).prod()) - 1.0
    cycle_total_actual = float((1.0 + result.returns.iloc[1:11]).prod()) - 1.0
    assert cycle_total_actual == pytest.approx(cycle_total_expected)
    # cycle_sum_expected (old, arithmetic sum): -0.8669117290491474; new
    # product-based total: -0.8669117290491475 (same to float precision —
    # no clamping triggered in this fixture, so the telescoping cycle
    # total is convention-independent; see the covered-call fixture note).
    # Structural bound: a single cycle can never lose more than its own cash
    # reserve (spot > 0 always keeps realized loss strictly inside -100%),
    # and now every INDIVIDUAL day's return is bounded too (not just the
    # cycle total) — see the module docstring and the new gap-fixture
    # tests for the case where the old fixed-base convention broke this.
    assert cycle_total_actual > -1.0
    assert (result.returns.iloc[1:11] > -1.0).all()
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

    # Independent day-by-day re-derivation of the equity-relative return
    # path (see the covered-call fixture above for the convention). Unlike
    # the covered-call/CSP fixtures, this crash (100 -> 50, through both
    # strikes) exceeds the position's own capital base *before* costs are
    # even applied: the old fixed-base convention divided every day's P&L
    # by the fixed $500 width collateral, so the settlement day alone
    # produced a return far below -100%, and the cycle's arithmetic sum
    # was -267.6% — a number the old test asserted as correct with no
    # bound check catching it. Under equity-relative returns, equity
    # cannot go negative: it is clamped at 0.0 on the day it would (see
    # module docstring's near-zero-equity guard), so day 10 here is
    # exactly -100% (total loss of the width collateral), not -267%.
    premiums_short = [
        bs_price(closes[i], short_strike, (10 - i) / 252, vol, rate, PUT) for i in range(11)
    ]
    premiums_long = [
        bs_price(closes[i], long_strike, (10 - i) / 252, vol, rate, PUT) for i in range(11)
    ]
    equity = capital_base
    expected_returns: list[float] = []
    for i in range(1, 11):
        book_prev = -100.0 * premiums_short[i - 1] + 100.0 * premiums_long[i - 1]
        book_end = -100.0 * premiums_short[i] + 100.0 * premiums_long[i]
        day_cost = (open_cost if i == 1 else 0.0) + (close_cost if i == 10 else 0.0)
        net_pnl = (book_end - book_prev) - day_cost
        floor = 1e-6 * abs(capital_base)
        if equity <= floor:
            expected_returns.append(0.0)
        else:
            new_equity = max(equity + net_pnl, 0.0)
            expected_returns.append((new_equity - equity) / equity)
            equity = new_equity

    for offset, expected in enumerate(expected_returns):
        assert result.returns.iloc[1 + offset] == pytest.approx(expected)

    cycle_total_expected = float((1.0 + pd.Series(expected_returns)).prod()) - 1.0
    cycle_total_actual = float((1.0 + result.returns.iloc[1:11]).prod()) - 1.0
    assert cycle_total_actual == pytest.approx(cycle_total_expected)
    # cycle_sum_expected (old, arithmetic sum, buggy convention): -2.6763781144119076
    # (a -267.6% "return" the old test asserted without a bound check).
    # New product-based cycle total: -1.0 exactly (clamped: the position
    # cannot lose more than its own width collateral).
    assert cycle_total_actual == pytest.approx(-1.0)
    assert (result.returns.iloc[1:11] >= -1.0).all()
    assert result.summary()["overlay_max_drawdown"] >= -1.0 - 1e-9

    assert len(result.events) == 1
    assert result.events[0].strike == pytest.approx(95.0)
    assert result.events[0].spot == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Gap fixtures (adversarial review repro): the old fixed-capital-base
# convention could produce daily returns below -100% once a day's
# mark-to-model P&L exceeded the fixed entry capital base — a smooth price
# path can't trigger this (P&L moves gradually relative to the base), so
# each fixture here forces a real overnight gap. Every structure must show
# every daily return >= -100% and cumprod(1 + returns) >= 0 throughout.
# ---------------------------------------------------------------------------


def _noisy_path(
    n: int, seed: int, mu: float = 0.0002, sigma: float = 0.015, start: float = 100.0
) -> list[float]:
    """A small deterministic (seeded) multiplicative random walk, used to
    give a gap fixture realistic day-to-day noise around the gap itself
    (a perfectly flat pre/post path can hide or distort the mechanism
    under test — see the module's near-zero-equity guard discussion)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(mu, sigma, n)
    return list(start * np.cumprod(1.0 + rets))


def _assert_bounded_returns(returns: pd.Series) -> None:
    valid = returns.dropna()
    assert (valid > -1.0).all()
    assert (valid.notna()).all()
    equity = (1.0 + valid).cumprod()
    assert (equity >= -1e-9).all()


def test_leaps_shipped_grid_survives_two_day_gap() -> None:
    """Shipped-grid LEAPS (delta=0.70, dte=504, from ``options/grid.py``'s
    ``_LEAPS_DELTAS``/``_LEAPS_DTES``) through a +40%/-35% two-day gap.
    Reproduces the adversarial review's flagship failure: under the old
    fixed-capital-base convention this fixture produced a single-day
    return of -3.56 and a max_drawdown of -7.22 (-721.5%, impossible for a
    loss-bounded structure and uninterpretable by ``metrics.cumprod``-
    based Sharpe/drawdown).
    """
    pre = _noisy_path(90, seed=1)
    gap_up = pre[-1] * 1.40
    gap_down = gap_up * 0.65
    post = _noisy_path(60, seed=2, start=gap_down)
    df = _frame_from_closes(pre + [gap_up, gap_down] + post)

    spec = OverlaySpec(
        structure=OverlayStructure.LEAPS,
        dte_target=504,
        strike_selector=StrikeSelector(mode="delta", value=0.70),
        roll_at_dte=5,
    )
    result = simulate_overlay(df, spec, VolProxyConfig(), OverlayCosts(), rate=0.03)

    _assert_bounded_returns(result.returns)
    assert result.summary()["overlay_max_drawdown"] >= -1.0 - 1e-9


def test_vertical_spread_survives_gap_through_both_strikes() -> None:
    """A credit put vertical, gapped in one day through both the short and
    long strikes (a -25% overnight move). Bounded loss must still hold on
    every individual day, not just the cycle total."""
    pre = _noisy_path(30, seed=3)
    gap = pre[-1] * 0.75
    post = _noisy_path(30, seed=4, start=gap)
    df = _frame_from_closes(pre + [gap] + post)

    spec = OverlaySpec(
        structure=OverlayStructure.VERTICAL_SPREAD,
        dte_target=60,
        strike_selector=StrikeSelector(mode="otm_pct", value=0.05),
        spread_width_pct=0.05,
        kind=PUT,
        roll_at_dte=5,
    )
    result = simulate_overlay(df, spec, VolProxyConfig(), OverlayCosts(), rate=0.03)

    _assert_bounded_returns(result.returns)
    assert result.summary()["overlay_max_drawdown"] >= -1.0 - 1e-9


def test_covered_call_survives_crash() -> None:
    """A covered call through a -65% overnight crash. Loss is bounded by
    the stock's own decline to zero (module docstring)."""
    pre = _noisy_path(60, seed=5)
    gap = pre[-1] * 0.35
    post = _noisy_path(30, seed=6, start=gap)
    df = _frame_from_closes(pre + [gap] + post)

    spec = OverlaySpec(
        structure=OverlayStructure.COVERED_CALL,
        dte_target=30,
        strike_selector=StrikeSelector(mode="otm_pct", value=0.05),
        roll_at_dte=5,
    )
    result = simulate_overlay(df, spec, VolProxyConfig(), OverlayCosts(), rate=0.03)

    _assert_bounded_returns(result.returns)
    assert result.summary()["overlay_max_drawdown"] >= -1.0 - 1e-9


def test_deep_otm_leaps_bounded_on_random_walk() -> None:
    """The adversarial review's trivial repro: a deep-OTM (delta=0.02)
    LEAPS on a seeded random walk. Under the old fixed-capital-base
    convention this produced daily returns from -1.30 to +3.81 (2 days
    below -100%) and a max_drawdown of -1.0001. The position is expected
    to expire worthless on most paths — an exact -100% day (full loss of
    the tiny premium) is itself correct and expected here (unlike the
    strict '> -1.0' gap fixtures above, whose LEAPS retain real extrinsic
    value going into the gap)."""
    closes = _noisy_path(300, seed=42, mu=0.0005, sigma=0.02)
    df = _frame_from_closes(closes)

    spec = OverlaySpec(
        structure=OverlayStructure.LEAPS,
        dte_target=252,
        strike_selector=StrikeSelector(mode="delta", value=0.02),
        roll_at_dte=5,
    )
    result = simulate_overlay(df, spec, VolProxyConfig(), OverlayCosts(), rate=0.03)

    valid = result.returns.dropna()
    assert (valid >= -1.0).all()
    assert (valid > -1.0 - 1e-9).all()
    equity = (1.0 + valid).cumprod()
    assert (equity >= -1e-9).all()
    assert result.summary()["overlay_max_drawdown"] >= -1.0 - 1e-9


# ---------------------------------------------------------------------------
# Compounding identity: capital_base * cumprod(1 + returns) reproduces the
# tracked dollar equity path exactly, within a cycle (module docstring).
# Verified independently via bs_price (not by calling simulate_overlay's own
# internals) on a genuinely volatile single-cycle fixture per structure.
# ---------------------------------------------------------------------------


def _independent_single_cycle_equity(
    structure: OverlayStructure,
    closes: list[float],
    dte: int,
    vol: float,
    rate: float,
    costs: OverlayCosts,
    otm_pct: float = 0.05,
    spread_width_pct: float = 0.05,
    kind: OptionKind = CALL,
) -> tuple[pd.Series, float]:
    """Independent (bs_price-only) re-derivation of the equity-relative
    return path this module's convention prescribes for a single
    hold-to-expiry cycle (no rolls): mirrors the per-day recurrence
    documented in the ``overlays`` module docstring."""
    n = len(closes)
    spot0 = closes[0]
    t0 = dte / 252.0

    if structure is OverlayStructure.COVERED_CALL:
        strike = spot0 * (1.0 + otm_pct)
        legs = [(CALL, strike, -1.0)]
        capital_base = spot0 * 100.0
    elif structure is OverlayStructure.CASH_SECURED_PUT:
        strike = spot0 * (1.0 - otm_pct)
        legs = [(PUT, strike, -1.0)]
        capital_base = strike * 100.0
    elif structure is OverlayStructure.VERTICAL_SPREAD:
        short_strike = spot0 * (1.0 - otm_pct) if kind is PUT else spot0 * (1.0 + otm_pct)
        width = spread_width_pct * spot0
        long_strike = short_strike - width if kind is PUT else short_strike + width
        legs = [(kind, short_strike, -1.0), (kind, long_strike, 1.0)]
        capital_base = width * 100.0
    else:  # LEAPS
        strike = strike_for_delta(spot0, 0.70, t0, vol, rate, CALL)
        legs = [(CALL, strike, 1.0)]
        capital_base = 100.0 * bs_price(spot0, strike, t0, vol, rate, CALL)

    def book_value(i: int, t_years: float) -> float:
        return 100.0 * sum(
            direction * bs_price(closes[i], strike_, t_years, vol, rate, kind_)
            for kind_, strike_, direction in legs
        )

    def leg_cost(i: int, t_years: float) -> float:
        total = 0.0
        for kind_, strike_, _direction in legs:
            price = bs_price(closes[i], strike_, t_years, vol, rate, kind_)
            total += costs.per_contract_commission
            total += costs.spread_haircut_pct * price * 100.0
        return total

    open_cost = leg_cost(0, t0)
    book_prev = book_value(0, t0)
    equity = capital_base
    returns: list[float] = [float("nan")]
    for i in range(1, n):
        remaining = dte - i
        t_years_end = max(remaining, 0) / 252.0
        mtm_end = book_value(i, t_years_end)
        pnl = mtm_end - book_prev
        if structure is OverlayStructure.COVERED_CALL:
            pnl += 100.0 * (closes[i] - closes[i - 1])
        elif structure is OverlayStructure.CASH_SECURED_PUT:
            pnl += capital_base * (rate / 252.0)
        day_cost = open_cost if i == 1 else 0.0
        should_settle = remaining <= 0
        if should_settle:
            day_cost += leg_cost(i, t_years_end)
        net_pnl = pnl - day_cost
        floor = 1e-6 * abs(capital_base)
        if equity <= floor:
            returns.append(0.0)
        else:
            new_equity = max(equity + net_pnl, 0.0)
            returns.append((new_equity - equity) / equity)
            equity = new_equity
        book_prev = mtm_end
        if should_settle:
            break

    index = pd.bdate_range("2020-01-01", periods=len(returns))
    return pd.Series(returns, index=index, dtype="float64"), capital_base


IDENTITY_STRUCTURES = [
    OverlayStructure.COVERED_CALL,
    OverlayStructure.CASH_SECURED_PUT,
    OverlayStructure.VERTICAL_SPREAD,
    OverlayStructure.LEAPS,
]


def _identity_spec(structure: OverlayStructure, dte: int) -> tuple[OverlaySpec, OptionKind]:
    """The single-cycle (``roll_at_dte=0``) spec used by the compounding-
    identity test, plus the vertical spread's leg ``kind`` (``CALL`` for
    every other structure, unused by ``_independent_single_cycle_equity``
    otherwise)."""
    if structure is OverlayStructure.LEAPS:
        selector = StrikeSelector(mode="delta", value=0.70)
        return (
            OverlaySpec(
                structure=structure, dte_target=dte, strike_selector=selector, roll_at_dte=0
            ),
            CALL,
        )
    selector = StrikeSelector(mode="otm_pct", value=0.05)
    if structure is OverlayStructure.VERTICAL_SPREAD:
        return (
            OverlaySpec(
                structure=structure,
                dte_target=dte,
                strike_selector=selector,
                spread_width_pct=0.05,
                kind=PUT,
                roll_at_dte=0,
            ),
            PUT,
        )
    return (
        OverlaySpec(structure=structure, dte_target=dte, strike_selector=selector, roll_at_dte=0),
        CALL,
    )


@pytest.mark.parametrize(
    "structure", IDENTITY_STRUCTURES, ids=[s.value for s in IDENTITY_STRUCTURES]
)
def test_compounding_identity_per_cycle(
    monkeypatch: pytest.MonkeyPatch, structure: OverlayStructure
) -> None:
    _patch_constant_vol(monkeypatch, 0.22)
    closes = _noisy_path(45, seed=11, mu=0.0003, sigma=0.02)
    df = _frame_from_closes(closes)
    dte = 40
    costs = OverlayCosts(per_contract_commission=0.65, spread_haircut_pct=0.05)

    spec, kind = _identity_spec(structure, dte)
    result = simulate_overlay(df, spec, VolProxyConfig(), costs, rate=0.03)

    expected_returns, capital_base = _independent_single_cycle_equity(
        structure, closes, dte, 0.22, 0.03, costs, kind=kind
    )
    actual = result.returns.iloc[: len(expected_returns)]

    pd.testing.assert_series_equal(
        actual.reset_index(drop=True),
        expected_returns.reset_index(drop=True),
        check_names=False,
        atol=1e-9,
    )

    valid = actual.dropna()
    equity_path = capital_base * (1.0 + valid).cumprod()
    expected_equity_path = capital_base * (1.0 + expected_returns.dropna()).cumprod()
    assert np.allclose(equity_path.to_numpy(), expected_equity_path.to_numpy(), atol=1e-6)


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
