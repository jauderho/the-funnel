"""Tests for funnel.options.pricing: golden BSM values, put-call parity,
monotonicity, bounds, edge cases, strike inversion, inverse-CDF accuracy,
and the causal vol proxy (PLAN.md "v2 — Options Overlay Module").
"""

import itertools
import math
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from funnel.options.pricing import (
    OptionKind,
    VolProxyConfig,
    _inv_norm_cdf,
    bs_delta,
    bs_price,
    prob_itm,
    realized_vol,
    strike_for_delta,
    synthetic_iv,
)

CALL = OptionKind.CALL
PUT = OptionKind.PUT

TRUNCATE_AT = 400

# ---------------------------------------------------------------------------
# Golden BSM values
# ---------------------------------------------------------------------------

# (spot, strike, t_years, vol, rate, expected_call, expected_put) — each
# hand-derived from the standard BSM closed form (d1/d2, N via erf) with an
# independent script, not copied from a table. Case 1 is the widely-cited
# S=K=100, T=1, vol=0.2, r=0.05 textbook example (~10.4506 / ~5.5735).
GOLDEN_CASES = [
    (100.0, 100.0, 1.0, 0.20, 0.05, 10.450584, 5.573526),
    (50.0, 45.0, 0.5, 0.30, 0.02, 7.290705, 1.842948),
    (30.0, 35.0, 0.25, 0.25, 0.01, 0.220587, 5.133196),
]


@pytest.mark.parametrize("spot,strike,t,vol,r,call_expected,put_expected", GOLDEN_CASES)
def test_golden_bsm_values(spot, strike, t, vol, r, call_expected, put_expected) -> None:
    assert bs_price(spot, strike, t, vol, r, CALL) == pytest.approx(call_expected, abs=1e-3)
    assert bs_price(spot, strike, t, vol, r, PUT) == pytest.approx(put_expected, abs=1e-3)


# ---------------------------------------------------------------------------
# Put-call parity: C - P = S - K*e^(-rt), across a grid including t=0/vol=0
# ---------------------------------------------------------------------------

PARITY_SPOTS = (50.0, 100.0, 150.0)
PARITY_STRIKES = (80.0, 100.0, 120.0)
PARITY_TS = (0.0, 0.1, 0.5, 1.0, 2.0)
PARITY_VOLS = (0.0, 0.1, 0.3, 0.5)
PARITY_RATES = (0.0, 0.02, 0.05)


@pytest.mark.parametrize(
    "spot,strike,t,vol,r",
    list(itertools.product(PARITY_SPOTS, PARITY_STRIKES, PARITY_TS, PARITY_VOLS, PARITY_RATES)),
)
def test_put_call_parity(spot, strike, t, vol, r) -> None:
    call = bs_price(spot, strike, t, vol, r, CALL)
    put = bs_price(spot, strike, t, vol, r, PUT)
    forward_parity = spot - strike * math.exp(-r * t)
    assert call - put == pytest.approx(forward_parity, abs=1e-9)


# ---------------------------------------------------------------------------
# Monotonicity
# ---------------------------------------------------------------------------

MONO_RATES = (0.0, 0.02, 0.05)
MONO_VOLS = (0.15, 0.25, 0.35)
MONO_TS = (0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0)
MONO_STRIKES = (80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0)


@pytest.mark.parametrize("rate,vol", list(itertools.product(MONO_RATES, MONO_VOLS)))
def test_call_price_increasing_in_time_otm(rate, vol) -> None:
    """OTM call (K=110 > S=100): price strictly increases with time to expiry."""
    prices = [bs_price(100.0, 110.0, t, vol, rate, CALL) for t in MONO_TS]
    assert all(a < b - 1e-9 for a, b in itertools.pairwise(prices))


@pytest.mark.parametrize("rate,vol", list(itertools.product(MONO_RATES, MONO_VOLS)))
def test_put_price_increasing_in_time_otm(rate, vol) -> None:
    """OTM put (K=90 < S=100): mirror image of the call case above."""
    prices = [bs_price(100.0, 90.0, t, vol, rate, PUT) for t in MONO_TS]
    assert all(a < b - 1e-9 for a, b in itertools.pairwise(prices))


@pytest.mark.parametrize(
    "rate,t,strike", list(itertools.product(MONO_RATES, (0.1, 0.5, 1.0, 2.0), (90.0, 100.0, 110.0)))
)
def test_price_increasing_in_vol(rate, t, strike) -> None:
    vols = (0.05, 0.1, 0.2, 0.3, 0.5, 0.8)
    call_prices = [bs_price(100.0, strike, t, v, rate, CALL) for v in vols]
    put_prices = [bs_price(100.0, strike, t, v, rate, PUT) for v in vols]
    assert all(a < b - 1e-9 for a, b in itertools.pairwise(call_prices))
    assert all(a < b - 1e-9 for a, b in itertools.pairwise(put_prices))


@pytest.mark.parametrize(
    "rate,t,vol", list(itertools.product(MONO_RATES, (0.1, 0.5, 1.0, 2.0), (0.15, 0.3, 0.5)))
)
def test_call_price_decreasing_in_strike(rate, t, vol) -> None:
    prices = [bs_price(100.0, k, t, vol, rate, CALL) for k in MONO_STRIKES]
    assert all(a > b + 1e-9 for a, b in itertools.pairwise(prices))


@pytest.mark.parametrize(
    "rate,t,vol", list(itertools.product(MONO_RATES, (0.1, 0.5, 1.0, 2.0), (0.15, 0.3, 0.5)))
)
def test_put_price_increasing_in_strike(rate, t, vol) -> None:
    """Put mirror-image of the call strike test above."""
    prices = [bs_price(100.0, k, t, vol, rate, PUT) for k in MONO_STRIKES]
    assert all(a < b - 1e-9 for a, b in itertools.pairwise(prices))


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------

BOUNDS_CASES = list(
    itertools.product(
        (50.0, 100.0, 150.0), (80.0, 100.0, 120.0), (0.1, 1.0, 3.0), (0.1, 0.3, 0.6), (0.0, 0.05)
    )
)


@pytest.mark.parametrize("spot,strike,t,vol,r", BOUNDS_CASES)
def test_call_delta_in_unit_interval(spot, strike, t, vol, r) -> None:
    delta = bs_delta(spot, strike, t, vol, r, CALL)
    assert 0.0 <= delta <= 1.0


@pytest.mark.parametrize("spot,strike,t,vol,r", BOUNDS_CASES)
def test_put_delta_in_negative_unit_interval(spot, strike, t, vol, r) -> None:
    delta = bs_delta(spot, strike, t, vol, r, PUT)
    assert -1.0 <= delta <= 0.0


@pytest.mark.parametrize(
    "spot,strike,t,vol,r,kind", [(*c, k) for c in BOUNDS_CASES for k in (CALL, PUT)]
)
def test_prob_itm_in_unit_interval(spot, strike, t, vol, r, kind) -> None:
    assert 0.0 <= prob_itm(spot, strike, t, vol, r, kind) <= 1.0


@pytest.mark.parametrize(
    "spot,strike,t,vol,r,kind", [(*c, k) for c in BOUNDS_CASES for k in (CALL, PUT)]
)
def test_price_nonnegative(spot, strike, t, vol, r, kind) -> None:
    assert bs_price(spot, strike, t, vol, r, kind) >= -1e-9


@pytest.mark.parametrize("spot,strike,t,vol,r", BOUNDS_CASES)
def test_call_price_at_least_intrinsic(spot, strike, t, vol, r) -> None:
    """European call on a non-dividend underlying is bounded below by
    intrinsic value; this does NOT hold for European puts (a deep-ITM
    European put can price below intrinsic due to time value of money —
    it cannot be exercised early to realize the strike sooner)."""
    price = bs_price(spot, strike, t, vol, r, CALL)
    intrinsic = max(spot - strike, 0.0)
    assert price >= intrinsic - 1e-9


# ---------------------------------------------------------------------------
# Edge cases: t=0, vol=0, deep ITM/OTM extremes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spot,strike", [(100.0, 90.0), (100.0, 100.0), (100.0, 110.0)])
@pytest.mark.parametrize("kind", [CALL, PUT])
def test_t_zero_is_intrinsic(spot, strike, kind) -> None:
    expected = max(spot - strike, 0.0) if kind is CALL else max(strike - spot, 0.0)
    assert bs_price(spot, strike, 0.0, 0.2, 0.05, kind) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("spot,strike", [(100.0, 90.0), (100.0, 100.0), (100.0, 110.0)])
@pytest.mark.parametrize("kind", [CALL, PUT])
def test_vol_zero_is_discounted_forward_intrinsic(spot, strike, kind) -> None:
    t, r = 1.0, 0.05
    discount = math.exp(-r * t)
    expected = (
        max(spot - strike * discount, 0.0) if kind is CALL else max(strike * discount - spot, 0.0)
    )
    assert bs_price(spot, strike, t, 0.0, r, kind) == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    "spot,strike", [(1e-6, 1e6), (1e6, 1e-6), (1e-3, 100.0), (100.0, 1e-3), (100.0, 1e9)]
)
@pytest.mark.parametrize("kind", [CALL, PUT])
def test_deep_itm_otm_extremes_never_nan(spot, strike, kind) -> None:
    price = bs_price(spot, strike, 1.0, 0.3, 0.05, kind)
    delta = bs_delta(spot, strike, 1.0, 0.3, 0.05, kind)
    prob = prob_itm(spot, strike, 1.0, 0.3, 0.05, kind)
    assert not math.isnan(price)
    assert not math.isnan(delta)
    assert not math.isnan(prob)


# ---------------------------------------------------------------------------
# strike_for_delta round-trip
# ---------------------------------------------------------------------------

ROUND_TRIP_DELTAS_CALL = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ROUND_TRIP_TV = list(itertools.product((0.1, 0.5, 1.0, 2.0), (0.15, 0.3, 0.5)))


@pytest.mark.parametrize("target_delta", ROUND_TRIP_DELTAS_CALL)
@pytest.mark.parametrize("t,vol", ROUND_TRIP_TV)
def test_strike_for_delta_round_trip_call(t, vol, target_delta) -> None:
    spot, rate = 100.0, 0.03
    strike = strike_for_delta(spot, target_delta, t, vol, rate, CALL)
    assert bs_delta(spot, strike, t, vol, rate, CALL) == pytest.approx(target_delta, abs=1e-6)


@pytest.mark.parametrize("target_delta", [-d for d in ROUND_TRIP_DELTAS_CALL])
@pytest.mark.parametrize("t,vol", ROUND_TRIP_TV)
def test_strike_for_delta_round_trip_put(t, vol, target_delta) -> None:
    spot, rate = 100.0, 0.03
    strike = strike_for_delta(spot, target_delta, t, vol, rate, PUT)
    assert bs_delta(spot, strike, t, vol, rate, PUT) == pytest.approx(target_delta, abs=1e-6)


# ---------------------------------------------------------------------------
# Inverse-CDF accuracy (Acklam's algorithm)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p,expected_quantile",
    [
        (0.975, 1.959964),
        (0.5, 0.0),
        (0.025, -1.959964),
        (0.9, 1.281552),
        (0.1, -1.281552),
        (0.999, 3.090232),
        (0.001, -3.090232),
    ],
)
def test_inv_norm_cdf_matches_known_quantiles(p, expected_quantile) -> None:
    assert _inv_norm_cdf(p) == pytest.approx(expected_quantile, abs=1e-6)


def test_inv_norm_cdf_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        _inv_norm_cdf(0.0)
    with pytest.raises(ValueError):
        _inv_norm_cdf(1.0)


# ---------------------------------------------------------------------------
# Causal volatility proxy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["rolling", "ewma"])
def test_realized_vol_truncation_invariance(
    trending_ohlcv: pd.DataFrame, method: Literal["rolling", "ewma"]
) -> None:
    """Same look-ahead guard idiom as tests/test_lookahead.py: truncating the
    future must never change a past value."""
    close = trending_ohlcv["close"]
    full = realized_vol(close, window=21, method=method)
    truncated = realized_vol(close.iloc[:TRUNCATE_AT], window=21, method=method)
    pd.testing.assert_series_equal(full.iloc[:TRUNCATE_AT], truncated, check_names=False)


@pytest.mark.parametrize("method", ["rolling", "ewma"])
def test_realized_vol_warmup_is_nan(
    trending_ohlcv: pd.DataFrame, method: Literal["rolling", "ewma"]
) -> None:
    window = 21
    vol = realized_vol(trending_ohlcv["close"], window=window, method=method)
    assert vol.iloc[:window].isna().all()
    assert vol.iloc[window:].notna().all()


def test_realized_vol_annualization_known_value() -> None:
    """Alternating +-1% daily log returns have a closed-form rolling std."""
    window = 20
    n = 60
    log_returns = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(n)])
    close = pd.Series(100.0 * np.exp(np.cumsum(log_returns)))
    close = pd.concat([pd.Series([100.0]), close]).reset_index(drop=True)

    vol = realized_vol(close, window=window, method="rolling")

    expected_daily_std = 0.01 * math.sqrt(window / (window - 1))
    expected_annualized = expected_daily_std * math.sqrt(252)
    assert vol.iloc[-1] == pytest.approx(expected_annualized, abs=1e-9)


def test_synthetic_iv_multiplier_scales_unclipped_vol(trending_ohlcv: pd.DataFrame) -> None:
    close = trending_ohlcv["close"]
    base = synthetic_iv(
        close, VolProxyConfig(window=21, method="ewma", risk_premium_multiplier=1.0, floor=0.0)
    )
    doubled = synthetic_iv(
        close, VolProxyConfig(window=21, method="ewma", risk_premium_multiplier=2.0, floor=0.0)
    )
    ratio = (doubled / base).dropna().to_numpy()
    assert np.allclose(ratio, 2.0, atol=1e-9)


def test_synthetic_iv_floor_applied(flat_ohlcv: pd.DataFrame) -> None:
    """A near-flat price series has realized vol far below the floor; the
    floor must clamp the (tiny) scaled realized vol up to its value."""
    close = flat_ohlcv["close"]
    config = VolProxyConfig(window=21, method="ewma", risk_premium_multiplier=1.1, floor=0.05)
    iv = synthetic_iv(close, config)
    assert np.allclose(iv.dropna().to_numpy(), 0.05, atol=1e-9)


def test_synthetic_iv_warmup_is_nan(trending_ohlcv: pd.DataFrame) -> None:
    config = VolProxyConfig(window=21, method="ewma", risk_premium_multiplier=1.1, floor=0.05)
    iv = synthetic_iv(trending_ohlcv["close"], config)
    assert iv.iloc[: config.window].isna().all()
    assert iv.iloc[config.window :].notna().all()
