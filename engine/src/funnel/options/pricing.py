"""Black-Scholes-Merton option pricing and a causal synthetic-IV proxy.

MODELING GROUND RULES (see PLAN.md, "v2 — Options Overlay Module")
--------------------------------------------------------------------------
Historical option chains are not available from free sources, so every
price, delta, and probability produced here is a **model price**, not a
market price. Two consequences follow directly and must be respected by
every caller:

1. **Dividend-adjusted underlying, q = 0.** Pricing is computed on the
   same dividend-ADJUSTED close series the rest of the engine uses
   (``funnel.data.sources``, ``auto_adjust=True``), with a dividend yield
   of ``q = 0`` in the BSM formula. Rationale: adjusted prices already
   fold dividends into price appreciation, putting the whole backtest in
   a total-return frame consistent with v1. Limitation: this means
   dividend-driven early exercise/assignment of American-style options
   cannot be modeled — the framework only prices (and only ever will
   price) the European exercise-at-expiry case.
2. **"Assignment probability" is a model quantity, not a forecast.**
   Because early assignment is unmodeled, every probability this module
   reports (``prob_itm``) is the *risk-neutral model probability of being
   in-the-money at expiry*, P(S_T beyond K) under the BSM measure — not
   a real-world assignment probability and not a market-implied one. Any
   downstream report (V2-M2 onward) MUST label these outputs as model
   P(ITM at expiry), never as "assignment probability" unqualified.

Rates are passed as a plain ``rate`` float argument by every function;
this module holds no opinion on which rate to use (that is V2-M2's
business).

Scalar API, vectorized by looping
----------------------------------
Every pricing function here is a **scalar** function (matches this
module's callers, which price one contract on one date at a time). This
mirrors the rest of the strategy/indicator code base's convention of
writing a scalar core and applying it row-by-row via ``.apply()`` /
``rolling().apply()`` (see ``funnel.strategies.indicators``) rather than
hand-rolling a second, array-broadcast implementation of the same
formula. Callers that need a price/delta/probability *series* should
``.apply()`` these functions over a ``pd.Series``/``pd.DataFrame``.
"""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
import pandas as pd

_SQRT_2 = math.sqrt(2.0)
_TRADING_DAYS_PER_YEAR = 252


class OptionKind(StrEnum):
    """European option type."""

    CALL = "call"
    PUT = "put"


# ---------------------------------------------------------------------------
# Normal distribution helpers
# ---------------------------------------------------------------------------


def _norm_cdf(x: float) -> float:
    """Standard normal CDF, computed via ``math.erf`` (no scipy import)."""
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))


# Peter Acklam's rational approximation to the inverse standard normal CDF.
# Relative error <= 1.15e-9 over the full open interval (0, 1). Reference:
# https://web.archive.org/web/20151030215612/http://home.online.no/~pjacklam/notes/invnorm/
_ACKLAM_A = (
    -3.969683028665376e01,
    2.209460984245205e02,
    -2.759285104469687e02,
    1.383577518672690e02,
    -3.066479806614716e01,
    2.506628277459239e00,
)
_ACKLAM_B = (
    -5.447609879822406e01,
    1.615858368580409e02,
    -1.556989798598866e02,
    6.680131188771972e01,
    -1.328068155288572e01,
)
_ACKLAM_C = (
    -7.784894002430293e-03,
    -3.223964580411365e-01,
    -2.400758277161838e00,
    -2.549732539343734e00,
    4.374664141464968e00,
    2.938163982698783e00,
)
_ACKLAM_D = (
    7.784695709041462e-03,
    3.224671290700398e-01,
    2.445134137142996e00,
    3.754408661907416e00,
)
_ACKLAM_P_LOW = 0.02425
_ACKLAM_P_HIGH = 1.0 - _ACKLAM_P_LOW


def _inv_norm_cdf(p: float) -> float:
    """Inverse standard normal CDF (quantile function) via Acklam's algorithm.

    ``p`` must be strictly inside ``(0, 1)``; the algorithm's central and
    tail branches are selected by ``_ACKLAM_P_LOW``/``_ACKLAM_P_HIGH``, but
    it is valid (accurate to <= 1.15e-9 relative error) across the whole
    open interval.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in the open interval (0, 1), got {p!r}")

    if p < _ACKLAM_P_LOW:
        q = math.sqrt(-2.0 * math.log(p))
        c = _ACKLAM_C
        d = _ACKLAM_D
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= _ACKLAM_P_HIGH:
        q = p - 0.5
        r = q * q
        a = _ACKLAM_A
        b = _ACKLAM_B
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    c = _ACKLAM_C
    d = _ACKLAM_D
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


# ---------------------------------------------------------------------------
# BSM core
# ---------------------------------------------------------------------------


def _intrinsic(spot: float, strike: float, kind: OptionKind) -> float:
    """Undiscounted intrinsic value: max(S-K, 0) call, max(K-S, 0) put."""
    if kind is OptionKind.CALL:
        return max(spot - strike, 0.0)
    return max(strike - spot, 0.0)


def _moneyness_step(reference: float, strike: float, kind: OptionKind) -> float:
    """Deterministic in-the-money indicator: 1.0 ITM, 0.0 OTM, 0.5 exactly
    at the money. Used for the t=0 / vol=0 degenerate cases of both
    ``bs_delta`` and ``prob_itm``, where there is no remaining randomness
    and the BSM formulas collapse to a step function of moneyness. The 0.5
    value at the exact boundary is the symmetric limit of N(d1)/N(d2) as
    t or vol shrinks to 0 with the reference sitting exactly on the strike.
    """
    itm = reference > strike if kind is OptionKind.CALL else reference < strike
    if itm:
        return 1.0
    otm = reference < strike if kind is OptionKind.CALL else reference > strike
    if otm:
        return 0.0
    return 0.5


def _d1_d2(
    spot: float, strike: float, t_years: float, vol: float, rate: float
) -> tuple[float, float]:
    """The BSM d1, d2 terms. Requires ``spot > 0``, ``strike > 0``,
    ``t_years > 0``, ``vol > 0`` (callers branch around the degenerate
    t=0/vol<=0 cases before reaching this)."""
    sqrt_t = math.sqrt(t_years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * vol * vol) * t_years) / (vol * sqrt_t)
    d2 = d1 - vol * sqrt_t
    return d1, d2


def bs_price(
    spot: float, strike: float, t_years: float, vol: float, rate: float, kind: OptionKind
) -> float:
    """European Black-Scholes-Merton price (q=0; see module docstring).

    Edge cases (never NaN for spot > 0, strike > 0):

    - ``t_years <= 0``: expired/at-expiry — returns undiscounted intrinsic
      value, ``max(S-K, 0)`` for a call, ``max(K-S, 0)`` for a put.
    - ``vol <= 0``: no remaining randomness — returns the discounted
      intrinsic value of the forward, ``max(S - K*e^(-rt), 0)`` for a call
      and ``max(K*e^(-rt) - S, 0)`` for a put. This is exactly the limit
      of the BSM formula as vol -> 0+ (N(d1), N(d2) collapse to a step
      function of whether the forward is above or below the strike).
    """
    if t_years <= 0.0:
        return _intrinsic(spot, strike, kind)
    if vol <= 0.0:
        discount = math.exp(-rate * t_years)
        return _intrinsic(spot, strike * discount, kind)

    d1, d2 = _d1_d2(spot, strike, t_years, vol, rate)
    discount = math.exp(-rate * t_years)
    if kind is OptionKind.CALL:
        return spot * _norm_cdf(d1) - strike * discount * _norm_cdf(d2)
    return strike * discount * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta(
    spot: float, strike: float, t_years: float, vol: float, rate: float, kind: OptionKind
) -> float:
    """BSM delta: dPrice/dSpot. Call delta in [0, 1], put delta in [-1, 0].

    Edge cases (never NaN for spot > 0, strike > 0):

    - ``t_years <= 0``: delta collapses to the ITM step function of spot
      vs. strike (1.0/0.0 for calls, -1.0/0.0 for puts). At-the-money
      convention: exactly ``spot == strike`` returns the symmetric
      midpoint of the step, 0.5 for calls / -0.5 for puts.
    - ``vol <= 0``: same step function, evaluated against the forward
      price ``spot * e^(rt)`` instead of spot (the deterministic
      terminal value when there is no remaining randomness), with the
      same at-the-money midpoint convention.
    """
    if t_years <= 0.0:
        step = _moneyness_step(spot, strike, kind)
        return step if kind is OptionKind.CALL else -step
    if vol <= 0.0:
        forward = spot * math.exp(rate * t_years)
        step = _moneyness_step(forward, strike, kind)
        return step if kind is OptionKind.CALL else -step

    d1, _ = _d1_d2(spot, strike, t_years, vol, rate)
    return _norm_cdf(d1) if kind is OptionKind.CALL else _norm_cdf(d1) - 1.0


def prob_itm(
    spot: float, strike: float, t_years: float, vol: float, rate: float, kind: OptionKind
) -> float:
    """Risk-neutral model probability of finishing in-the-money at expiry.

    This is P(S_T > K) = N(d2) for a call and P(S_T < K) = N(-d2) for a
    put, under the BSM risk-neutral measure — it is used downstream as the
    assignment-probability proxy, but it is NOT a real-world probability
    (the true-measure drift differs from the risk-free rate) and it is NOT
    a market-implied probability (no market chain is observed). Any report
    surfacing this number must label it "model P(ITM at expiry)", not
    "assignment probability" unqualified (see module docstring).

    Edge cases mirror ``bs_delta``: ``t_years <= 0`` or ``vol <= 0``
    collapse to the deterministic ITM step function (0.5 exactly at the
    money), evaluated against the forward when only vol vanishes.
    """
    if t_years <= 0.0:
        return _moneyness_step(spot, strike, kind)
    if vol <= 0.0:
        forward = spot * math.exp(rate * t_years)
        return _moneyness_step(forward, strike, kind)

    _, d2 = _d1_d2(spot, strike, t_years, vol, rate)
    return _norm_cdf(d2) if kind is OptionKind.CALL else _norm_cdf(-d2)


def strike_for_delta(
    spot: float, target_delta: float, t_years: float, vol: float, rate: float, kind: OptionKind
) -> float:
    """Inverse of ``bs_delta``: the strike whose BSM delta equals ``target_delta``.

    ``target_delta`` must be in the open interval ``(0, 1)`` for calls and
    ``(-1, 0)`` for puts, and ``t_years > 0``, ``vol > 0`` (this inverts the
    non-degenerate branch of ``bs_delta`` only).

    Closed form, chosen over bisection because it is exact (up to the
    inverse-CDF approximation error, <= 1.15e-9) and non-iterative:
    delta = N(d1) for a call, N(d1) - 1 for a put, so
    ``d1 = N^-1(target_delta)`` (call) or ``N^-1(target_delta + 1)`` (put).
    Substituting into the d1 definition and solving for K gives::

        K = S * exp(-vol*sqrt(t)*d1 + (r + vol**2/2)*t)
    """
    n_target = target_delta if kind is OptionKind.CALL else target_delta + 1.0
    d1 = _inv_norm_cdf(n_target)
    return spot * math.exp(-vol * math.sqrt(t_years) * d1 + (rate + 0.5 * vol * vol) * t_years)


# ---------------------------------------------------------------------------
# Causal volatility proxy
# ---------------------------------------------------------------------------


def realized_vol(
    close: pd.Series, window: int = 21, method: Literal["rolling", "ewma"] = "ewma"
) -> pd.Series:
    """Annualized realized volatility of daily log returns, strictly causal.

    The value at row ``t`` depends only on log returns at or before ``t``
    (either a trailing ``window``-bar rolling std, or an EWMA std with
    ``span=window``) — identical causality discipline to
    ``funnel.strategies.base``: truncating the future never changes a past
    value. Both methods use ``min_periods=window``, so the first ``window``
    rows are NaN (warmup); annualization multiplies the daily std by
    ``sqrt(252)``.
    """
    log_returns = np.log(close / close.shift(1))
    if method == "rolling":
        daily_vol = log_returns.rolling(window, min_periods=window).std()
    else:
        daily_vol = log_returns.ewm(span=window, adjust=False, min_periods=window).std()
    return daily_vol * math.sqrt(_TRADING_DAYS_PER_YEAR)


@dataclass(slots=True, frozen=True)
class VolProxyConfig:
    """Parameters for the synthetic implied-vol proxy fed to the pricing core."""

    window: int = 21
    """Lookback (rolling) or span (EWMA) for the realized-vol estimate."""

    method: Literal["rolling", "ewma"] = "ewma"
    """Realized-vol estimator: trailing rolling std, or EWMA std."""

    risk_premium_multiplier: float = 1.1
    """Synthetic IV = realized vol * this multiplier. Implied vol typically
    trades above subsequently-realized vol (the volatility risk premium);
    this multiplier is a crude, configurable stand-in for that premium, not
    a fitted or market-observed quantity. Every downstream report using
    ``synthetic_iv`` must carry the synthetic-pricing caveat (see module
    docstring)."""

    floor: float = 0.05
    """Annualized vol floor. Realized vol can be arbitrarily close to zero
    on a quiet window; flooring it keeps every ``bs_price``/``bs_delta``
    call in this module's normal (vol > 0) branch instead of silently
    sliding into the degenerate discounted-intrinsic case."""


def synthetic_iv(close: pd.Series, config: VolProxyConfig) -> pd.Series:
    """Synthetic implied vol: ``realized_vol * risk_premium_multiplier``,
    floored at ``config.floor``. Causal (inherits ``realized_vol``'s
    causality); warmup rows stay NaN (``clip`` does not fill NaN)."""
    vol = realized_vol(close, window=config.window, method=config.method)
    return (vol * config.risk_premium_multiplier).clip(lower=config.floor)
