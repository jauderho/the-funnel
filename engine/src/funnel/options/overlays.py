"""Defined-risk options overlay structures simulated on the daily grid.

Builds on ``funnel.options.pricing`` (V2-M1): every leg is priced with
``bs_price`` under the same q=0 / synthetic-IV ground rules documented
there. This module adds the four overlay structures named in PLAN.md
("v2 — Options Overlay Module"): covered calls, cash-secured puts, credit
vertical spreads, and LEAPS (stock-substitute long calls).

STRUCTURAL DEFINED-RISK ENFORCEMENT (PRD §2, hard constraint)
--------------------------------------------------------------------------
Every structure here is provably loss-bounded by construction:

- **Covered call**: the short call is fully collateralized by the owned
  shares (100 * contracts). The short leg can never create a loss beyond
  the stock's own decline to zero — identical floor to buy-and-hold.
- **Cash-secured put**: the short put is fully collateralized by a
  strike * 100 * contracts cash reserve. Maximum loss is that reserve
  (the worst case is being assigned stock at the strike that is then
  worth zero).
- **Credit vertical spread**: the long wing (further OTM, same
  expiration) caps the short leg's loss at (width - credit received);
  both legs open and close together, so the spread's width bounds the
  position's max loss at width * 100 * contracts.
- **LEAPS**: a long call has no obligation beyond the premium paid — loss
  is bounded at -100% of that premium (the position's own capital base).

``is_defined_risk`` checks the *inputs* that would break these guarantees
(a non-positive spread width collapses the vertical's cap; a
delta-selector with |delta| >= 1 or a non-positive DTE has no valid,
loss-bounded contract to construct) and is called both by
``OverlaySpec.__post_init__`` (so a bad config fails at construction time)
and again by ``simulate_overlay`` (so the engine never simulates an
unvalidated spec). Anything it rejects raises ``UndefinedRiskError`` —
per PRD §2 this is a hard rejection, never a warning.

CAPITAL BASE & DAILY RETURN CONVENTION (the subtle part — read carefully)
--------------------------------------------------------------------------
Every structure's daily return is P&L for the day divided by that
structure's own **capital base**, so Sharpe/drawdown are comparable across
structures and against buy-and-hold:

- Covered call: stock value at position entry (spot * 100 * contracts).
- Cash-secured put: strike * 100 * contracts (the cash reserve).
- Credit vertical: width * 100 * contracts (the max-loss collateral),
  width = spread_width_pct * spot-at-entry.
- LEAPS: premium paid at entry (100 * contracts * entry price).

The capital base is **re-established at every position transition**
(initial entry, scheduled roll, avoid-assignment roll, or expiry
settlement + re-entry), computed from that transition's own spot/vol. This
is a deliberate generalization of "at entry" to "at entry of the current
cycle": a multi-year daily series rolls dozens of times, and re-basing the
denominator each cycle is the only way to keep cycle P&L / capital-base
economically meaningful throughout (analogous to how buy-and-hold's daily
return implicitly re-bases every day using the prior close as the
denominator — overlays re-base once per roll cycle instead of once per
day). All costs incurred on a transition day (closing the outgoing
position and opening its replacement) are charged against the OUTGOING
position's capital base, since that is the capital economically deployed
through the close of that trading day; the new position's own capital
base takes over starting the following day's return.

MARK-TO-MODEL P&L AND ASSIGNMENT
--------------------------------------------------------------------------
Each open option leg is marked to model daily via ``bs_price`` at the
current spot/vol and the leg's own remaining time to its expiry
(``t_years <= 0`` at expiry collapses ``bs_price`` to intrinsic value with
no special-casing needed — see ``pricing.py``). A leg's dollar book value
is ``direction * bs_price(...) * 100 * contracts`` (``direction`` = +1.0
long, -1.0 short); a day's option P&L is simply the change in that book
value, which telescopes correctly over a position's life regardless of
path: total P&L over a cycle = premium collected/paid at entry minus the
settlement/close price. For covered calls and cash-secured puts, this
telescoping *already* reproduces the exact cash-flow economics of "shares
called away at strike and immediately repurchased at spot" (covered call)
or "shares put at strike and immediately sold at spot" (cash-secured put):
the short leg's negative intrinsic at settlement exactly offsets the
stock/cash leg's own move beyond the strike. No literal share round-trip
trade (and no separate cost for it) is modeled — that would double-count
economics already captured by the continuous stock/option marking. An
``AssignmentEvent`` is still recorded whenever a short leg (covered call,
cash-secured put, or a vertical's short leg) settles in-the-money, purely
as a reporting/labeling artifact; LEAPS has no short leg and never
produces one.

Costs (``OverlayCosts``) are applied uniformly on every open and every
close, whichever position and whichever trigger (scheduled roll,
avoid-assignment roll, or natural expiry) causes it — this matches the
literal reading of ``OverlayCosts.spread_haircut_pct``: haircut "on every
open AND close".

Every probability surfaced here (``prob_itm``) is the model P(ITM at
expiry) inherited from ``pricing.py`` — never a real-world assignment
probability. See that module's docstring for the full caveat.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import numpy as np
import pandas as pd

from funnel.backtest.metrics import TRADING_DAYS_PER_YEAR, max_drawdown, sharpe
from funnel.options.pricing import (
    OptionKind,
    VolProxyConfig,
    bs_price,
    prob_itm,
    strike_for_delta,
    synthetic_iv,
)

_SHARES_PER_CONTRACT = 100.0


class OverlayStructure(StrEnum):
    """The four defined-risk overlay structures this module simulates."""

    COVERED_CALL = "covered_call"
    CASH_SECURED_PUT = "cash_secured_put"
    VERTICAL_SPREAD = "vertical_spread"
    LEAPS = "leaps"


class UndefinedRiskError(ValueError):
    """Raised when an ``OverlaySpec`` cannot be proven loss-bounded.

    Per PRD §2 this is a hard rejection: a config that is not provably
    defined-risk is never simulated, and is never merely warned about.
    """


@dataclass(slots=True, frozen=True)
class StrikeSelector:
    """How a leg's strike is chosen at each position entry/roll."""

    mode: Literal["delta", "otm_pct"]
    """``"delta"``: strike inverted from a target BSM delta via
    ``strike_for_delta`` (sign convention enforced per structure — see
    ``OverlaySpec.__post_init__``). ``"otm_pct"``: strike = spot * (1 +/-
    value), direction chosen by option kind, not by the sign of ``value``.
    """

    value: float
    """Target delta (``0 < |value| < 1``, correctly signed for the
    structure) in ``"delta"`` mode, or a positive OTM fraction
    (``0 < value < 1``) in ``"otm_pct"`` mode."""


def _reference_kind(structure: OverlayStructure, kind: OptionKind) -> OptionKind:
    """The option kind whose delta sign convention governs the strike selector.

    Covered calls and LEAPS always transact a call; cash-secured puts
    always transact a put; a vertical spread's short (selector) leg is
    whichever ``kind`` the spec specifies.
    """
    if structure is OverlayStructure.CASH_SECURED_PUT:
        return OptionKind.PUT
    if structure is OverlayStructure.VERTICAL_SPREAD:
        return kind
    return OptionKind.CALL


def is_defined_risk(spec: OverlaySpec) -> tuple[bool, str]:
    """Structural bounded-loss check (PRD §2). See module docstring for the
    per-structure "why loss is bounded" rationale; this function checks the
    *inputs* that would break each rationale's guarantee.
    """
    if spec.dte_target <= 0:
        return False, (
            "dte_target must be positive: an expiring contract needs positive "
            "time to expiry for any loss-bound reasoning to apply"
        )
    if spec.strike_selector.mode == "delta" and not (0.0 < abs(spec.strike_selector.value) < 1.0):
        return False, (
            "delta-mode strike selector requires 0 < |delta| < 1; |delta| >= 1 "
            "has no valid inverse strike and cannot be provably bounded"
        )
    if spec.structure is OverlayStructure.VERTICAL_SPREAD and spec.spread_width_pct <= 0.0:
        return False, (
            "credit vertical spread requires spread_width_pct > 0: a "
            "zero/negative-width long wing does not cap the short leg's loss"
        )

    if spec.structure is OverlayStructure.COVERED_CALL:
        reason = (
            "covered call: the short call is fully collateralized by the "
            "owned shares; loss is bounded below by the stock's own decline "
            "to zero, with no additional obligation from the short leg"
        )
    elif spec.structure is OverlayStructure.CASH_SECURED_PUT:
        reason = (
            "cash-secured put: the short put is fully collateralized by the "
            "strike * 100 * contracts cash reserve; max loss is that reserve"
        )
    elif spec.structure is OverlayStructure.VERTICAL_SPREAD:
        reason = (
            "credit vertical spread: the long wing caps the short leg's loss "
            "at (width - credit received); loss cannot exceed width * 100 * "
            "contracts"
        )
    else:
        reason = (
            "LEAPS: a long call has no obligation beyond the premium paid; "
            "loss is bounded at -100% of the premium paid"
        )
    return True, reason


@dataclass(slots=True, frozen=True)
class OverlaySpec:
    """A fully-specified overlay configuration, validated at construction.

    ``dte_target`` is a **trading-day** approximation of calendar DTE (a
    documented modeling simplification: "30 DTE" here means 30 trading
    days out, not 30 calendar days) used directly as the expiry offset on
    the daily grid.
    """

    structure: OverlayStructure
    dte_target: int
    strike_selector: StrikeSelector
    roll_at_dte: int = 5
    """Roll/close when remaining trading days <= this. 0 means hold to
    (and settle at) expiry only — no early roll."""
    avoid_assignment: bool = False
    """Covered call / cash-secured put only: additionally roll early
    whenever the short leg's model P(ITM at expiry) exceeds
    ``assignment_prob_trigger``."""
    assignment_prob_trigger: float = 0.65
    spread_width_pct: float = 0.05
    """Vertical spread only: long-wing strike offset from the short leg,
    as a fraction of spot at entry."""
    kind: OptionKind = OptionKind.CALL
    """Vertical spread only: PUT builds a bull put spread (short put +
    further-OTM long put); CALL builds a bear call spread (short call +
    further-OTM long call). Ignored by the other three structures."""
    contracts: int = 1

    def __post_init__(self) -> None:
        ok, reason = is_defined_risk(self)
        if not ok:
            raise UndefinedRiskError(reason)

        if not (5 <= self.dte_target <= 756):
            raise ValueError(f"dte_target must be in [5, 756], got {self.dte_target!r}")
        if not (0 <= self.roll_at_dte < self.dte_target):
            raise ValueError(
                f"roll_at_dte must satisfy 0 <= roll_at_dte < dte_target, "
                f"got roll_at_dte={self.roll_at_dte!r}, dte_target={self.dte_target!r}"
            )
        if self.contracts < 1:
            raise ValueError(f"contracts must be >= 1, got {self.contracts!r}")
        if not (0.0 < self.assignment_prob_trigger < 1.0):
            raise ValueError(
                f"assignment_prob_trigger must be in (0, 1), got {self.assignment_prob_trigger!r}"
            )
        if self.avoid_assignment and self.structure not in (
            OverlayStructure.COVERED_CALL,
            OverlayStructure.CASH_SECURED_PUT,
        ):
            raise ValueError("avoid_assignment is only meaningful for covered call / CSP")

        if self.strike_selector.mode == "delta":
            ref_kind = _reference_kind(self.structure, self.kind)
            value = self.strike_selector.value
            if ref_kind is OptionKind.CALL and value <= 0.0:
                raise ValueError(f"call-referenced delta selector must be > 0, got {value!r}")
            if ref_kind is OptionKind.PUT and value >= 0.0:
                raise ValueError(f"put-referenced delta selector must be < 0, got {value!r}")
        elif not (0.0 < self.strike_selector.value < 1.0):
            raise ValueError(
                f"otm_pct strike selector value must be in (0, 1), "
                f"got {self.strike_selector.value!r}"
            )


@dataclass(slots=True, frozen=True)
class OverlayCosts:
    """Per-contract-leg transaction cost assumptions, applied on every
    open and every close (whichever position, whichever trigger)."""

    per_contract_commission: float = 0.65
    """Dollars per contract per leg, charged on every open and close."""

    spread_haircut_pct: float = 0.05
    """Fraction of a leg's model premium lost to bid/ask on every open and
    close — the synthetic stand-in for a real bid/ask spread."""


@dataclass(slots=True, frozen=True)
class AssignmentEvent:
    """A short leg (covered call, cash-secured put, or a vertical's short
    leg) settling in-the-money at its own expiry. Reporting/labeling only
    — see the module docstring for why no separate share round-trip is
    modeled."""

    date: pd.Timestamp
    structure: OverlayStructure
    strike: float
    spot: float
    moneyness: float
    """spot / strike at settlement (> 1.0 means the underlying finished
    above the strike)."""


@dataclass(slots=True, frozen=True)
class OverlayResult:
    """Daily-return outcome of one ``simulate_overlay`` run."""

    returns: pd.Series
    """Daily overlay returns on the structure's own (per-cycle) capital
    base. NaN before the first position enters (vol-proxy warmup);
    defined (possibly 0.0) every day thereafter."""

    underlying_returns: pd.Series
    """Buy-and-hold daily returns over the identical valid window as
    ``returns`` (NaN wherever ``returns`` is NaN), for honest comparison."""

    events: list[AssignmentEvent]
    n_rolls: int
    """Every position transition after the initial entry — scheduled
    rolls, avoid-assignment rolls, and expiry-triggered re-entries alike
    (the close-then-reopen mechanics are identical in all three cases)."""

    premium_collected_annualized: float
    """Sum, over every cycle, of that cycle's net premium collected as a
    fraction of that cycle's own capital base, annualized over the full
    backtest's elapsed years. 0.0 for LEAPS (a debit strategy — there is
    no premium collected) or if no position ever entered."""

    mean_prob_itm_at_entry: float
    """Mean, across cycles, of the model P(ITM at expiry) of the labeled
    leg (the short leg, or for LEAPS the long leg) at that cycle's own
    entry. 0.0 if no position ever entered."""

    upside_forgone: float
    """max(buy-and-hold total return - overlay total return, 0.0) over the
    shared valid window — the capped-upside cost of the overlay, never
    hidden as a negative number."""

    def summary(self) -> dict[str, float]:
        """Sharpe / max-drawdown for both the overlay and the underlying,
        reusing ``funnel.backtest.metrics`` so the numbers are computed
        identically to every other strategy in the funnel."""
        return {
            "overlay_sharpe": sharpe(self.returns),
            "overlay_max_drawdown": max_drawdown(self.returns),
            "underlying_sharpe": sharpe(self.underlying_returns),
            "underlying_max_drawdown": max_drawdown(self.underlying_returns),
        }


@dataclass(slots=True, frozen=True)
class _Leg:
    kind: OptionKind
    strike: float
    direction: float
    """+1.0 long, -1.0 short."""


@dataclass(slots=True)
class _Position:
    legs: tuple[_Leg, ...]
    capital_base: float
    entry_index: int
    expiry_index: int
    book_value_prev: float
    """Dollar mark-to-model value of ``legs`` only (excludes the stock/cash
    leg), as of the position's own entry (or the most recent day it was
    marked)."""
    open_cost: float
    """Dollars, charged against the return of the first day after entry."""
    premium_dollars: float


def _selector_strike(
    spot: float, selector: StrikeSelector, t_years: float, vol: float, rate: float, kind: OptionKind
) -> float:
    if selector.mode == "delta":
        return strike_for_delta(spot, selector.value, t_years, vol, rate, kind)
    pct = selector.value
    return spot * (1.0 + pct) if kind is OptionKind.CALL else spot * (1.0 - pct)


def _book_value_dollars(
    legs: tuple[_Leg, ...], spot: float, t_years: float, vol: float, rate: float, contracts: int
) -> float:
    return (
        _SHARES_PER_CONTRACT
        * contracts
        * sum(
            leg.direction * bs_price(spot, leg.strike, t_years, vol, rate, leg.kind) for leg in legs
        )
    )


def _leg_costs(
    legs: tuple[_Leg, ...],
    spot: float,
    t_years: float,
    vol: float,
    rate: float,
    contracts: int,
    costs: OverlayCosts,
) -> float:
    total = 0.0
    for leg in legs:
        price = bs_price(spot, leg.strike, t_years, vol, rate, leg.kind)
        total += costs.per_contract_commission * contracts
        total += costs.spread_haircut_pct * price * _SHARES_PER_CONTRACT * contracts
    return total


def _open_position(
    spec: OverlaySpec, spot: float, vol: float, rate: float, entry_index: int, costs: OverlayCosts
) -> _Position:
    t_years = spec.dte_target / TRADING_DAYS_PER_YEAR
    contracts = spec.contracts

    if spec.structure is OverlayStructure.COVERED_CALL:
        strike = _selector_strike(spot, spec.strike_selector, t_years, vol, rate, OptionKind.CALL)
        legs = (_Leg(OptionKind.CALL, strike, -1.0),)
        capital_base = spot * _SHARES_PER_CONTRACT * contracts
    elif spec.structure is OverlayStructure.CASH_SECURED_PUT:
        strike = _selector_strike(spot, spec.strike_selector, t_years, vol, rate, OptionKind.PUT)
        legs = (_Leg(OptionKind.PUT, strike, -1.0),)
        capital_base = strike * _SHARES_PER_CONTRACT * contracts
    elif spec.structure is OverlayStructure.VERTICAL_SPREAD:
        short_strike = _selector_strike(spot, spec.strike_selector, t_years, vol, rate, spec.kind)
        width_dollars = spec.spread_width_pct * spot
        if spec.kind is OptionKind.CALL:
            long_strike = short_strike + width_dollars
        else:
            long_strike = short_strike - width_dollars
        legs = (
            _Leg(spec.kind, short_strike, -1.0),
            _Leg(spec.kind, long_strike, 1.0),
        )
        capital_base = width_dollars * _SHARES_PER_CONTRACT * contracts
    else:  # LEAPS
        strike = _selector_strike(spot, spec.strike_selector, t_years, vol, rate, OptionKind.CALL)
        legs = (_Leg(OptionKind.CALL, strike, 1.0),)
        capital_base = 0.0  # set below from the leg's own premium

    book_value = _book_value_dollars(legs, spot, t_years, vol, rate, contracts)
    if spec.structure is OverlayStructure.LEAPS:
        capital_base = book_value
        premium_dollars = 0.0
    else:
        premium_dollars = -book_value

    open_cost = _leg_costs(legs, spot, t_years, vol, rate, contracts, costs)

    return _Position(
        legs=legs,
        capital_base=capital_base,
        entry_index=entry_index,
        expiry_index=entry_index + spec.dte_target,
        book_value_prev=book_value,
        open_cost=open_cost,
        premium_dollars=premium_dollars,
    )


def _extra_leg_pnl_dollars(
    structure: OverlayStructure,
    position: _Position,
    contracts: int,
    spot: float,
    spot_prev: float,
    rate: float,
) -> float:
    if structure is OverlayStructure.COVERED_CALL:
        return _SHARES_PER_CONTRACT * contracts * (spot - spot_prev)
    if structure is OverlayStructure.CASH_SECURED_PUT:
        return position.capital_base * (rate / TRADING_DAYS_PER_YEAR)
    return 0.0


def _short_leg(position: _Position) -> _Leg | None:
    for leg in position.legs:
        if leg.direction < 0.0:
            return leg
    return None


def _labeled_leg(position: _Position) -> _Leg:
    """The leg whose P(ITM) is reported: the short leg, or (LEAPS) the
    lone long leg."""
    return _short_leg(position) or position.legs[0]


def _check_assignment(
    structure: OverlayStructure, position: _Position, spot: float, date: pd.Timestamp
) -> AssignmentEvent | None:
    short = _short_leg(position)
    if short is None:
        return None
    itm = spot > short.strike if short.kind is OptionKind.CALL else spot < short.strike
    if not itm:
        return None
    return AssignmentEvent(
        date=date,
        structure=structure,
        strike=short.strike,
        spot=spot,
        moneyness=spot / short.strike,
    )


def simulate_overlay(
    df: pd.DataFrame,
    spec: OverlaySpec,
    vol_config: VolProxyConfig,
    costs: OverlayCosts,
    rate: float = 0.03,
) -> OverlayResult:
    """Simulate one overlay structure over ``df``'s daily grid.

    No look-ahead: every decision at day ``t`` (entry, roll, avoid-
    assignment check, settlement) uses only ``vol``/``close`` at or before
    ``t`` (``synthetic_iv`` is itself strictly causal — see
    ``pricing.py``). See the module docstring for the capital-base,
    mark-to-model, and cost conventions.
    """
    ok, reason = is_defined_risk(spec)
    if not ok:
        raise UndefinedRiskError(reason)

    close = df["close"]
    n = len(close)
    vol = synthetic_iv(close, vol_config)

    close_arr = close.to_numpy(dtype="float64")
    vol_arr = vol.to_numpy(dtype="float64")

    returns = pd.Series(np.nan, index=df.index, dtype="float64")
    position: _Position | None = None
    n_rolls = 0
    events: list[AssignmentEvent] = []
    cycle_premium_fractions: list[float] = []
    cycle_prob_itm_entries: list[float] = []

    def _enter(spot: float, vol_today: float, index: int) -> _Position:
        new_position = _open_position(spec, spot, vol_today, rate, index, costs)
        cycle_premium_fractions.append(new_position.premium_dollars / new_position.capital_base)
        labeled = _labeled_leg(new_position)
        t_years = spec.dte_target / TRADING_DAYS_PER_YEAR
        cycle_prob_itm_entries.append(
            prob_itm(spot, labeled.strike, t_years, vol_today, rate, labeled.kind)
        )
        return new_position

    for i in range(n):
        spot_i = close_arr[i]
        vol_i = vol_arr[i]

        if position is None:
            if np.isnan(vol_i):
                continue
            position = _enter(spot_i, vol_i, i)
            continue

        remaining_after_today = position.expiry_index - i
        t_years_end = max(remaining_after_today, 0) / TRADING_DAYS_PER_YEAR
        mtm_end = _book_value_dollars(
            position.legs, spot_i, t_years_end, vol_i, rate, spec.contracts
        )

        pnl = mtm_end - position.book_value_prev
        pnl += _extra_leg_pnl_dollars(
            spec.structure, position, spec.contracts, spot_i, close_arr[i - 1], rate
        )

        day_cost = 0.0
        if i - 1 == position.entry_index:
            day_cost += position.open_cost

        should_settle = remaining_after_today <= 0
        should_roll_early = False
        if not should_settle:
            if remaining_after_today <= spec.roll_at_dte:
                should_roll_early = True
            elif spec.avoid_assignment:
                short = _short_leg(position)
                assert short is not None  # enforced by OverlaySpec.__post_init__
                short_prob_itm = prob_itm(
                    spot_i, short.strike, t_years_end, vol_i, rate, short.kind
                )
                should_roll_early = short_prob_itm > spec.assignment_prob_trigger
        should_close = should_settle or should_roll_early

        if should_close:
            day_cost += _leg_costs(
                position.legs, spot_i, t_years_end, vol_i, rate, spec.contracts, costs
            )
            if should_settle:
                event = _check_assignment(spec.structure, position, spot_i, df.index[i])
                if event is not None:
                    events.append(event)

        returns.iloc[i] = (pnl - day_cost) / position.capital_base

        if should_close:
            n_rolls += 1
            position = _enter(spot_i, vol_i, i)
        else:
            position.book_value_prev = mtm_end

    valid = returns.notna()
    underlying_returns = close.pct_change().where(valid)

    n_valid_days = int(valid.sum())
    if n_valid_days > 0:
        bh_total = float((1.0 + underlying_returns[valid]).prod() - 1.0)
        ov_total = float((1.0 + returns[valid]).prod() - 1.0)
        upside_forgone = max(bh_total - ov_total, 0.0)
        years = n_valid_days / TRADING_DAYS_PER_YEAR
        premium_collected_annualized = float(sum(cycle_premium_fractions)) / years
    else:
        upside_forgone = 0.0
        premium_collected_annualized = 0.0

    mean_prob_itm_at_entry = (
        float(np.mean(cycle_prob_itm_entries)) if cycle_prob_itm_entries else 0.0
    )

    return OverlayResult(
        returns=returns,
        underlying_returns=underlying_returns,
        events=events,
        n_rolls=n_rolls,
        premium_collected_annualized=premium_collected_annualized,
        mean_prob_itm_at_entry=mean_prob_itm_at_entry,
        upside_forgone=upside_forgone,
    )
