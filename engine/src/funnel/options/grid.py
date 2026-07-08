"""Overlay configuration grid: enumerates every (structure, param-set) pair.

Mirrors ``funnel.strategies.grid``'s shape: a single flat, unique-named list
of concrete overlay configs is built once by ``build_overlay_grid`` and swept
by ``funnel.options.sweep`` against every symbol in scope. Every ``OverlaySpec``
built here passes through its own ``__post_init__`` defined-risk validation
(``funnel.options.overlays.is_defined_risk``) — a config that is not provably
loss-bounded raises ``UndefinedRiskError`` at construction time, not later.
"""

from collections import Counter
from dataclasses import dataclass

from funnel.options.overlays import OverlaySpec, OverlayStructure, StrikeSelector
from funnel.options.pricing import OptionKind

_COVERED_CALL_DELTAS: tuple[float, ...] = (0.15, 0.25, 0.35)
_COVERED_CALL_DTES: tuple[int, ...] = (21, 45)
_CSP_DELTAS: tuple[float, ...] = (-0.15, -0.25, -0.35)
_CSP_DTES: tuple[int, ...] = (21, 45)
_ASSIGNMENT_PROB_TRIGGER = 0.65

_VERTICAL_SHORT_DELTA_MAGNITUDES: tuple[float, ...] = (0.20, 0.30)
_VERTICAL_DTES: tuple[int, ...] = (30, 45)
_VERTICAL_WIDTH_PCT = 0.05
_VERTICAL_KINDS: tuple[tuple[OptionKind, str], ...] = (
    (OptionKind.PUT, "bull_put"),
    (OptionKind.CALL, "bear_call"),
)

_LEAPS_DELTAS: tuple[float, ...] = (0.70, 0.80)
_LEAPS_DTES: tuple[int, ...] = (252, 504)

_HOLD_TO_EXPIRY_DELTA_MAGNITUDE = 0.25
"""Delta magnitude used by the hold-to-expiry (``roll_at_dte=0``) variants
below. Every other config in this grid rolls at the default ``roll_at_dte=5``
(before ``simulate_overlay``'s ``remaining_after_today <= spec.roll_at_dte``
check can ever reach true expiry), which makes the settlement/assignment
path (``_check_assignment`` in ``options/overlays.py``) structurally
unreachable for the whole grid. These variants set ``roll_at_dte=0`` so a
position is only closed by a scheduled expiry (or, for the ``avoid=True``
pair, an early P(ITM)-triggered roll) — exercising the assignment path a
real report can actually observe."""


@dataclass(slots=True, frozen=True)
class OverlayConfig:
    """One concrete, runnable overlay: a validated spec bound to a unique name."""

    name: str
    """Unique identifier, e.g. ``"covered_call_d25_dte45_roll5_avoid"``."""

    spec: OverlaySpec

    description: str
    """Human-readable one-line summary of the structure and its parameters."""


def _covered_call_configs() -> list[OverlayConfig]:
    configs: list[OverlayConfig] = []
    for delta in _COVERED_CALL_DELTAS:
        for dte in _COVERED_CALL_DTES:
            for avoid in (True, False):
                spec = OverlaySpec(
                    structure=OverlayStructure.COVERED_CALL,
                    dte_target=dte,
                    strike_selector=StrikeSelector(mode="delta", value=delta),
                    avoid_assignment=avoid,
                    assignment_prob_trigger=_ASSIGNMENT_PROB_TRIGGER,
                )
                tag = "avoid" if avoid else "noavoid"
                name = f"covered_call_d{int(delta * 100)}_dte{dte}_roll{spec.roll_at_dte}_{tag}"
                description = (
                    f"Covered call: short {delta:.2f}-delta call, {dte} DTE, "
                    f"{'rolls early to avoid assignment' if avoid else 'no avoid-assignment roll'}"
                )
                configs.append(OverlayConfig(name=name, spec=spec, description=description))
    return configs


def _covered_call_hold_to_expiry_configs() -> list[OverlayConfig]:
    """Hold-to-expiry variants (``roll_at_dte=0``) so settlement/assignment
    is actually reachable — see ``_HOLD_TO_EXPIRY_DELTA_MAGNITUDE``."""
    configs: list[OverlayConfig] = []
    delta = _HOLD_TO_EXPIRY_DELTA_MAGNITUDE
    for dte in _COVERED_CALL_DTES:
        for avoid in (True, False):
            spec = OverlaySpec(
                structure=OverlayStructure.COVERED_CALL,
                dte_target=dte,
                strike_selector=StrikeSelector(mode="delta", value=delta),
                roll_at_dte=0,
                avoid_assignment=avoid,
                assignment_prob_trigger=_ASSIGNMENT_PROB_TRIGGER,
            )
            tag = "avoid" if avoid else "noavoid"
            name = f"covered_call_d{int(delta * 100)}_dte{dte}_roll0_{tag}_hold"
            description = (
                f"Covered call: short {delta:.2f}-delta call, {dte} DTE, held to "
                f"expiry (roll_at_dte=0){' with P(ITM)-triggered early roll' if avoid else ''}"
            )
            configs.append(OverlayConfig(name=name, spec=spec, description=description))
    return configs


def _cash_secured_put_configs() -> list[OverlayConfig]:
    configs: list[OverlayConfig] = []
    for delta in _CSP_DELTAS:
        for dte in _CSP_DTES:
            for avoid in (True, False):
                spec = OverlaySpec(
                    structure=OverlayStructure.CASH_SECURED_PUT,
                    dte_target=dte,
                    strike_selector=StrikeSelector(mode="delta", value=delta),
                    avoid_assignment=avoid,
                    assignment_prob_trigger=_ASSIGNMENT_PROB_TRIGGER,
                )
                tag = "avoid" if avoid else "noavoid"
                name = (
                    f"cash_secured_put_d{int(abs(delta) * 100)}_dte{dte}_"
                    f"roll{spec.roll_at_dte}_{tag}"
                )
                description = (
                    f"Cash-secured put: short {delta:.2f}-delta put, {dte} DTE, "
                    f"{'rolls early to avoid assignment' if avoid else 'no avoid-assignment roll'}"
                )
                configs.append(OverlayConfig(name=name, spec=spec, description=description))
    return configs


def _cash_secured_put_hold_to_expiry_configs() -> list[OverlayConfig]:
    """Hold-to-expiry mirror of ``_covered_call_hold_to_expiry_configs``."""
    configs: list[OverlayConfig] = []
    delta = -_HOLD_TO_EXPIRY_DELTA_MAGNITUDE
    for dte in _CSP_DTES:
        for avoid in (True, False):
            spec = OverlaySpec(
                structure=OverlayStructure.CASH_SECURED_PUT,
                dte_target=dte,
                strike_selector=StrikeSelector(mode="delta", value=delta),
                roll_at_dte=0,
                avoid_assignment=avoid,
                assignment_prob_trigger=_ASSIGNMENT_PROB_TRIGGER,
            )
            tag = "avoid" if avoid else "noavoid"
            name = f"cash_secured_put_d{int(abs(delta) * 100)}_dte{dte}_roll0_{tag}_hold"
            description = (
                f"Cash-secured put: short {delta:.2f}-delta put, {dte} DTE, held to "
                f"expiry (roll_at_dte=0){' with P(ITM)-triggered early roll' if avoid else ''}"
            )
            configs.append(OverlayConfig(name=name, spec=spec, description=description))
    return configs


def _vertical_spread_configs() -> list[OverlayConfig]:
    configs: list[OverlayConfig] = []
    for kind, label in _VERTICAL_KINDS:
        for magnitude in _VERTICAL_SHORT_DELTA_MAGNITUDES:
            # Strike-selector sign convention (OverlaySpec.__post_init__): a
            # vertical's short leg is referenced against its own `kind`, so a
            # bull put spread (short put) needs a negative delta while a bear
            # call spread (short call) needs a positive one.
            delta = -magnitude if kind is OptionKind.PUT else magnitude
            for dte in _VERTICAL_DTES:
                spec = OverlaySpec(
                    structure=OverlayStructure.VERTICAL_SPREAD,
                    dte_target=dte,
                    strike_selector=StrikeSelector(mode="delta", value=delta),
                    spread_width_pct=_VERTICAL_WIDTH_PCT,
                    kind=kind,
                )
                name = (
                    f"vertical_{label}_d{int(magnitude * 100)}_"
                    f"w{int(_VERTICAL_WIDTH_PCT * 100)}_dte{dte}"
                )
                description = (
                    f"Vertical spread ({label.replace('_', ' ')}): short "
                    f"{magnitude:.2f}-delta {kind.value}, "
                    f"{int(_VERTICAL_WIDTH_PCT * 100)}% width, {dte} DTE"
                )
                configs.append(OverlayConfig(name=name, spec=spec, description=description))
    return configs


def _vertical_spread_hold_to_expiry_configs() -> list[OverlayConfig]:
    """Hold-to-expiry variant (``roll_at_dte=0``), one per kind, at the
    canonical hold-to-expiry delta magnitude and a fixed 30 DTE / 5% width."""
    configs: list[OverlayConfig] = []
    magnitude = _HOLD_TO_EXPIRY_DELTA_MAGNITUDE
    dte = _VERTICAL_DTES[0]
    for kind, label in _VERTICAL_KINDS:
        delta = -magnitude if kind is OptionKind.PUT else magnitude
        spec = OverlaySpec(
            structure=OverlayStructure.VERTICAL_SPREAD,
            dte_target=dte,
            strike_selector=StrikeSelector(mode="delta", value=delta),
            spread_width_pct=_VERTICAL_WIDTH_PCT,
            kind=kind,
            roll_at_dte=0,
        )
        name = (
            f"vertical_{label}_d{int(magnitude * 100)}_"
            f"w{int(_VERTICAL_WIDTH_PCT * 100)}_dte{dte}_roll0_hold"
        )
        description = (
            f"Vertical spread ({label.replace('_', ' ')}): short "
            f"{magnitude:.2f}-delta {kind.value}, {int(_VERTICAL_WIDTH_PCT * 100)}% width, "
            f"{dte} DTE, held to expiry (roll_at_dte=0)"
        )
        configs.append(OverlayConfig(name=name, spec=spec, description=description))
    return configs


def _leaps_configs() -> list[OverlayConfig]:
    configs: list[OverlayConfig] = []
    for delta in _LEAPS_DELTAS:
        for dte in _LEAPS_DTES:
            spec = OverlaySpec(
                structure=OverlayStructure.LEAPS,
                dte_target=dte,
                strike_selector=StrikeSelector(mode="delta", value=delta),
            )
            name = f"leaps_d{int(delta * 100)}_dte{dte}"
            description = f"LEAPS: long {delta:.2f}-delta call, {dte} DTE"
            configs.append(OverlayConfig(name=name, spec=spec, description=description))
    return configs


def build_overlay_grid() -> list[OverlayConfig]:
    """Enumerate every (structure, parameter set) overlay config to be swept.

    Covers all four ``OverlayStructure`` values named in PLAN.md: covered
    calls and cash-secured puts (delta target x DTE x avoid-assignment),
    vertical spreads (both bull-put and bear-call, short-leg delta x DTE at
    a fixed 5% width), and LEAPS (delta x DTE).

    Also includes hold-to-expiry (``roll_at_dte=0``) variants for
    covered_call, cash_secured_put, and vertical_spread (LEAPS has no short
    leg, so assignment is not applicable there) — see
    ``_HOLD_TO_EXPIRY_DELTA_MAGNITUDE``'s docstring for why these are
    necessary: every other config's default ``roll_at_dte=5`` closes the
    position before ``simulate_overlay`` can ever reach true expiry, making
    the settlement/assignment path structurally unreachable without them.
    """
    configs = (
        _covered_call_configs()
        + _covered_call_hold_to_expiry_configs()
        + _cash_secured_put_configs()
        + _cash_secured_put_hold_to_expiry_configs()
        + _vertical_spread_configs()
        + _vertical_spread_hold_to_expiry_configs()
        + _leaps_configs()
    )
    _assert_unique_names(configs)
    return configs


def _assert_unique_names(configs: list[OverlayConfig]) -> None:
    seen: set[str] = set()
    for config in configs:
        if config.name in seen:
            raise ValueError(f"duplicate overlay config name: {config.name}")
        seen.add(config.name)


def summarize_overlay_grid(configs: list[OverlayConfig] | None = None) -> dict[str, int]:
    """Per-structure config counts, e.g. ``{"covered_call": 12, "leaps": 4, ...}``."""
    if configs is None:
        configs = build_overlay_grid()
    counts = Counter(config.spec.structure.value for config in configs)
    return dict(sorted(counts.items()))
