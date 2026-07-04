"""Profile and slider-value shapes (PRD §8): named profiles built from four 0-100 sliders."""

from dataclasses import dataclass

SLIDER_MIN = 0
SLIDER_MAX = 100


@dataclass(slots=True, frozen=True)
class SliderValues:
    """The four user-facing sliders, each an int in ``[0, 100]``.

    ``capital``: small book (0) -> institutional size (100). Soft-mapped —
    see ``funnel.profiles.mapping.ranking_weights``.

    ``risk_tolerance``: conservative (0) -> aggressive (100). Hard-mapped
    onto ``FunnelThresholds.max_oos_sharpe`` and ``.min_trades``.

    ``time_horizon``: intraday (0) -> multi-month (100). Soft-mapped onto a
    turnover preference; values below
    ``funnel.profiles.mapping.TIME_HORIZON_INTRADAY_UNSUPPORTED_BELOW`` are
    flagged as unsupported in v1 (EOD data only).

    ``drawdown_tolerance``: shallow (0) -> deep (100). Hard-mapped onto
    ``FunnelThresholds.max_dd_floor``.
    """

    capital: int
    risk_tolerance: int
    time_horizon: int
    drawdown_tolerance: int

    def __post_init__(self) -> None:
        for field_name in ("capital", "risk_tolerance", "time_horizon", "drawdown_tolerance"):
            value = getattr(self, field_name)
            if not (SLIDER_MIN <= value <= SLIDER_MAX):
                raise ValueError(f"{field_name}={value} out of range [{SLIDER_MIN}, {SLIDER_MAX}]")


@dataclass(slots=True, frozen=True)
class Profile:
    """A named, saveable profile: slider values plus identity/provenance."""

    name: str
    sliders: SliderValues
    created_at: str
    """ISO date string, e.g. ``"2026-07-03"``."""

    preset: bool = False
    """``True`` for the shipped presets; presets are never deleted or
    overwritten by ``funnel.profiles.store``."""


PRESETS: tuple[Profile, ...] = (
    Profile(
        name="Retirement Core",
        sliders=SliderValues(capital=60, risk_tolerance=20, time_horizon=90, drawdown_tolerance=25),
        created_at="2026-07-03",
        preset=True,
    ),
    Profile(
        name="Swing Sandbox",
        sliders=SliderValues(capital=25, risk_tolerance=65, time_horizon=35, drawdown_tolerance=60),
        created_at="2026-07-03",
        preset=True,
    ),
)
