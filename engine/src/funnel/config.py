"""Configuration dataclasses for the six-filter survival funnel.

These are the single source of truth for thresholds consumed by both the
backtest engine and the reporting layer, so that any report of "what passed"
is always traceable to the exact numbers that were applied.
"""

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class FunnelThresholds:
    """Pass/fail thresholds for the six-filter survival funnel.

    Each field gates one filter in the funnel; a strategy config must clear
    all of them (per walk-forward window, out-of-sample) to be reported as a
    survivor.
    """

    max_dd_floor: float = -0.35
    """Filter 1 — maximum drawdown floor. OOS max drawdown must be >= this
    (less negative) to pass; deeper drawdowns are rejected regardless of
    return quality."""

    min_oos_sharpe: float = 0.5
    """Filter 2 — minimum out-of-sample Sharpe ratio required to pass."""

    max_oos_sharpe: float = 2.5
    """Filter 3 — maximum out-of-sample Sharpe ratio. Implausibly high OOS
    Sharpe is treated as a red flag (overfitting, data error) rather than a
    win, and is rejected."""

    max_oos_is_ratio: float = 1.3
    """Filter 4 — maximum ratio of OOS Sharpe to in-sample Sharpe. Ratios
    much greater than 1 indicate an unstable or lucky OOS window rather than
    a genuinely robust edge."""

    min_trades: int = 30
    """Filter 5 — minimum number of OOS trades required for the sample to be
    statistically meaningful."""

    require_positive_is_sharpe: bool = True
    """Filter 6 — the in-sample Sharpe ratio must be positive. A strategy
    that could not even fit its own training window is rejected outright."""


@dataclass(slots=True, frozen=True)
class WalkForwardConfig:
    """Parameters controlling the walk-forward validation split."""

    n_windows: int = 5
    """Number of rolling walk-forward windows to stitch together for the
    out-of-sample (OOS) series."""

    is_fraction: float = 0.7
    """Fraction of each window allocated to in-sample (IS) fitting; the
    remainder (1 - is_fraction) is held out as OOS."""


@dataclass(slots=True, frozen=True)
class CostModel:
    """Per-trade transaction cost assumptions, applied on every side."""

    default_bps_per_side: float = 1.0
    """Default transaction cost, in basis points, charged per side of a
    trade for non-crypto instruments."""

    crypto_bps_per_side: float = 5.0
    """Transaction cost, in basis points, charged per side of a trade for
    crypto instruments (wider spreads / higher venue fees than equities)."""
