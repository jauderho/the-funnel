"""The hybrid slider-to-funnel mapping (PRD §8, PLAN.md decision 3).

Two sliders (``drawdown_tolerance``, ``risk_tolerance``) are HARD-mapped:
they move actual ``FunnelThresholds`` fields, so a strategy that fails the
profile-adjusted funnel is excluded outright — the same way any strategy
fails the base funnel. Two sliders (``capital``, ``time_horizon``) are
SOFT-mapped: they only produce ``RankingWeights`` used to reorder surviving
strategies. Soft scoring (``score_rows``) never removes a row; it only
changes sort order. Both mappings are linear, deterministic, and monotone
in the slider value, and both are exposed via ``explain_mapping`` so the UI
can show the user exactly what their sliders did.
"""

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from funnel.config import FunnelThresholds
from funnel.data.universe import AssetClass
from funnel.profiles.models import SliderValues

# --- Hard mapping endpoints (drawdown_tolerance -> max_dd_floor) -----------

DD_FLOOR_AT_MIN_TOLERANCE = -0.15
"""``drawdown_tolerance == 0`` (shallowest): tightest max-DD floor."""

DD_FLOOR_AT_MAX_TOLERANCE = -0.50
"""``drawdown_tolerance == 100`` (deepest): loosest max-DD floor."""

# --- Hard mapping endpoints (risk_tolerance -> max_oos_sharpe, min_trades) -

MAX_OOS_SHARPE_AT_MIN_RISK = 2.0
"""``risk_tolerance == 0`` (conservative): tightest too-good-to-be-true ceiling."""

MAX_OOS_SHARPE_AT_MAX_RISK = 4.0
"""``risk_tolerance == 100`` (aggressive): relaxed too-good-to-be-true ceiling."""

MIN_TRADES_AT_MIN_RISK = 40
"""``risk_tolerance == 0`` (conservative): demands more statistical evidence."""

MIN_TRADES_AT_MAX_RISK = 20
"""``risk_tolerance == 100`` (aggressive): tolerates a thinner trade sample."""

# --- Soft mapping constants -------------------------------------------------

NICHE_PENALTY_AT_MAX_CAPITAL = 1.0
"""``capital == 100`` (institutional): full niche penalty applied to crypto
rows in ``score_rows`` — big books can't practically trade niche/thin
markets. ``capital == 0`` applies no penalty (a small book can trade
niches)."""

CRYPTO_PENALTY_WEIGHT = 0.5
"""Coefficient applied to ``niche_penalty`` when subtracting from a crypto
row's soft score (see ``score_rows``)."""

TURNOVER_ZSCORE_CLIP = 2.0
"""Clip bound for the trade-count z-score used in the turnover-preference term."""

TURNOVER_PREFERENCE_WEIGHT = 0.1
"""Coefficient applied to ``turnover_preference * clipped_zscore`` in ``score_rows``."""

TIME_HORIZON_INTRADAY_UNSUPPORTED_BELOW = 20
"""Below this ``time_horizon`` value, the slider is asking for a granularity
(intraday) that v1's EOD-only data source cannot provide."""


def _lerp(value: int, at_min: float, at_max: float) -> float:
    """Linear interpolation of ``value`` (in ``[0, 100]``) between ``at_min`` and ``at_max``."""
    fraction = value / 100.0
    return at_min + fraction * (at_max - at_min)


def thresholds_for(sliders: SliderValues, base: FunnelThresholds) -> FunnelThresholds:
    """Hard-map ``drawdown_tolerance`` and ``risk_tolerance`` onto ``base``.

    All other ``FunnelThresholds`` fields are inherited from ``base``
    unchanged. Every interpolation is linear and monotone in its slider:
    increasing ``drawdown_tolerance`` strictly deepens (decreases)
    ``max_dd_floor``; increasing ``risk_tolerance`` strictly increases
    ``max_oos_sharpe`` and strictly decreases ``min_trades``.
    """
    max_dd_floor = _lerp(
        sliders.drawdown_tolerance, DD_FLOOR_AT_MIN_TOLERANCE, DD_FLOOR_AT_MAX_TOLERANCE
    )
    max_oos_sharpe = _lerp(
        sliders.risk_tolerance, MAX_OOS_SHARPE_AT_MIN_RISK, MAX_OOS_SHARPE_AT_MAX_RISK
    )
    min_trades = round(
        _lerp(sliders.risk_tolerance, MIN_TRADES_AT_MIN_RISK, MIN_TRADES_AT_MAX_RISK)
    )
    return replace(
        base,
        max_dd_floor=max_dd_floor,
        max_oos_sharpe=max_oos_sharpe,
        min_trades=min_trades,
    )


@dataclass(slots=True, frozen=True)
class RankingWeights:
    """Soft re-ranking weights derived from ``capital`` and ``time_horizon``.

    Used only by ``score_rows`` to reorder surviving strategies; never used
    to exclude a row (exclusion is the hard-mapped funnel's job alone).
    """

    niche_penalty: float
    """In ``[0, 1]``. Applied (scaled by ``CRYPTO_PENALTY_WEIGHT``) against
    crypto-asset rows' soft score. 0 = small book, no penalty; 1 =
    institutional, full penalty."""

    turnover_preference: float
    """Signed, in ``[-1, 1]``. Positive prefers higher OOS trade counts
    (intraday-ish, more reps); negative prefers lower trade counts
    (multi-month, fewer reps); 0 at the midpoint slider value."""


def ranking_weights(sliders: SliderValues) -> RankingWeights:
    """Soft-map ``capital`` and ``time_horizon`` into ``RankingWeights``."""
    niche_penalty = _lerp(sliders.capital, 0.0, NICHE_PENALTY_AT_MAX_CAPITAL)
    turnover_preference = (50 - sliders.time_horizon) / 50.0
    return RankingWeights(niche_penalty=niche_penalty, turnover_preference=turnover_preference)


def _trade_count_zscore(trade_counts: pd.Series) -> pd.Series:
    """Z-score of ``trade_counts``, clipped to +/- ``TURNOVER_ZSCORE_CLIP``.

    A zero (or undefined, e.g. a single-row) standard deviation yields an
    all-zero z-score rather than dividing by zero or producing NaN/inf.
    """
    std = trade_counts.std()
    if not std or pd.isna(std):
        return pd.Series(0.0, index=trade_counts.index)
    z = (trade_counts - trade_counts.mean()) / std
    return z.clip(lower=-TURNOVER_ZSCORE_CLIP, upper=TURNOVER_ZSCORE_CLIP)


def score_rows(
    sweep_df: pd.DataFrame,
    weights: RankingWeights,
    asset_classes: Mapping[str, AssetClass],
) -> pd.Series:
    """Deterministic soft score per row: re-orders results, never filters them.

    ``score = oos_sharpe``
              ``- (niche_penalty * CRYPTO_PENALTY_WEIGHT)`` for crypto symbols
              ``+ (turnover_preference * clipped_trade_count_zscore * TURNOVER_PREFERENCE_WEIGHT)``

    The niche penalty only ever lowers a crypto row's score relative to
    non-crypto rows of the same OOS Sharpe; the turnover term only ever
    reorders rows within the surviving set by how their trade count
    deviates from the surviving set's mean. Neither term can push a row
    out of (or into) the result set — that is decided entirely upstream by
    ``thresholds_for`` + the six-filter funnel.
    """
    score = sweep_df["oos_sharpe"].astype("float64").copy()

    is_crypto = sweep_df["symbol"].map(
        lambda symbol: asset_classes.get(symbol) == AssetClass.CRYPTO
    )
    score = score - np.where(is_crypto, weights.niche_penalty * CRYPTO_PENALTY_WEIGHT, 0.0)

    zscore = _trade_count_zscore(sweep_df["oos_trade_count"].astype("float64"))
    score = score + weights.turnover_preference * zscore * TURNOVER_PREFERENCE_WEIGHT

    return score.rename("soft_score")


def intraday_warning(sliders: SliderValues) -> str | None:
    """Return a warning string if ``time_horizon`` requests unsupported intraday granularity.

    Returns ``None`` when ``time_horizon >= TIME_HORIZON_INTRADAY_UNSUPPORTED_BELOW``.
    """
    if sliders.time_horizon < TIME_HORIZON_INTRADAY_UNSUPPORTED_BELOW:
        return (
            f"time_horizon={sliders.time_horizon} requests intraday-like granularity: "
            "intraday granularity unsupported in v1 — EOD data only."
        )
    return None


def _turnover_direction(turnover_preference: float) -> str:
    """Plain-language direction for ``explain_mapping``'s time_horizon entry."""
    if turnover_preference > 0:
        return "favors higher-turnover"
    if turnover_preference < 0:
        return "favors lower-turnover"
    return "neutral"


def explain_mapping(sliders: SliderValues, base: FunnelThresholds) -> dict[str, str]:
    """Human-readable, per-slider explanation of the hard and soft mappings applied.

    One entry per slider (``drawdown_tolerance``, ``risk_tolerance``,
    ``capital``, ``time_horizon``), describing exactly what threshold or
    weight the slider's current value produced. Intended to be rendered
    directly in the UI so the hybrid mapping is never a black box.
    """
    thresholds = thresholds_for(sliders, base)
    weights = ranking_weights(sliders)

    explanation = {
        "drawdown_tolerance": (
            f"drawdown_tolerance={sliders.drawdown_tolerance} -> max OOS drawdown floor "
            f"set to {thresholds.max_dd_floor:.1%} (hard filter)."
        ),
        "risk_tolerance": (
            f"risk_tolerance={sliders.risk_tolerance} -> max OOS Sharpe ceiling set to "
            f"{thresholds.max_oos_sharpe:.2f}, min OOS trades required set to "
            f"{thresholds.min_trades} (both hard filters)."
        ),
        "capital": (
            f"capital={sliders.capital} -> niche (crypto) penalty weight set to "
            f"{weights.niche_penalty:.2f} (soft re-ranking only, never excludes)."
        ),
        "time_horizon": (
            f"time_horizon={sliders.time_horizon} -> turnover preference set to "
            f"{weights.turnover_preference:+.2f} "
            f"({_turnover_direction(weights.turnover_preference)} strategies; "
            "soft re-ranking only, never excludes)."
        ),
    }

    warning = intraday_warning(sliders)
    if warning is not None:
        explanation["time_horizon"] += f" WARNING: {warning}"

    return explanation
