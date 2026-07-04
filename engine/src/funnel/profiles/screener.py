"""Profile-adjusted screening of sweep results (PRD §8, §11.1).

Re-applies the six-filter funnel with profile-adjusted (hard-mapped)
thresholds to the raw sweep results DataFrame, then re-ranks survivors by
the soft score. Also enforces the hard tradeability constraints: every
surviving single-asset row is tradeable on the long-only track (with a note
that short signals are clipped to flat), while any cross-sectional-momentum
row is flagged research-only / not tradeable (it holds short positions,
which conflicts with the long-only constraint).
"""

from collections.abc import Mapping

import pandas as pd

from funnel.config import FunnelThresholds
from funnel.data.universe import AssetClass
from funnel.profiles.mapping import RankingWeights, ranking_weights, score_rows, thresholds_for
from funnel.profiles.models import SliderValues

LONG_ONLY_NOTE = "Signals are clipped to long-only on the tradeable track (shorts map to flat)."
"""Per PRD §2/§11.1 hard constraint: the tradeable track never holds shorts."""

CROSS_SECTIONAL_FAMILY_PREFIX = "cross_sectional"
"""Family-name prefix used to identify research-only, short-holding rows
(guarded by name since these rows are not part of ``sweep_df`` today, but
the screener must not silently mark them tradeable if they ever are)."""


def _passes_six_filters(row: pd.Series, thresholds: FunnelThresholds) -> bool:
    """Recompute the six-filter funnel verdict for one sweep row.

    Mirrors ``funnel.backtest.funnel.apply_funnel`` exactly (same six
    comparisons, same overfit-gap special case when ``is_sharpe <= 0``) but
    reads from a sweep DataFrame row's scalar columns instead of a
    ``WalkForwardResult``, since constructing a ``WalkForwardResult`` from a
    row would require its (discarded) return series. A dedicated test
    asserts this stays in agreement with ``apply_funnel`` on sampled rows.
    """
    is_sharpe = row["is_sharpe"]
    oos_sharpe = row["oos_sharpe"]
    oos_max_drawdown = row["oos_max_drawdown"]
    oos_trade_count = row["oos_trade_count"]

    passes_max_dd_floor = oos_max_drawdown > thresholds.max_dd_floor
    passes_min_oos_sharpe = oos_sharpe > thresholds.min_oos_sharpe
    passes_max_oos_sharpe = oos_sharpe < thresholds.max_oos_sharpe

    if is_sharpe > 0:
        passes_overfit_gap = oos_sharpe <= is_sharpe * thresholds.max_oos_is_ratio
    else:
        passes_overfit_gap = False

    passes_min_trades = oos_trade_count >= thresholds.min_trades

    if thresholds.require_positive_is_sharpe:
        passes_positive_is_sharpe = is_sharpe > 0
    else:
        passes_positive_is_sharpe = True

    return (
        passes_max_dd_floor
        and passes_min_oos_sharpe
        and passes_max_oos_sharpe
        and passes_overfit_gap
        and passes_min_trades
        and passes_positive_is_sharpe
    )


def _is_cross_sectional(family: str) -> bool:
    return family.startswith(CROSS_SECTIONAL_FAMILY_PREFIX)


def screen(
    sweep_df: pd.DataFrame,
    sliders: SliderValues,
    base_thresholds: FunnelThresholds,
    asset_classes: Mapping[str, AssetClass],
) -> pd.DataFrame:
    """Screen sweep results against profile-adjusted funnel thresholds.

    Steps: drop skipped rows; recompute the six-filter pass/fail per row
    under ``thresholds_for(sliders, base_thresholds)``; keep only rows that
    survive; annotate tradeability (``tradeable`` / ``long_only_note`` /
    research-only exclusion for any ``cross_sectional*`` family); sort
    descending by soft score, added as a ``soft_score`` column.
    """
    thresholds = thresholds_for(sliders, base_thresholds)
    weights: RankingWeights = ranking_weights(sliders)

    working = sweep_df.loc[~sweep_df["skipped"]].copy()

    survives = working.apply(lambda row: _passes_six_filters(row, thresholds), axis=1)
    survivors = working.loc[survives].copy()

    is_cross_sectional = survivors["family"].map(_is_cross_sectional)
    survivors["tradeable"] = ~is_cross_sectional
    survivors["long_only_note"] = LONG_ONLY_NOTE

    survivors["soft_score"] = score_rows(survivors, weights, asset_classes)
    survivors = survivors.sort_values("soft_score", ascending=False)

    return survivors


def screen_summary(df: pd.DataFrame, top_n: int = 10) -> dict[str, object]:
    """Summarize a screened DataFrame: counts and a top-N preview.

    ``df`` is expected to be the output of ``screen`` (already sorted
    descending by ``soft_score``), so the top-N preview is simply the head.
    """
    return {
        "n_survivors": len(df),
        "n_tradeable": int(df["tradeable"].sum()) if "tradeable" in df.columns else 0,
        "n_research_only": int((~df["tradeable"]).sum()) if "tradeable" in df.columns else 0,
        "top": df.head(top_n).to_dict(orient="records"),
    }
