"""The funnel attrition report: how many backtests survive each stage, and why.

Honesty-by-design (PLAN.md): the report always embeds the exact
``FunnelThresholds`` that were applied, is computed from the raw sweep
DataFrame (never a pre-filtered subset), and always reports the skipped
(insufficient-history) count separately rather than folding it into either
survivors or failures.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from funnel.config import FunnelThresholds

TOP_SURVIVORS_CAP = 25


@dataclass(slots=True)
class CategoryAttrition:
    """Survival rate and mean OOS Sharpe for one category or family."""

    name: str
    n_total: int
    n_survived: int
    survival_rate: float
    mean_oos_sharpe: float


@dataclass(slots=True)
class AttritionReport:
    """Full funnel attrition summary computed from one sweep DataFrame."""

    thresholds: FunnelThresholds
    """The exact thresholds applied to produce this report."""

    n_total_backtests: int
    """Total rows in the sweep, including skipped pairs."""

    n_skipped: int
    """Rows skipped for insufficient history (reported separately, never
    folded into either survivors or failures)."""

    n_run: int
    """``n_total_backtests - n_skipped``: backtests that actually ran."""

    n_positive_oos_sharpe: int
    """Of the backtests that ran, how many have ``oos_sharpe > 0``."""

    n_clears_min_oos_sharpe: int
    """Of the backtests that ran, how many pass filter 2 (min OOS Sharpe)."""

    n_survived: int
    """Of the backtests that ran, how many pass all six filters."""

    by_category: list[CategoryAttrition] = field(default_factory=list)
    by_family: list[CategoryAttrition] = field(default_factory=list)
    top_survivors: pd.DataFrame = field(default_factory=pd.DataFrame)
    """Survivors sorted by ``oos_sharpe`` descending, capped at
    ``TOP_SURVIVORS_CAP`` rows."""


def _group_attrition(df: pd.DataFrame, group_col: str) -> list[CategoryAttrition]:
    results: list[CategoryAttrition] = []
    for name, group in df.groupby(group_col, sort=True):
        n_total = len(group)
        n_survived = int(group["survived"].sum())
        survival_rate = n_survived / n_total if n_total else 0.0
        mean_oos_sharpe = float(group["oos_sharpe"].mean()) if n_total else 0.0
        results.append(
            CategoryAttrition(
                name=str(name),
                n_total=n_total,
                n_survived=n_survived,
                survival_rate=survival_rate,
                mean_oos_sharpe=mean_oos_sharpe,
            )
        )
    return results


def build_attrition_report(df: pd.DataFrame, thresholds: FunnelThresholds) -> AttritionReport:
    """Compute the full attrition report from a raw sweep DataFrame.

    ``df`` is the DataFrame produced by ``funnel.backtest.sweep.run_sweep``
    (one row per config x asset pair, including skipped rows).
    """
    n_total_backtests = len(df)
    skipped_mask = df["skipped"].astype(bool)
    n_skipped = int(skipped_mask.sum())
    ran = df.loc[~skipped_mask]
    n_run = len(ran)

    n_positive_oos_sharpe = int((ran["oos_sharpe"] > 0).sum())
    n_clears_min_oos_sharpe = int(ran["passes_min_oos_sharpe"].sum())
    n_survived = int(ran["survived"].sum())

    by_category = _group_attrition(ran, "category")
    by_family = _group_attrition(ran, "family")

    survivors = ran.loc[ran["survived"]].sort_values("oos_sharpe", ascending=False)
    top_survivors = survivors.head(TOP_SURVIVORS_CAP).reset_index(drop=True)

    return AttritionReport(
        thresholds=thresholds,
        n_total_backtests=n_total_backtests,
        n_skipped=n_skipped,
        n_run=n_run,
        n_positive_oos_sharpe=n_positive_oos_sharpe,
        n_clears_min_oos_sharpe=n_clears_min_oos_sharpe,
        n_survived=n_survived,
        by_category=by_category,
        by_family=by_family,
        top_survivors=top_survivors,
    )


def render_text(report: AttritionReport) -> str:
    """Render the funnel attrition path as human-readable text."""
    t = report.thresholds
    lines = [
        "Funnel attrition report",
        "========================",
        (
            f"Thresholds applied: max_dd_floor={t.max_dd_floor}, "
            f"min_oos_sharpe={t.min_oos_sharpe}, max_oos_sharpe={t.max_oos_sharpe}, "
            f"max_oos_is_ratio={t.max_oos_is_ratio}, min_trades={t.min_trades}, "
            f"require_positive_is_sharpe={t.require_positive_is_sharpe}"
        ),
        "",
        f"Total backtests: {report.n_total_backtests}  (skipped: {report.n_skipped})",
        f"Ran:              {report.n_run}",
        f"  Positive OOS Sharpe:      {report.n_positive_oos_sharpe}",
        f"  Clears min OOS Sharpe:    {report.n_clears_min_oos_sharpe}",
        f"  Survived all six filters: {report.n_survived}",
        "",
        "By category (n_total, n_survived, survival_rate, mean_oos_sharpe):",
    ]
    for cat in report.by_category:
        lines.append(
            f"  {cat.name:<20} {cat.n_total:>6} {cat.n_survived:>6} "
            f"{cat.survival_rate:>8.2%} {cat.mean_oos_sharpe:>8.3f}"
        )
    lines.append("")
    lines.append("By family (n_total, n_survived, survival_rate, mean_oos_sharpe):")
    for fam in report.by_family:
        lines.append(
            f"  {fam.name:<28} {fam.n_total:>6} {fam.n_survived:>6} "
            f"{fam.survival_rate:>8.2%} {fam.mean_oos_sharpe:>8.3f}"
        )
    lines.append("")
    lines.append(f"Top survivors (capped at {TOP_SURVIVORS_CAP}, sorted by oos_sharpe desc):")
    if report.top_survivors.empty:
        lines.append("  (none)")
    else:
        for _, row in report.top_survivors.iterrows():
            lines.append(
                f"  {row['config_name']:<32} {row['symbol']:<10} "
                f"oos_sharpe={row['oos_sharpe']:.3f} oos_max_dd={row['oos_max_drawdown']:.3f}"
            )
    return "\n".join(lines)


def to_dict(report: AttritionReport) -> dict[str, object]:
    """Serialize the report to a plain dict (for the future API)."""
    t = report.thresholds
    return {
        "thresholds": {
            "max_dd_floor": t.max_dd_floor,
            "min_oos_sharpe": t.min_oos_sharpe,
            "max_oos_sharpe": t.max_oos_sharpe,
            "max_oos_is_ratio": t.max_oos_is_ratio,
            "min_trades": t.min_trades,
            "require_positive_is_sharpe": t.require_positive_is_sharpe,
        },
        "n_total_backtests": report.n_total_backtests,
        "n_skipped": report.n_skipped,
        "n_run": report.n_run,
        "n_positive_oos_sharpe": report.n_positive_oos_sharpe,
        "n_clears_min_oos_sharpe": report.n_clears_min_oos_sharpe,
        "n_survived": report.n_survived,
        "by_category": [asdict(c) for c in report.by_category],
        "by_family": [asdict(f) for f in report.by_family],
        "top_survivors": report.top_survivors.to_dict(orient="records"),
    }


def write_funnel_report(report: AttritionReport, path: Path) -> None:
    """Write per-category/family attrition rows to ``path`` as CSV (``funnel_report.csv``)."""
    rows: list[dict[str, object]] = []
    for cat in report.by_category:
        rows.append(
            {
                "group_type": "category",
                "name": cat.name,
                "n_total": cat.n_total,
                "n_survived": cat.n_survived,
                "survival_rate": cat.survival_rate,
                "mean_oos_sharpe": cat.mean_oos_sharpe,
            }
        )
    for fam in report.by_family:
        rows.append(
            {
                "group_type": "family",
                "name": fam.name,
                "n_total": fam.n_total,
                "n_survived": fam.n_survived,
                "survival_rate": fam.survival_rate,
                "mean_oos_sharpe": fam.mean_oos_sharpe,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
