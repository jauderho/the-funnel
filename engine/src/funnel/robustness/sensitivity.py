"""Parameter sensitivity: is a family's edge robust across its parameter grid, or a fluke?

Groups the raw sweep DataFrame (``funnel.backtest.sweep.run_sweep`` output) by
strategy ``family`` and reports the spread of OOS Sharpe across every config x
asset backtest in that family. A family whose OOS Sharpe is only good for one
exact parameter setting — high standard deviation, low fraction of
Sharpe-positive rows — is a curve-fit red flag: the "edge" is likely an
artifact of that one setting rather than something the family reliably does.
A family with a tight OOS Sharpe spread and a high positive fraction across
many parameter settings and assets is evidence of a real, parameter-insensitive
edge.
"""

from pathlib import Path

import pandas as pd


def family_sensitivity(sweep_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize OOS Sharpe sensitivity per strategy family.

    ``sweep_df`` is the raw sweep DataFrame (one row per config x asset
    pair, including skipped rows). Skipped rows are excluded before
    grouping — they carry no OOS Sharpe to be sensitive about.

    Returns one row per family with columns ``family``, ``n_configs``
    (distinct config count), ``n_backtests`` (row count), ``mean_oos_sharpe``,
    ``std_oos_sharpe`` (sample std, ddof=1, across all that family's rows),
    ``positive_fraction`` (fraction of rows with ``oos_sharpe > 0``), sorted
    by ``mean_oos_sharpe`` descending.
    """
    ran = sweep_df.loc[~sweep_df["skipped"].astype(bool)]

    rows: list[dict[str, object]] = []
    for family, group in ran.groupby("family", sort=True):
        n_backtests = len(group)
        std_oos_sharpe = group["oos_sharpe"].std(ddof=1) if n_backtests >= 2 else 0.0
        rows.append(
            {
                "family": family,
                "n_configs": int(group["config_name"].nunique()),
                "n_backtests": n_backtests,
                "mean_oos_sharpe": float(group["oos_sharpe"].mean()),
                "std_oos_sharpe": float(std_oos_sharpe) if pd.notna(std_oos_sharpe) else 0.0,
                "positive_fraction": float((group["oos_sharpe"] > 0).sum() / n_backtests),
            }
        )

    result = pd.DataFrame(
        rows,
        columns=[
            "family",
            "n_configs",
            "n_backtests",
            "mean_oos_sharpe",
            "std_oos_sharpe",
            "positive_fraction",
        ],
    )
    return result.sort_values("mean_oos_sharpe", ascending=False).reset_index(drop=True)


def write_sensitivity(df: pd.DataFrame, path: Path) -> None:
    """Write the family sensitivity DataFrame to ``path`` as CSV (``sensitivity.csv``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
