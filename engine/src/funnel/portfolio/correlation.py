"""Cross-strategy correlation matrix and redundancy flags (PRD §11.4).

Two survivors that are near-perfectly correlated are, in substance, the same
trade wearing different indicator names — running both adds cost and
apparent diversification without adding anything real. This module surfaces
that directly: a pairwise correlation matrix of daily returns, and a
long-form list of pairs whose correlation crosses a "these are the same
trade" threshold.
"""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

DEFAULT_MIN_OVERLAP = 60
"""Default minimum number of overlapping return observations required to
compute a pair's correlation. Pairs with less overlap than this produce a
statistically meaningless correlation estimate and are reported as NaN
rather than a number that looks precise but isn't."""


def correlation_matrix(
    returns_by_name: Mapping[str, pd.Series],
    min_overlap: int = DEFAULT_MIN_OVERLAP,
) -> pd.DataFrame:
    """Pairwise Pearson correlation of daily returns across strategies.

    Each pair's returns are aligned on their overlapping index (inner join,
    NaNs dropped) before computing Pearson correlation. Pairs with fewer
    than ``min_overlap`` overlapping observations are reported as ``NaN``
    rather than a correlation computed on too little data. The diagonal is
    always ``1.0`` (a series' correlation with itself, given at least one
    observation) and the matrix is symmetric.

    Returns a square DataFrame indexed and columned by
    ``sorted(returns_by_name.keys())``.
    """
    names = sorted(returns_by_name.keys())
    matrix = pd.DataFrame(index=names, columns=names, dtype="float64")

    for i, name_a in enumerate(names):
        for name_b in names[i:]:
            if name_a == name_b:
                corr = 1.0
            else:
                aligned = pd.concat(
                    [returns_by_name[name_a], returns_by_name[name_b]], axis=1, join="inner"
                ).dropna()
                if len(aligned) < min_overlap:
                    corr = float("nan")
                else:
                    corr = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
            matrix.loc[name_a, name_b] = corr
            matrix.loc[name_b, name_a] = corr

    return matrix


def redundancy_flags(corr: pd.DataFrame, threshold: float = 0.85) -> pd.DataFrame:
    """Long-form list of strategy pairs whose correlation is ``>= threshold``.

    Reads only the upper triangle of ``corr`` (excluding the diagonal) so
    each unordered pair is reported exactly once. NaN entries (insufficient
    overlap) never flag as redundant. Returns columns ``a``, ``b``, ``corr``,
    sorted by ``corr`` descending — the "your strategies are the same
    trade" warning, highest-overlap pairs first.
    """
    names = list(corr.index)
    rows: list[dict[str, object]] = []
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            value = corr.loc[name_a, name_b]
            if pd.notna(value) and value >= threshold:
                rows.append({"a": name_a, "b": name_b, "corr": float(value)})

    result = pd.DataFrame(rows, columns=["a", "b", "corr"])
    return result.sort_values("corr", ascending=False).reset_index(drop=True)


def write_correlation(corr: pd.DataFrame, path: Path) -> None:
    """Write the correlation matrix to ``path`` as CSV (``correlation_matrix.csv``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    corr.to_csv(path)
