"""Layer 3 — combine multiple (weighted) position series into one signal.

Both functions here operate on position/weight series in ``[-1, 1]`` on the
*same* asset — either raw ``{-1,0,1}`` positions or already-sized weighted
positions from ``funnel.layers.sizing``. The result of ``combine_signals`` is
again a ``[-1, 1]`` series, so it can be fed straight into
``funnel.backtest.engine.strategy_returns``.
"""

from collections.abc import Mapping

import pandas as pd

from funnel.backtest.metrics import sharpe


def combine_signals(
    weighted_positions: Mapping[str, pd.Series],
    weights: Mapping[str, float] | None = None,
) -> pd.Series:
    """Blend multiple strategies' position series into one, on the same asset.

    Aligns all series on the union of their indices; a strategy with no
    value at a given row (outside its own index) contributes 0.0 there
    rather than NaN, so a strategy that starts later or ends earlier never
    poisons the blend at rows it simply doesn't cover. The result is the
    (given- or equal-) weighted average of the aligned series — still a
    position/weight series in ``[-1, 1]`` as long as the inputs are and the
    weights sum to (at most) 1.0.

    ``weights`` maps name -> weight; missing names default to equal weight
    (``1 / len(weighted_positions)``) when ``weights`` is ``None``.
    """
    if not weighted_positions:
        raise ValueError("weighted_positions must be non-empty")

    names = list(weighted_positions.keys())
    if weights is None:
        resolved_weights = dict.fromkeys(names, 1.0 / len(names))
    else:
        resolved_weights = dict(weights)

    indices = iter(s.index for s in weighted_positions.values())
    union_index = next(indices)
    for index in indices:
        union_index = union_index.union(index)

    combined = pd.Series(0.0, index=union_index, dtype="float64")
    for name in names:
        aligned = weighted_positions[name].reindex(union_index).fillna(0.0)
        combined = combined + resolved_weights[name] * aligned

    return combined.astype("float64")


def select_uncorrelated(
    returns_by_name: Mapping[str, pd.Series],
    max_pairwise_corr: float = 0.7,
    min_count: int = 2,
) -> list[str]:
    """Greedily select a set of strategies whose return streams are mutually uncorrelated.

    Greedy-by-Sharpe is a deliberate simplicity choice over an optimal
    subset search (which is combinatorial and not worth the complexity for
    a handful of candidate strategies): candidates are ordered by Sharpe
    ratio descending (``funnel.backtest.metrics.sharpe``, ties broken by
    name for determinism), the best is always taken, and each subsequent
    candidate is added only if its pairwise Pearson correlation with
    *every* already-selected series is ``<= max_pairwise_corr`` (computed
    over each pair's overlapping index; a non-overlapping or single-point
    overlap pair is treated as uncorrelated, i.e. does not block
    selection). This favors a good-enough, explainable diversification set
    over a globally optimal one.

    ``min_count`` is a soft target, not a hard requirement: if fewer than
    ``min_count`` candidates pass the correlation gate, the function
    returns whatever it found (it never forces in a correlated candidate
    just to hit the count).
    """
    ordered = sorted(
        returns_by_name.keys(), key=lambda name: (-sharpe(returns_by_name[name]), name)
    )

    selected: list[str] = []
    for candidate in ordered:
        if _too_correlated(candidate, selected, returns_by_name, max_pairwise_corr):
            continue
        selected.append(candidate)

    return selected


def _too_correlated(
    candidate: str,
    selected: list[str],
    returns_by_name: Mapping[str, pd.Series],
    max_pairwise_corr: float,
) -> bool:
    candidate_returns = returns_by_name[candidate]
    for name in selected:
        aligned = pd.concat(
            [candidate_returns, returns_by_name[name]], axis=1, join="inner"
        ).dropna()
        if len(aligned) < 2:
            continue
        corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
        if pd.notna(corr) and corr > max_pairwise_corr:
            return True
    return False
