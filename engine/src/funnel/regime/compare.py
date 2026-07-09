"""Compare regime detectors against each other.

Detectors are simple, individually defensible methods that nonetheless
often disagree with each other — this module is the "treat it as a
research direction" deliverable: it quantifies how much they disagree
(``agreement_matrix``) and summarizes each detector's overall behavior
(``compare_detectors``), so a user can decide whether to trust any given
detector for routing rather than taking one implementation's labels on
faith.
"""

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from funnel.regime.base import (
    Regime,
    RegimeDetector,
    RegimeMetrics,
    regime_conditioned_metrics,
)


def compare_detectors(df: pd.DataFrame, detectors: Mapping[str, RegimeDetector]) -> pd.DataFrame:
    """Summarize each detector's labeling behavior on the same market proxy frame.

    Returns one row per detector (indexed by the ``detectors`` mapping key,
    column ``detector``) with:

    - ``fraction_trending``: fraction of days labeled ``Regime.TRENDING``.
    - ``n_switches``: number of days where the label differs from the
      previous day (regime changes).
    - ``mean_spell_length``: mean number of consecutive days per unbroken
      regime spell (``len(df) / (n_switches + 1)``).

    Thin convenience wrapper around ``compare_detectors_from_labels`` for a
    caller that only wants the comparison table and has no other use for
    the raw labels. A caller that *also* needs the labels for something
    else (``funnel.pipeline``'s regime stage, which reuses them for
    ``agreement_matrix`` and regime-conditioned performance) should call
    each detector's ``classify`` once itself and use
    ``compare_detectors_from_labels`` directly — some detectors
    (``regime.changepoint.ChangePointDetector`` in particular) are
    expensive enough that calling ``classify`` twice measurably doubles
    this stage's wall time for no benefit, since ``classify`` is a pure
    function of ``df`` and returns the identical series both times.
    """
    labels_by_detector = {name: detector.classify(df) for name, detector in detectors.items()}
    return compare_detectors_from_labels(labels_by_detector)


def compare_detectors_from_labels(labels_by_detector: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Same output as ``compare_detectors``, from already-computed label series.

    See ``compare_detectors`` for the column documentation and the
    duplicate-``classify``-call rationale for preferring this function when
    the caller already has (or separately needs) the labels.
    """
    columns = ["detector", "fraction_trending", "n_switches", "mean_spell_length"]
    rows: list[dict[str, object]] = []
    for name, labels in labels_by_detector.items():
        rows.append({"detector": name, **_spell_stats(labels)})
    return pd.DataFrame(rows, columns=columns)


def _spell_stats(labels: pd.Series) -> dict[str, float]:
    n = len(labels)
    if n == 0:
        return {"fraction_trending": 0.0, "n_switches": 0, "mean_spell_length": 0.0}
    fraction_trending = float((labels == Regime.TRENDING).sum() / n)
    # First row always compares True against shift(1)'s NaN; subtract that
    # spurious "switch" so n_switches counts only actual regime changes.
    n_switches = int((labels != labels.shift(1)).sum()) - 1
    mean_spell_length = n / (n_switches + 1)
    return {
        "fraction_trending": fraction_trending,
        "n_switches": n_switches,
        "mean_spell_length": mean_spell_length,
    }


def agreement_matrix(labels: Mapping[str, pd.Series]) -> pd.DataFrame:
    """Pairwise fraction of days on which each pair of detectors agree.

    ``labels`` maps a detector name to its label series (all assumed
    aligned to the same index). The diagonal is always 1.0 (a detector
    agrees with itself on every day).
    """
    names = list(labels.keys())
    matrix = pd.DataFrame(index=names, columns=names, dtype="float64")
    for a in names:
        for b in names:
            agree = (labels[a] == labels[b]).mean()
            matrix.loc[a, b] = float(agree)
    return matrix


def assemble_regime_performance(
    runs: Mapping[str, tuple[pd.Series, pd.Series]],
) -> pd.DataFrame:
    """Build the ``regime_performance.csv`` frame from per-strategy runs.

    ``runs`` maps a strategy/config identifier to a ``(returns, regimes)``
    pair, as consumed by ``funnel.regime.base.regime_conditioned_metrics``.
    Returns one row per (identifier, regime) with columns ``name``,
    ``regime``, ``sharpe``, ``max_drawdown``, ``n_days``, ``total_return``.
    """
    rows: list[dict[str, object]] = []
    for name, (returns, regimes) in runs.items():
        per_regime = regime_conditioned_metrics(returns, regimes)
        for regime, metrics in per_regime.items():
            rows.append({"name": name, "regime": str(regime), **_metrics_dict(metrics)})
    return pd.DataFrame(
        rows, columns=["name", "regime", "sharpe", "max_drawdown", "n_days", "total_return"]
    )


def _metrics_dict(metrics: RegimeMetrics) -> dict[str, object]:
    return {
        "sharpe": metrics.sharpe,
        "max_drawdown": metrics.max_drawdown,
        "n_days": metrics.n_days,
        "total_return": metrics.total_return,
    }


def write_regime_performance(metrics_rows: pd.DataFrame, path: Path) -> None:
    """Write a regime-performance DataFrame to ``path`` as CSV (``regime_performance.csv``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics_rows.to_csv(path, index=False)
