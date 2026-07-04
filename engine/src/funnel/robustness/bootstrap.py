"""Bootstrap stress test: how much does sequencing and sampling risk change the picture?

PRD §6 asks for two related but distinct questions about a survivor's stitched
OOS daily return series, and this module answers each with the resampling
scheme that actually produces a meaningful answer:

- **Drawdown risk is a sequencing question.** The same multiset of daily
  returns played back in a different order produces a different equity path
  and therefore a different max drawdown — that is the entire point of
  "what if the bad days had clustered differently". So drawdown percentiles
  come from pure *order permutations*: reshuffle the existing returns (no
  replacement, no resampling), and recompute ``max_drawdown`` per permuted
  path.
- **Sharpe is order-invariant under any permutation** — mean and standard
  deviation of a fixed multiset of numbers do not depend on the order the
  numbers appear in, so every permuted path has exactly the same Sharpe as
  the original. Feeding pure permutations into "Sharpe distribution" would
  therefore report a degenerate distribution equal to the observed Sharpe
  ``n_reshuffles`` times over, which answers nothing. A meaningful Sharpe
  distribution requires resampling *with* replacement (the standard i.i.d.
  bootstrap): draw ``len(returns)`` observations with replacement, which can
  change the mean/std and therefore the Sharpe. That is what this module
  uses for the p5/p50/p95 Sharpe figures.

Both schemes are seeded from the same ``numpy.random.default_rng(seed)`` for
full determinism given a fixed seed.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from funnel.backtest.metrics import max_drawdown, sharpe

DEFAULT_DD_FLOOR = -0.35
"""Default survivability floor for max drawdown, matching
``FunnelThresholds.max_dd_floor``'s default — callers should pass the actual
configured threshold rather than relying on this default in production use."""


@dataclass(slots=True, frozen=True)
class BootstrapResult:
    """Bootstrap stress test outcome for one (config, symbol) survivor's OOS returns."""

    sharpe_p5: float
    """5th percentile Sharpe across the with-replacement bootstrap resamples."""

    sharpe_p50: float
    """50th percentile (median) Sharpe across the with-replacement bootstrap resamples."""

    sharpe_p95: float
    """95th percentile Sharpe across the with-replacement bootstrap resamples."""

    worst_case_drawdown: float
    """Most negative max drawdown (<=0.0) across all order-permutation paths."""

    dd_p5: float
    """5th percentile max drawdown across all order-permutation paths."""

    n_reshuffles: int
    """Number of reshuffles/resamples performed for each scheme."""

    verdict: str
    """``"fragile"`` if ``worst_case_drawdown`` breaches the survivability
    floor (is more negative than ``dd_floor``), else ``"solid"``."""


def bootstrap_stress(
    oos_returns: pd.Series,
    n_reshuffles: int = 200,
    *,
    seed: int,
    dd_floor: float = DEFAULT_DD_FLOOR,
) -> BootstrapResult:
    """Stress-test a stitched OOS return series via order permutation and i.i.d. bootstrap.

    Drawdown percentiles come from ``n_reshuffles`` pure order permutations of
    ``oos_returns`` (reordering the same daily returns — no replacement).
    Sharpe percentiles come from ``n_reshuffles`` i.i.d. resamples of the same
    size drawn *with* replacement, since Sharpe is order-invariant under
    permutation alone (see module docstring). Both use the same
    ``numpy.random.default_rng(seed)`` for determinism.

    ``dd_floor`` is the survivability floor (e.g.
    ``FunnelThresholds.max_dd_floor``); the verdict is ``"fragile"`` if
    ``worst_case_drawdown < dd_floor``, else ``"solid"``.
    """
    rng = np.random.default_rng(seed)
    clean = oos_returns.dropna().to_numpy()
    n = len(clean)

    drawdowns: list[float] = []
    for _ in range(n_reshuffles):
        permuted = rng.permutation(clean)
        drawdowns.append(max_drawdown(pd.Series(permuted)))

    sharpes: list[float] = []
    for _ in range(n_reshuffles):
        resampled = rng.choice(clean, size=n, replace=True)
        sharpes.append(sharpe(pd.Series(resampled)))

    dd_array = np.array(drawdowns)
    sharpe_array = np.array(sharpes)

    worst_case_drawdown = float(dd_array.min()) if n_reshuffles else 0.0
    dd_p5 = float(np.percentile(dd_array, 5)) if n_reshuffles else 0.0
    sharpe_p5 = float(np.percentile(sharpe_array, 5)) if n_reshuffles else 0.0
    sharpe_p50 = float(np.percentile(sharpe_array, 50)) if n_reshuffles else 0.0
    sharpe_p95 = float(np.percentile(sharpe_array, 95)) if n_reshuffles else 0.0

    verdict = "fragile" if worst_case_drawdown < dd_floor else "solid"

    return BootstrapResult(
        sharpe_p5=sharpe_p5,
        sharpe_p50=sharpe_p50,
        sharpe_p95=sharpe_p95,
        worst_case_drawdown=worst_case_drawdown,
        dd_p5=dd_p5,
        n_reshuffles=n_reshuffles,
        verdict=verdict,
    )


def run_bootstrap_for_survivors(
    sweep_df: pd.DataFrame,
    oos_returns_by_key: Mapping[tuple[str, str], pd.Series],
    dd_floor: float,
    n_reshuffles: int = 200,
    *,
    seed: int,
) -> pd.DataFrame:
    """Run the bootstrap stress test on every surviving (config, symbol) row in the sweep.

    ``oos_returns_by_key`` maps ``(config_name, symbol)`` to that pair's
    stitched OOS return series (``WalkForwardResult.oos_returns``). ``dd_floor``
    is the survivability floor to apply (callers pass
    ``FunnelThresholds.max_dd_floor``).

    Returns one row per survivor with columns ``config_name``, ``family``,
    ``symbol``, ``oos_sharpe``, ``sharpe_p5``, ``sharpe_p50``, ``sharpe_p95``,
    ``worst_case_drawdown``, ``dd_p5``, ``verdict``.
    """
    survivors = sweep_df.loc[sweep_df["survived"].astype(bool)]

    rows: list[dict[str, object]] = []
    for _, row in survivors.iterrows():
        key = (row["config_name"], row["symbol"])
        oos_returns = oos_returns_by_key[key]
        result = bootstrap_stress(oos_returns, n_reshuffles, seed=seed, dd_floor=dd_floor)
        rows.append(
            {
                "config_name": row["config_name"],
                "family": row["family"],
                "symbol": row["symbol"],
                "oos_sharpe": row["oos_sharpe"],
                "sharpe_p5": result.sharpe_p5,
                "sharpe_p50": result.sharpe_p50,
                "sharpe_p95": result.sharpe_p95,
                "worst_case_drawdown": result.worst_case_drawdown,
                "dd_p5": result.dd_p5,
                "verdict": result.verdict,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "config_name",
            "family",
            "symbol",
            "oos_sharpe",
            "sharpe_p5",
            "sharpe_p50",
            "sharpe_p95",
            "worst_case_drawdown",
            "dd_p5",
            "verdict",
        ],
    )


def write_bootstrap(df: pd.DataFrame, path: Path) -> None:
    """Write the bootstrap stress test DataFrame to ``path`` as CSV (``bootstrap.csv``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
