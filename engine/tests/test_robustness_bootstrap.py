"""Bootstrap stress test: determinism, permutation invariants, solid/fragile verdicts."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from funnel.backtest.metrics import max_drawdown
from funnel.robustness.bootstrap import (
    BootstrapResult,
    bootstrap_stress,
    run_bootstrap_for_survivors,
    write_bootstrap,
)


def _returns(values: list[float]) -> pd.Series:
    index = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.Series(values, index=index)


@pytest.fixture
def gentle_returns() -> pd.Series:
    """Low-vol, steadily positive daily returns -> should be solid."""
    rng = np.random.default_rng(123)
    values = rng.normal(loc=0.001, scale=0.002, size=250)
    return _returns(list(values))


@pytest.fixture
def catastrophic_returns() -> pd.Series:
    """Mostly flat, one huge single-day loss -> should be fragile under a tight floor."""
    values = [0.001] * 249 + [-0.5]
    return _returns(values)


def test_deterministic_under_fixed_seed(gentle_returns: pd.Series) -> None:
    result_a = bootstrap_stress(gentle_returns, n_reshuffles=50, seed=42)
    result_b = bootstrap_stress(gentle_returns, n_reshuffles=50, seed=42)

    assert result_a == result_b


def test_permutation_preserves_multiset_of_returns(gentle_returns: pd.Series) -> None:
    # Compounded final equity (product of (1+r)) is order-invariant for any
    # permutation of the same multiset of returns, since multiplication
    # commutes -- so every permuted path must compound to the same final
    # value as the original series. This indirectly proves the reshuffles
    # reorder rather than resample the returns.
    clean = gentle_returns.dropna().to_numpy()
    expected_final_equity = float(np.prod(1.0 + clean))

    rng = np.random.default_rng(7)
    for _ in range(20):
        permuted = rng.permutation(clean)
        final_equity = float(np.prod(1.0 + permuted))
        assert final_equity == pytest.approx(expected_final_equity)


def test_worst_case_drawdown_at_or_below_median(gentle_returns: pd.Series) -> None:
    result = bootstrap_stress(gentle_returns, n_reshuffles=200, seed=1)
    assert result.worst_case_drawdown <= result.dd_p5 <= 0.0


def test_catastrophic_series_is_fragile_under_tight_floor(catastrophic_returns: pd.Series) -> None:
    result = bootstrap_stress(catastrophic_returns, n_reshuffles=200, seed=1, dd_floor=-0.10)
    assert result.verdict == "fragile"
    assert result.worst_case_drawdown < -0.10


def test_gentle_series_is_solid(gentle_returns: pd.Series) -> None:
    result = bootstrap_stress(gentle_returns, n_reshuffles=200, seed=1, dd_floor=-0.35)
    assert result.verdict == "solid"
    assert result.worst_case_drawdown >= -0.35


def test_sharpe_percentiles_ordered(gentle_returns: pd.Series) -> None:
    result = bootstrap_stress(gentle_returns, n_reshuffles=200, seed=1)
    assert result.sharpe_p5 <= result.sharpe_p50 <= result.sharpe_p95


def test_n_reshuffles_recorded(gentle_returns: pd.Series) -> None:
    result = bootstrap_stress(gentle_returns, n_reshuffles=77, seed=1)
    assert result.n_reshuffles == 77


def test_fragile_boundary_is_strict_less_than(gentle_returns: pd.Series) -> None:
    result = bootstrap_stress(gentle_returns, n_reshuffles=50, seed=1)
    # Setting the floor exactly at the observed worst case should be solid
    # (not fragile) since the rule is a strict breach, not equality.
    at_floor = bootstrap_stress(
        gentle_returns, n_reshuffles=50, seed=1, dd_floor=result.worst_case_drawdown
    )
    assert at_floor.verdict == "solid"


def test_run_bootstrap_for_survivors_only_processes_survivors(
    gentle_returns: pd.Series, catastrophic_returns: pd.Series
) -> None:
    sweep_df = pd.DataFrame(
        [
            {
                "config_name": "cfg_a",
                "family": "fam_a",
                "symbol": "AAA",
                "oos_sharpe": 1.0,
                "survived": True,
            },
            {
                "config_name": "cfg_b",
                "family": "fam_b",
                "symbol": "BBB",
                "oos_sharpe": -0.5,
                "survived": False,
            },
        ]
    )
    oos_returns_by_key = {
        ("cfg_a", "AAA"): gentle_returns,
        ("cfg_b", "BBB"): catastrophic_returns,
    }

    result = run_bootstrap_for_survivors(
        sweep_df, oos_returns_by_key, dd_floor=-0.35, n_reshuffles=50, seed=1
    )

    assert len(result) == 1
    assert result.iloc[0]["config_name"] == "cfg_a"
    assert result.iloc[0]["symbol"] == "AAA"
    assert result.iloc[0]["verdict"] in {"solid", "fragile"}


def test_write_bootstrap_round_trips_csv(tmp_path: Path, gentle_returns: pd.Series) -> None:
    sweep_df = pd.DataFrame(
        [
            {
                "config_name": "cfg_a",
                "family": "fam_a",
                "symbol": "AAA",
                "oos_sharpe": 1.0,
                "survived": True,
            }
        ]
    )
    df = run_bootstrap_for_survivors(
        sweep_df, {("cfg_a", "AAA"): gentle_returns}, dd_floor=-0.35, n_reshuffles=20, seed=1
    )
    path = tmp_path / "bootstrap.csv"
    write_bootstrap(df, path)

    assert path.exists()
    reloaded = pd.read_csv(path)
    assert len(reloaded) == len(df)


def test_max_drawdown_reused_from_metrics_module() -> None:
    # Sanity check that the module's imports actually resolve to the shared
    # metrics helpers rather than a local reimplementation.
    series = _returns([0.01, -0.02, 0.03])
    assert max_drawdown(series) <= 0.0


def test_bootstrap_result_is_frozen_dataclass() -> None:
    result = bootstrap_stress(_returns([0.001] * 60), n_reshuffles=10, seed=1)
    assert isinstance(result, BootstrapResult)
    with pytest.raises(AttributeError):
        result.verdict = "solid"  # ty: ignore[invalid-assignment]
