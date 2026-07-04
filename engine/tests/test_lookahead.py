"""The critical no-look-ahead guard, parameterized over every StrategyConfig.

For each config: compute positions on the full fixture, then again on the
fixture truncated to t=400, and assert the two agree everywhere in the
overlap [:400]. Truncating the future must never change a past decision.
"""

import pandas as pd
import pytest

from funnel.strategies.base import VALID_POSITIONS
from funnel.strategies.grid import StrategyConfig, build_all_configs

TRUNCATE_AT = 400

ALL_CONFIGS = build_all_configs()


@pytest.fixture(params=ALL_CONFIGS, ids=[c.name for c in ALL_CONFIGS])
def config(request: pytest.FixtureRequest) -> StrategyConfig:
    return request.param


@pytest.fixture(params=["trending_ohlcv", "mean_reverting_ohlcv"])
def ohlcv(request: pytest.FixtureRequest) -> pd.DataFrame:
    """Run the guard on both regimes so branches idle in one are exercised in the other."""
    return request.getfixturevalue(request.param)


def test_no_lookahead(config: StrategyConfig, ohlcv: pd.DataFrame) -> None:
    full_positions = config.fn(ohlcv, **config.params)
    truncated = ohlcv.iloc[:TRUNCATE_AT]
    truncated_positions = config.fn(truncated, **config.params)

    pd.testing.assert_series_equal(
        full_positions.iloc[:TRUNCATE_AT],
        truncated_positions,
        check_names=False,
    )


def test_output_contract(config: StrategyConfig, trending_ohlcv: pd.DataFrame) -> None:
    positions = config.fn(trending_ohlcv, **config.params)

    assert positions.index.equals(trending_ohlcv.index)
    assert positions.dtype == "float64"
    assert not positions.isna().any()
    assert set(positions.unique()).issubset(VALID_POSITIONS)
