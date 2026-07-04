"""Tests for the backtest engine: cost timing and no-look-ahead position application."""

import pandas as pd
import pytest

from funnel.backtest.engine import cost_bps_for, strategy_returns
from funnel.config import CostModel
from funnel.data.universe import AssetClass


def test_costs_charged_exactly_on_position_changes() -> None:
    # positions: flat -> long -> long -> short -> flat
    positions = pd.Series([0.0, 1.0, 1.0, -1.0, 0.0])
    close = pd.Series([100.0, 102.0, 101.0, 103.0, 104.0])
    cost_bps = 10.0  # 10 bps/side

    net = strategy_returns(positions, close, cost_bps)

    asset_return = close.pct_change()
    # Row 1: position_0=0 held into row1 -> gross 0; cost = |pos1-pos0|=1 side.
    expected_row1 = 0.0 * asset_return.iloc[1] - 1 * (cost_bps / 1e4)
    # Row 2: position_1=1 held into row2; cost = |pos2-pos1|=0.
    expected_row2 = 1.0 * asset_return.iloc[2] - 0 * (cost_bps / 1e4)
    # Row 3: position_2=1 held into row3; cost = |pos3-pos2|=2 sides (long->short).
    expected_row3 = 1.0 * asset_return.iloc[3] - 2 * (cost_bps / 1e4)
    # Row 4: position_3=-1 held into row4; cost = |pos4-pos3|=1 side.
    expected_row4 = -1.0 * asset_return.iloc[4] - 1 * (cost_bps / 1e4)

    assert net.iloc[0] == pytest.approx(expected_row1)
    assert net.iloc[1] == pytest.approx(expected_row2)
    assert net.iloc[2] == pytest.approx(expected_row3)
    assert net.iloc[3] == pytest.approx(expected_row4)


def test_no_cost_when_position_unchanged() -> None:
    positions = pd.Series([1.0, 1.0, 1.0])
    close = pd.Series([100.0, 105.0, 103.0])
    net = strategy_returns(positions, close, cost_bps_per_side=10.0)
    asset_return = close.pct_change()
    assert net.iloc[0] == pytest.approx(asset_return.iloc[1])
    assert net.iloc[1] == pytest.approx(asset_return.iloc[2])


def test_first_row_dropped_not_zero_filled() -> None:
    positions = pd.Series([0.0, 1.0, 1.0])
    close = pd.Series([100.0, 101.0, 102.0])
    net = strategy_returns(positions, close, cost_bps_per_side=1.0)
    assert len(net) == 2
    assert not net.isna().any()


def test_position_t_applies_to_return_t_plus_1_no_lookahead() -> None:
    # If position at t is decided from data up to t, applying it to return
    # t -> t+1 means a change in the future close (row 2) must not affect
    # the return already realized at row 1.
    positions = pd.Series([0.0, 1.0, 0.0])
    close_a = pd.Series([100.0, 110.0, 90.0])
    close_b = pd.Series([100.0, 110.0, 500.0])  # only the future close differs

    net_a = strategy_returns(positions, close_a, cost_bps_per_side=0.0)
    net_b = strategy_returns(positions, close_b, cost_bps_per_side=0.0)

    # Row corresponding to the 0->1 return (index 1) must be identical.
    assert net_a.iloc[0] == pytest.approx(net_b.iloc[0])


def test_cost_bps_for_crypto_uses_crypto_rate() -> None:
    costs = CostModel(default_bps_per_side=1.0, crypto_bps_per_side=5.0)
    assert cost_bps_for(AssetClass.CRYPTO, costs) == 5.0


def test_cost_bps_for_non_crypto_uses_default_rate() -> None:
    costs = CostModel(default_bps_per_side=1.0, crypto_bps_per_side=5.0)
    assert cost_bps_for(AssetClass.LARGE_CAP, costs) == 1.0
    assert cost_bps_for(AssetClass.INDEX_ETF, costs) == 1.0
