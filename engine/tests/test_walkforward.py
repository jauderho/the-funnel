"""Tests for walk-forward window splitting, stitching, and per-window independence."""

import numpy as np
import pandas as pd
import pytest

from funnel.backtest.walkforward import (
    MIN_OOS_ROWS,
    InsufficientHistoryError,
    _is_oos_split,
    _window_bounds,
    walk_forward_oos,
)
from funnel.config import WalkForwardConfig
from funnel.strategies.base import Category
from funnel.strategies.grid import StrategyConfig
from funnel.strategies.trend import ma_crossover, time_series_momentum


def test_window_bounds_are_contiguous_and_cover_series() -> None:
    n_rows = 601
    n_windows = 5
    bounds = _window_bounds(n_rows, n_windows)

    assert len(bounds) == n_windows
    assert bounds[0][0] == 0
    assert bounds[-1][1] == n_rows
    for (_, end_prev), (start_next, _) in zip(bounds, bounds[1:], strict=False):
        assert end_prev == start_next  # contiguous, no gap or overlap

    total = sum(end - start for start, end in bounds)
    assert total == n_rows


def test_window_bounds_near_equal_length() -> None:
    bounds = _window_bounds(103, 5)
    sizes = [end - start for start, end in bounds]
    assert max(sizes) - min(sizes) <= 1


def test_is_oos_split_ratio() -> None:
    start, split, end = _is_oos_split(0, 100, is_fraction=0.7)
    assert start == 0
    assert split == 70
    assert end == 100


def _make_trend_df(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2015-01-01", periods=n)
    close = 100.0 + np.cumsum(rng.normal(loc=0.02, scale=1.0, size=n))
    return pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0, "close": close, "volume": 1e6},
        index=index,
    ).astype("float64")


@pytest.fixture
def strategy_config() -> StrategyConfig:
    return StrategyConfig(
        name="ma_5_20",
        family="ma_crossover",
        fn=ma_crossover,
        params={"fast": 5, "slow": 20},
        category=Category.TREND,
    )


def test_stitched_oos_length_equals_sum_of_tails(strategy_config: StrategyConfig) -> None:
    df = _make_trend_df(700, seed=1)
    wf = WalkForwardConfig(n_windows=5, is_fraction=0.7)
    result = walk_forward_oos(df, strategy_config, wf, cost_bps=1.0)

    bounds = _window_bounds(len(df), wf.n_windows)
    expected_len = 0
    for start, end in bounds:
        _, split, _ = _is_oos_split(start, end, wf.is_fraction)
        # strategy_returns drops the window's very first row, which always
        # falls in the IS segment (split > start always, since is_fraction
        # > 0) — so the OOS segment length is simply end - split.
        expected_len += end - split

    assert len(result.oos_returns) == expected_len


def test_is_oos_split_lengths_roughly_70_30(strategy_config: StrategyConfig) -> None:
    df = _make_trend_df(700, seed=2)
    wf = WalkForwardConfig(n_windows=5, is_fraction=0.7)
    result = walk_forward_oos(df, strategy_config, wf, cost_bps=1.0)

    total = len(result.is_returns) + len(result.oos_returns)
    is_frac = len(result.is_returns) / total
    assert is_frac == pytest.approx(0.7, abs=0.02)


def test_per_window_independence_mutating_late_window_data(strategy_config: StrategyConfig) -> None:
    """Mutating window 5's data must not change window 1's OOS returns."""
    df = _make_trend_df(700, seed=3)
    wf = WalkForwardConfig(n_windows=5, is_fraction=0.7)

    baseline = walk_forward_oos(df, strategy_config, wf, cost_bps=1.0)

    bounds = _window_bounds(len(df), wf.n_windows)
    last_start, last_end = bounds[-1]
    mutated = df.copy()
    mutated.iloc[last_start:last_end, mutated.columns.get_loc("close")] *= 5.0

    mutated_result = walk_forward_oos(mutated, strategy_config, wf, cost_bps=1.0)

    first_start, first_end = bounds[0]
    _, first_split, _ = _is_oos_split(first_start, first_end, wf.is_fraction)
    n_first_window_oos = first_end - first_split

    pd.testing.assert_series_equal(
        baseline.oos_returns.iloc[:n_first_window_oos],
        mutated_result.oos_returns.iloc[:n_first_window_oos],
    )


def test_recompute_per_window_warms_up_within_window_not_full_series() -> None:
    """A long-lookback strategy must warm up (position 0.0) at the start of
    *every* window when recomputed per-window, not just at the start of the
    full series. This is the concrete, observable difference between
    "recompute per window" (correct) and "compute once on the full series,
    then slice" (leaks pre-window history) that walk_forward_oos commits to.
    """
    lookback = 90
    config = StrategyConfig(
        name="tsm_90",
        family="time_series_momentum",
        fn=time_series_momentum,
        params={"lookback": lookback},
        category=Category.TREND,
    )
    df = _make_trend_df(700, seed=4)
    wf = WalkForwardConfig(n_windows=5, is_fraction=0.7)
    bounds = _window_bounds(len(df), wf.n_windows)

    # A later window, far from the start of the full series.
    start, end = bounds[2]
    window_df = df.iloc[start:end]

    positions_window_local = config.fn(window_df, **config.params)
    positions_full_series_sliced = config.fn(df, **config.params).iloc[start:end]

    # Window-local: the first `lookback` rows of the window are warmup (0.0).
    assert (positions_window_local.iloc[:lookback] == 0.0).all()
    # Full-series-sliced: this window has hundreds of rows of prior history,
    # so it is not in warmup here — at least one nonzero position appears
    # in what would otherwise be the window-local warmup region.
    assert (positions_full_series_sliced.iloc[:lookback] != 0.0).any()


def test_insufficient_history_raises() -> None:
    df = _make_trend_df(100, seed=5)
    wf = WalkForwardConfig(n_windows=5, is_fraction=0.7)
    config = StrategyConfig(
        name="ma_5_20",
        family="ma_crossover",
        fn=ma_crossover,
        params={"fast": 5, "slow": 20},
        category=Category.TREND,
    )
    with pytest.raises(InsufficientHistoryError):
        walk_forward_oos(df, config, wf, cost_bps=1.0)


def test_sufficient_history_does_not_raise(strategy_config: StrategyConfig) -> None:
    n_windows = 5
    # Construct the smallest series that should just barely pass: each
    # window's OOS segment must yield >= MIN_OOS_ROWS returns, i.e. >=
    # MIN_OOS_ROWS + 1 OOS rows per window.
    is_fraction = 0.7
    oos_rows_needed = MIN_OOS_ROWS + 1
    window_size = int(np.ceil(oos_rows_needed / (1 - is_fraction))) + 5
    n = window_size * n_windows
    df = _make_trend_df(n, seed=6)
    wf = WalkForwardConfig(n_windows=n_windows, is_fraction=is_fraction)
    result = walk_forward_oos(df, strategy_config, wf, cost_bps=1.0)
    assert len(result.oos_returns) > 0
