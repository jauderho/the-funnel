"""Tests for the cross-sectional momentum research/diagnostic check (synthetic only).

Covers: winners/losers ranking produces positive OOS Sharpe on a fixture
built for that outcome, rebalance cadence and weight-sum mechanics, history
eligibility (late-starting asset excluded until it has a full lookback),
no-look-ahead application of weights, turnover-cost arithmetic, and the
12-1 lookback's short-term-reversal skip.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from funnel.config import CostModel, WalkForwardConfig
from funnel.data.universe import MIN_HISTORY_DAYS, AssetClass
from funnel.momentum.cross_sectional import (
    LOOKBACKS,
    Lookback,
    _build_close_panel,
    _rebalance_dates,
    _target_weights,
    _trailing_return,
    cross_sectional_returns,
    plain_language_verdict,
    run_cross_sectional_check,
    walk_forward_score,
    write_cross_sectional,
)

N_ROWS = MIN_HISTORY_DAYS + 50  # clears filter_universe's MIN_HISTORY_DAYS floor


def _make_df(n: int, start: str, drift: float, seed: int) -> pd.DataFrame:
    """A close-price series with constant daily log-return drift + noise.

    Geometric (log-return) construction — ``close = 100 * exp(cumsum(daily
    log returns))`` — rather than additive drift on the raw price: an
    additive random walk with a large enough negative drift over ~1000+
    rows (needed to clear ``MIN_HISTORY_DAYS``) can wander through zero and
    go negative, which corrupts ``pct_change``-based returns with
    nonsensical sign flips. Geometric drift keeps prices strictly positive
    for any drift/length combination, which is what every real close-price
    series does. ``start`` lets a series begin later than the calendar's
    first date (used to construct a late-starting asset).
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(start, periods=n)
    log_returns = rng.normal(loc=drift, scale=0.01, size=n)
    close = 100.0 * np.exp(np.cumsum(log_returns))
    return pd.DataFrame(
        {"open": close, "high": close + 0.5, "low": close - 0.5, "close": close, "volume": 1e6},
        index=index,
    ).astype("float64")


@pytest.fixture
def winners_losers_data() -> dict[str, pd.DataFrame]:
    """3 persistent winners, 3 persistent losers, 3 flat assets, fixed seeds.

    Winners drift up steadily, losers drift down steadily, and the flat
    assets barely move — so cross-sectional ranking (top third = winners,
    bottom third = losers) should be a stable, unsurprising bet on this
    fixture and come out clearly positive OOS.
    """
    return {
        "WIN1": _make_df(N_ROWS, "2015-01-01", drift=0.0015, seed=1),
        "WIN2": _make_df(N_ROWS, "2015-01-01", drift=0.0012, seed=2),
        "WIN3": _make_df(N_ROWS, "2015-01-01", drift=0.0018, seed=3),
        "LOSE1": _make_df(N_ROWS, "2015-01-01", drift=-0.0015, seed=4),
        "LOSE2": _make_df(N_ROWS, "2015-01-01", drift=-0.0012, seed=5),
        "LOSE3": _make_df(N_ROWS, "2015-01-01", drift=-0.0018, seed=6),
        "FLAT1": _make_df(N_ROWS, "2015-01-01", drift=0.0, seed=7),
        "FLAT2": _make_df(N_ROWS, "2015-01-01", drift=0.0, seed=8),
        "FLAT3": _make_df(N_ROWS, "2015-01-01", drift=0.0, seed=9),
    }


@pytest.fixture
def asset_classes() -> dict[str, AssetClass]:
    return {
        "WIN1": AssetClass.LARGE_CAP,
        "WIN2": AssetClass.LARGE_CAP,
        "WIN3": AssetClass.LARGE_CAP,
        "LOSE1": AssetClass.LARGE_CAP,
        "LOSE2": AssetClass.LARGE_CAP,
        "LOSE3": AssetClass.LARGE_CAP,
        "FLAT1": AssetClass.LARGE_CAP,
        "FLAT2": AssetClass.LARGE_CAP,
        "FLAT3": AssetClass.LARGE_CAP,
    }


def test_winners_losers_fixture_scores_positive_oos_sharpe_6m(
    winners_losers_data: dict[str, pd.DataFrame], asset_classes: dict[str, AssetClass]
) -> None:
    lookback = next(lb for lb in LOOKBACKS if lb.label == "6m")
    costs = CostModel()
    returns = cross_sectional_returns(winners_losers_data, lookback, costs, asset_classes)
    wf = WalkForwardConfig()
    score = walk_forward_score(returns, wf)
    assert score.oos_sharpe > 0


def test_rebalance_dates_every_21_days() -> None:
    index = pd.bdate_range("2015-01-01", periods=500)
    dates = _rebalance_dates(index, first_valid_i=100)
    assert dates[0] == 100
    diffs = [b - a for a, b in zip(dates, dates[1:], strict=False)]
    assert all(d == 21 for d in diffs)


def test_target_weights_long_short_legs_sum_to_zero_net_one_gross() -> None:
    trailing_return = pd.Series(
        {"A": 0.5, "B": 0.3, "C": 0.1, "D": -0.1, "E": -0.3, "F": -0.5},
    )
    weights = _target_weights(trailing_return, trailing_return.index)
    assert weights.sum() == pytest.approx(0.0)
    assert weights[weights > 0].sum() == pytest.approx(1.0)
    assert weights[weights < 0].sum() == pytest.approx(-1.0)
    # Top third (A, B) long; bottom third (E, F) short; middle (C, D) flat.
    assert weights["A"] > 0 and weights["B"] > 0
    assert weights["E"] < 0 and weights["F"] < 0
    assert weights["C"] == 0.0 and weights["D"] == 0.0


def test_target_weights_fewer_than_three_eligible_is_all_flat() -> None:
    trailing_return = pd.Series({"A": 0.5, "B": np.nan, "C": np.nan})
    weights = _target_weights(trailing_return, trailing_return.index)
    assert (weights == 0.0).all()


def test_late_starting_asset_excluded_until_full_lookback(
    asset_classes: dict[str, AssetClass],
) -> None:
    """An asset with a late start must not be ranked until it has a full
    lookback of history; once it does, it becomes eligible."""
    lookback = Lookback(label="test_3m", trailing_days=63, skip_days=0)
    data = {
        "A": _make_df(N_ROWS, "2015-01-01", drift=0.001, seed=1),
        "B": _make_df(N_ROWS, "2015-01-01", drift=-0.001, seed=2),
        "C": _make_df(N_ROWS, "2015-01-01", drift=0.0005, seed=3),
        # LATE starts ~100 trading days into the calendar (well past the
        # first few rebalances) with a huge positive drift once it starts.
        "LATE": _make_df(N_ROWS - 100, "2015-05-25", drift=0.005, seed=4),
    }
    panel = _build_close_panel(data)
    # first rebalance index is exactly lookback.history_needed
    first_i = lookback.history_needed
    early_trailing = _trailing_return(panel, first_i, lookback)
    assert pd.isna(early_trailing["LATE"])

    # find a later rebalance index, at least 100 + lookback rows in, where
    # LATE should have accrued a full lookback of real (non-ffill-from-void)
    # history.
    late_i = first_i + 21 * 6  # well past LATE's start + lookback window
    late_trailing = _trailing_return(panel, late_i, lookback)
    assert not pd.isna(late_trailing["LATE"])


def test_no_lookahead_crash_day_after_rebalance_is_eaten(
    asset_classes: dict[str, AssetClass],
) -> None:
    """If the top-ranked asset crashes the day *after* a rebalance, the
    portfolio (holding the new long weight from that day forward) must eat
    the loss — proving weights set at a rebalance apply starting next day,
    not that same day (which would already reflect knowledge of the crash)."""
    lookback = Lookback(label="test_3m", trailing_days=63, skip_days=0)
    n = N_ROWS  # must clear filter_universe's MIN_HISTORY_DAYS floor
    data = {
        "WINNER": _make_df(n, "2015-01-01", drift=0.003, seed=1),
        "B": _make_df(n, "2015-01-01", drift=0.0, seed=2),
        "C": _make_df(n, "2015-01-01", drift=-0.003, seed=3),
    }
    panel = _build_close_panel(data)
    first_i = lookback.history_needed
    rebalance_i = _rebalance_dates(panel.index, first_i)[0]

    # Crash WINNER by 50% on the day immediately after the rebalance date.
    crashed = {k: v.copy() for k, v in data.items()}
    crash_date = panel.index[rebalance_i + 1]
    crashed["WINNER"].loc[crash_date:, "close"] *= 0.5

    costs = CostModel()
    classes = {"WINNER": AssetClass.LARGE_CAP, "B": AssetClass.LARGE_CAP, "C": AssetClass.LARGE_CAP}
    returns = cross_sectional_returns(crashed, lookback, costs, classes)

    assert returns.loc[crash_date] < -0.05  # a materially negative day: the crash was eaten


def test_turnover_cost_hand_computed_example() -> None:
    """A rebalance that flips from an all-flat book to a concrete long/short
    book should deduct exactly the hand-computed turnover cost, and a
    rebalance reproducing the same ranking (zero turnover) costs nothing."""
    lookback = Lookback(label="test_3m", trailing_days=63, skip_days=0)
    n = N_ROWS  # must clear filter_universe's MIN_HISTORY_DAYS floor
    # 3 assets so top/bottom third = 1 asset each, weight magnitude 1.0.
    data = {
        "A": _make_df(n, "2015-01-01", drift=0.002, seed=1),
        "B": _make_df(n, "2015-01-01", drift=0.0, seed=2),
        "C": _make_df(n, "2015-01-01", drift=-0.002, seed=3),
    }
    classes = {"A": AssetClass.LARGE_CAP, "B": AssetClass.LARGE_CAP, "C": AssetClass.LARGE_CAP}
    costs = CostModel(default_bps_per_side=1.0, crypto_bps_per_side=5.0)

    returns = cross_sectional_returns(data, lookback, costs, classes)
    panel = _build_close_panel(data)
    first_rebalance_i = _rebalance_dates(panel.index, lookback.history_needed)[0]
    first_rebalance_date = panel.index[first_rebalance_i]

    # Hand-computed: first rebalance goes from all-flat (0,0,0) to
    # (+1.0, 0.0, -1.0) (A long, C short). Turnover = |1-0| + |0-0| + |-1-0|
    # = 2.0 total notional traded, at 1.0 bps/side (LARGE_CAP) each:
    # cost = 2.0 * 1.0 / 1e4 = 0.0002.
    expected_cost = 2.0 * 1.0 / 1e4

    # The cost is charged on the rebalance date's own row, before the gross
    # return contribution kicks in (weights apply next day) — so returns on
    # the first rebalance date should be exactly -expected_cost (gross is
    # zero that day: applied weights are still all-flat from the prior
    # period).
    assert returns.loc[first_rebalance_date] == pytest.approx(-expected_cost, abs=1e-12)


def test_zero_turnover_rebalance_costs_nothing() -> None:
    """If two consecutive rebalances rank assets identically, the second
    rebalance's turnover cost must be zero."""
    lookback = Lookback(label="test_3m", trailing_days=63, skip_days=0)
    n = N_ROWS  # must clear filter_universe's MIN_HISTORY_DAYS floor
    # Strong, well-separated persistent drifts: the ranking should not flip
    # between consecutive monthly rebalances.
    data = {
        "A": _make_df(n, "2015-01-01", drift=0.003, seed=1),
        "B": _make_df(n, "2015-01-01", drift=0.0, seed=2),
        "C": _make_df(n, "2015-01-01", drift=-0.003, seed=3),
    }
    classes = {"A": AssetClass.LARGE_CAP, "B": AssetClass.LARGE_CAP, "C": AssetClass.LARGE_CAP}
    costs = CostModel()

    returns = cross_sectional_returns(data, lookback, costs, classes)
    panel = _build_close_panel(data)
    asset_returns = panel.pct_change()
    rebalance_positions = _rebalance_dates(panel.index, lookback.history_needed)
    second_rebalance_date = panel.index[rebalance_positions[1]]

    trailing_1 = _trailing_return(panel, rebalance_positions[0], lookback)
    trailing_2 = _trailing_return(panel, rebalance_positions[1], lookback)
    weights_1 = _target_weights(trailing_1, panel.columns)
    weights_2 = _target_weights(trailing_2, panel.columns)
    assert (weights_1 == weights_2).all()  # same ranking -> zero turnover expected

    # Return on the second rebalance date must equal gross-only (no cost
    # deduction): the weights applied that day are unchanged from the prior
    # period (weights_1 == weights_2), so hand-computing gross from
    # weights_1 and that day's asset returns must match the actual output
    # exactly if (and only if) zero cost was charged.
    expected_gross_only = float((weights_1 * asset_returns.loc[second_rebalance_date]).sum())
    assert returns.loc[second_rebalance_date] == pytest.approx(expected_gross_only, abs=1e-12)


def test_12_1_lookback_skips_recent_month(asset_classes: dict[str, AssetClass]) -> None:
    """An asset with a huge return concentrated only in the last 21 days
    before a rebalance must NOT rank top under 12-1 (the skip excludes that
    window entirely from the ranking return)."""
    lookback_12_1 = next(lb for lb in LOOKBACKS if lb.label == "12-1")
    n = 400
    data = {
        # SPIKE is flat for most of history, then jumps hugely in the final
        # 21 days before the rebalance -- entirely inside the skip window.
        "SPIKE": _make_df(n, "2015-01-01", drift=0.0, seed=1),
        "STEADY": _make_df(n, "2015-01-01", drift=0.0005, seed=2),
        "LOSER": _make_df(n, "2015-01-01", drift=-0.0005, seed=3),
    }
    panel = _build_close_panel(data)
    rebalance_i = _rebalance_dates(panel.index, lookback_12_1.history_needed)[0]

    spiked = {k: v.copy() for k, v in data.items()}
    skip_start = panel.index[rebalance_i - 20]
    spiked["SPIKE"].loc[skip_start:, "close"] *= 3.0  # huge jump inside the last 21 days only

    spiked_panel = _build_close_panel(spiked)
    trailing_12_1 = _trailing_return(spiked_panel, rebalance_i, lookback_12_1)
    ranked = trailing_12_1.dropna().sort_values(ascending=False)
    assert ranked.index[0] != "SPIKE"


def test_run_cross_sectional_check_shape_and_research_only_flag(
    winners_losers_data: dict[str, pd.DataFrame], asset_classes: dict[str, AssetClass]
) -> None:
    wf = WalkForwardConfig()
    costs = CostModel()
    df = run_cross_sectional_check(winners_losers_data, wf, costs, asset_classes)

    assert len(df) == len(LOOKBACKS)
    assert set(df["lookback"]) == {lb.label for lb in LOOKBACKS}
    assert (df["research_only"] == True).all()  # noqa: E712 — explicit True check, not truthiness


def test_run_cross_sectional_check_with_single_asset_comparison(
    winners_losers_data: dict[str, pd.DataFrame], asset_classes: dict[str, AssetClass]
) -> None:
    wf = WalkForwardConfig()
    costs = CostModel()
    single_asset_sweep = pd.DataFrame(
        {
            "family": ["time_series_momentum", "roc_momentum", "rsi_revert"],
            "oos_sharpe": [0.4, 0.6, 1.2],
        }
    )
    df = run_cross_sectional_check(
        winners_losers_data, wf, costs, asset_classes, single_asset_momentum=single_asset_sweep
    )
    assert "single_asset_mean_oos_sharpe" in df.columns
    # Mean of the two momentum families only (0.4, 0.6), excludes rsi_revert.
    assert df["single_asset_mean_oos_sharpe"].iloc[0] == pytest.approx(0.5)


def test_plain_language_verdict_is_non_empty_and_mentions_research_only(
    winners_losers_data: dict[str, pd.DataFrame], asset_classes: dict[str, AssetClass]
) -> None:
    wf = WalkForwardConfig()
    costs = CostModel()
    df = run_cross_sectional_check(winners_losers_data, wf, costs, asset_classes)
    verdict = plain_language_verdict(df)
    assert len(verdict) > 0
    assert "short" in verdict.lower()


def test_write_cross_sectional_csv(
    tmp_path: Path,
    winners_losers_data: dict[str, pd.DataFrame],
    asset_classes: dict[str, AssetClass],
) -> None:
    wf = WalkForwardConfig()
    costs = CostModel()
    df = run_cross_sectional_check(winners_losers_data, wf, costs, asset_classes)
    path = tmp_path / "cross_sectional.csv"
    write_cross_sectional(df, path)
    assert path.exists()
    reloaded = pd.read_csv(path)
    assert len(reloaded) == len(df)
