"""Tests for the six-filter survival funnel: one fixture per filter, isolating failures."""

import pandas as pd

from funnel.backtest.funnel import apply_funnel
from funnel.backtest.walkforward import WalkForwardResult
from funnel.config import FunnelThresholds

THRESHOLDS = FunnelThresholds(
    max_dd_floor=-0.35,
    min_oos_sharpe=0.5,
    max_oos_sharpe=2.5,
    max_oos_is_ratio=1.3,
    min_trades=30,
    require_positive_is_sharpe=True,
)


def _result(
    is_sharpe: float = 1.0,
    oos_sharpe: float = 1.0,
    oos_max_drawdown: float = -0.10,
    oos_trade_count: int = 50,
) -> WalkForwardResult:
    empty = pd.Series([], dtype="float64")
    return WalkForwardResult(
        is_sharpe=is_sharpe,
        oos_sharpe=oos_sharpe,
        oos_max_drawdown=oos_max_drawdown,
        oos_trade_count=oos_trade_count,
        oos_returns=empty,
        is_returns=empty,
    )


def test_passes_all_six_filters() -> None:
    result = _result(is_sharpe=1.0, oos_sharpe=1.0, oos_max_drawdown=-0.10, oos_trade_count=50)
    verdict = apply_funnel(result, THRESHOLDS)
    assert verdict.passes_max_dd_floor
    assert verdict.passes_min_oos_sharpe
    assert verdict.passes_max_oos_sharpe
    assert verdict.passes_overfit_gap
    assert verdict.passes_min_trades
    assert verdict.passes_positive_is_sharpe
    assert verdict.survived


def test_fails_only_max_dd_floor() -> None:
    # Deep drawdown (-0.50, worse than -0.35 floor); everything else passes.
    result = _result(is_sharpe=1.0, oos_sharpe=1.0, oos_max_drawdown=-0.50, oos_trade_count=50)
    verdict = apply_funnel(result, THRESHOLDS)
    assert not verdict.passes_max_dd_floor
    assert verdict.passes_min_oos_sharpe
    assert verdict.passes_max_oos_sharpe
    assert verdict.passes_overfit_gap
    assert verdict.passes_min_trades
    assert verdict.passes_positive_is_sharpe
    assert not verdict.survived


def test_fails_only_min_oos_sharpe() -> None:
    # oos_sharpe below the 0.5 floor; is_sharpe kept low enough that the
    # overfit-gap filter still passes (oos <= is * 1.3).
    result = _result(is_sharpe=0.3, oos_sharpe=0.2, oos_max_drawdown=-0.10, oos_trade_count=50)
    verdict = apply_funnel(result, THRESHOLDS)
    assert verdict.passes_max_dd_floor
    assert not verdict.passes_min_oos_sharpe
    assert verdict.passes_max_oos_sharpe
    assert verdict.passes_overfit_gap
    assert verdict.passes_min_trades
    assert verdict.passes_positive_is_sharpe
    assert not verdict.survived


def test_fails_only_max_oos_sharpe() -> None:
    # Implausibly high OOS Sharpe (3.0 > 2.5 ceiling); is_sharpe set high
    # enough that the overfit-gap ratio still passes.
    result = _result(is_sharpe=3.0, oos_sharpe=3.0, oos_max_drawdown=-0.10, oos_trade_count=50)
    verdict = apply_funnel(result, THRESHOLDS)
    assert verdict.passes_max_dd_floor
    assert verdict.passes_min_oos_sharpe
    assert not verdict.passes_max_oos_sharpe
    assert verdict.passes_overfit_gap
    assert verdict.passes_min_trades
    assert verdict.passes_positive_is_sharpe
    assert not verdict.survived


def test_fails_only_overfit_gap() -> None:
    # oos_sharpe (2.0) is within [0.5, 2.5] but far exceeds is_sharpe (0.6) * 1.3 = 0.78.
    result = _result(is_sharpe=0.6, oos_sharpe=2.0, oos_max_drawdown=-0.10, oos_trade_count=50)
    verdict = apply_funnel(result, THRESHOLDS)
    assert verdict.passes_max_dd_floor
    assert verdict.passes_min_oos_sharpe
    assert verdict.passes_max_oos_sharpe
    assert not verdict.passes_overfit_gap
    assert verdict.passes_min_trades
    assert verdict.passes_positive_is_sharpe
    assert not verdict.survived


def test_fails_only_min_trades() -> None:
    result = _result(is_sharpe=1.0, oos_sharpe=1.0, oos_max_drawdown=-0.10, oos_trade_count=10)
    verdict = apply_funnel(result, THRESHOLDS)
    assert verdict.passes_max_dd_floor
    assert verdict.passes_min_oos_sharpe
    assert verdict.passes_max_oos_sharpe
    assert verdict.passes_overfit_gap
    assert not verdict.passes_min_trades
    assert verdict.passes_positive_is_sharpe
    assert not verdict.survived


def test_fails_only_positive_is_sharpe() -> None:
    # is_sharpe <= 0: filter 6 fails. Note filter 4 (overfit gap) also fails
    # by definition when is_sharpe <= 0 (documented rule) — both booleans
    # are recorded, but this fixture's *intent* is to isolate filter 6, so
    # oos_sharpe/oos_max_drawdown/trades are kept passing everywhere else.
    result = _result(is_sharpe=-0.2, oos_sharpe=1.0, oos_max_drawdown=-0.10, oos_trade_count=50)
    verdict = apply_funnel(result, THRESHOLDS)
    assert verdict.passes_max_dd_floor
    assert verdict.passes_min_oos_sharpe
    assert verdict.passes_max_oos_sharpe
    assert not verdict.passes_overfit_gap  # is_sharpe <= 0 -> gap filter fails too, by rule
    assert verdict.passes_min_trades
    assert not verdict.passes_positive_is_sharpe
    assert not verdict.survived


def test_all_six_outcomes_recorded_even_after_first_failure() -> None:
    # Every filter fails simultaneously; all six booleans must still be
    # individually present and False (not short-circuited).
    result = _result(is_sharpe=-1.0, oos_sharpe=-1.0, oos_max_drawdown=-0.90, oos_trade_count=0)
    verdict = apply_funnel(result, THRESHOLDS)
    assert verdict.passes_max_dd_floor is False
    assert verdict.passes_min_oos_sharpe is False
    assert verdict.passes_max_oos_sharpe is True  # -1.0 < 2.5 ceiling, so this one passes
    assert verdict.passes_overfit_gap is False
    assert verdict.passes_min_trades is False
    assert verdict.passes_positive_is_sharpe is False
    assert verdict.survived is False


def test_overfit_gap_rule_boundary_is_inclusive() -> None:
    # oos_sharpe == is_sharpe * max_oos_is_ratio exactly -> passes (<=).
    is_sharpe = 1.0
    oos_sharpe = is_sharpe * THRESHOLDS.max_oos_is_ratio
    result = _result(
        is_sharpe=is_sharpe, oos_sharpe=oos_sharpe, oos_max_drawdown=-0.10, oos_trade_count=50
    )
    verdict = apply_funnel(result, THRESHOLDS)
    assert verdict.passes_overfit_gap


def test_require_positive_is_sharpe_false_skips_filter_six() -> None:
    thresholds = FunnelThresholds(require_positive_is_sharpe=False)
    result = _result(is_sharpe=-0.5, oos_sharpe=1.0, oos_max_drawdown=-0.10, oos_trade_count=50)
    verdict = apply_funnel(result, thresholds)
    assert verdict.passes_positive_is_sharpe
    # Overfit gap still fails since is_sharpe <= 0 regardless of the flag.
    assert not verdict.passes_overfit_gap
