"""Tests for the regime detection module (PRD §9).

Synthetic, network-free fixtures throughout. hmmlearn is available in this
environment, so the HMM detector is exercised directly rather than skipped.
"""

import numpy as np
import pandas as pd
import pytest

from funnel.regime.base import Regime, regime_conditioned_metrics
from funnel.regime.changepoint import ChangePointDetector
from funnel.regime.compare import (
    agreement_matrix,
    assemble_regime_performance,
    compare_detectors,
    compare_detectors_from_labels,
)
from funnel.regime.hmm import HMMDetector
from funnel.regime.ma_filter import MAFilterDetector
from funnel.regime.realized_vol import RealizedVolDetector

FIRST_HALF = 400
N_ROWS = 800
TRUNCATE_AT = 600


def _to_ohlcv(index: pd.DatetimeIndex, close: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    daily_range = np.abs(rng.normal(loc=0.5, scale=0.2, size=len(close))) + 0.05
    open_ = close + rng.normal(loc=0.0, scale=0.1, size=len(close))
    high = np.maximum(open_, close) + daily_range
    low = np.minimum(open_, close) - daily_range
    volume = np.abs(rng.normal(loc=1_000_000.0, scale=200_000.0, size=len(close)))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    ).astype("float64")


@pytest.fixture
def switching_ohlcv() -> pd.DataFrame:
    """First ~400 rows: strong trend, low vol. Next ~400 rows: flat, high-vol chop."""
    rng = np.random.default_rng(123)
    index = pd.bdate_range("2018-01-01", periods=N_ROWS)

    trend_part = (
        100.0
        + np.linspace(0.0, 80.0, FIRST_HALF)
        + np.cumsum(rng.normal(loc=0.0, scale=0.15, size=FIRST_HALF))
    )
    chop_start = trend_part[-1]
    chop_len = N_ROWS - FIRST_HALF
    t = np.arange(chop_len)
    # A pullback off the trend high that decays to a lower flat level and
    # oscillates in a bounded range around it, plus noise: a genuinely
    # range-bound, high-vol chop segment (not a wandering random walk, and
    # not a symmetric oscillation around the trend's own ending level,
    # which would cross a trailing SMA ~50/50 by construction regardless
    # of "choppiness").
    target_level = chop_start - 35.0
    decay = np.exp(-t / 100.0)
    level = target_level + (chop_start - target_level) * decay
    oscillation = 3.0 * np.sin(2 * np.pi * t / 20.0)
    chop_noise = rng.normal(loc=0.0, scale=1.5, size=chop_len)
    chop_part = level + oscillation + chop_noise

    close = np.concatenate([trend_part, chop_part])
    return _to_ohlcv(index, close, rng)


# --------------------------------------------------------------------------
# MA filter
# --------------------------------------------------------------------------


def test_ma_filter_labels_match_switching_fixture(switching_ohlcv: pd.DataFrame) -> None:
    detector = MAFilterDetector(window=50)
    labels = detector.classify(switching_ohlcv)

    warmup = 50
    first_half = labels.iloc[warmup:FIRST_HALF]
    second_half = labels.iloc[FIRST_HALF:]

    frac_trending_first = (first_half == Regime.TRENDING).mean()
    frac_choppy_second = (second_half == Regime.CHOPPY).mean()

    assert frac_trending_first >= 0.7
    assert frac_choppy_second >= 0.7


def test_ma_filter_truncation_invariant(switching_ohlcv: pd.DataFrame) -> None:
    detector = MAFilterDetector(window=50)
    full_labels = detector.classify(switching_ohlcv)
    truncated_labels = detector.classify(switching_ohlcv.iloc[:TRUNCATE_AT])

    pd.testing.assert_series_equal(
        full_labels.iloc[:TRUNCATE_AT], truncated_labels, check_names=False
    )


def test_ma_filter_no_nans_and_valid_values(switching_ohlcv: pd.DataFrame) -> None:
    labels = MAFilterDetector().classify(switching_ohlcv)
    assert not labels.isna().any()
    assert set(labels.unique()).issubset({Regime.TRENDING, Regime.CHOPPY})
    assert labels.index.equals(switching_ohlcv.index)


# --------------------------------------------------------------------------
# Realized vol
# --------------------------------------------------------------------------


def test_realized_vol_truncation_invariant(switching_ohlcv: pd.DataFrame) -> None:
    detector = RealizedVolDetector()
    full_labels = detector.classify(switching_ohlcv)
    truncated_labels = detector.classify(switching_ohlcv.iloc[:TRUNCATE_AT])

    pd.testing.assert_series_equal(
        full_labels.iloc[:TRUNCATE_AT], truncated_labels, check_names=False
    )


def test_realized_vol_no_nans_and_valid_values(switching_ohlcv: pd.DataFrame) -> None:
    labels = RealizedVolDetector().classify(switching_ohlcv)
    assert not labels.isna().any()
    assert set(labels.unique()).issubset({Regime.TRENDING, Regime.CHOPPY})


# --------------------------------------------------------------------------
# Change-point
# --------------------------------------------------------------------------


def test_changepoint_no_nans_valid_labels_and_warmup(switching_ohlcv: pd.DataFrame) -> None:
    detector = ChangePointDetector(min_train=60, refit_every=21)
    labels = detector.classify(switching_ohlcv)

    assert not labels.isna().any()
    assert set(labels.unique()).issubset({Regime.TRENDING, Regime.CHOPPY})
    assert (labels.iloc[:60] == Regime.CHOPPY).all()


def test_changepoint_deterministic(switching_ohlcv: pd.DataFrame) -> None:
    detector = ChangePointDetector(min_train=60, refit_every=21)
    labels_a = detector.classify(switching_ohlcv)
    labels_b = detector.classify(switching_ohlcv)
    pd.testing.assert_series_equal(labels_a, labels_b)


def test_changepoint_halves_differ_materially(switching_ohlcv: pd.DataFrame) -> None:
    detector = ChangePointDetector(min_train=60, refit_every=21)
    labels = detector.classify(switching_ohlcv)

    first_half = labels.iloc[60:FIRST_HALF]
    second_half = labels.iloc[FIRST_HALF:]
    frac_trending_first = (first_half == Regime.TRENDING).mean()
    frac_trending_second = (second_half == Regime.TRENDING).mean()

    assert abs(frac_trending_first - frac_trending_second) >= 0.3


def test_changepoint_max_window_none_matches_omitting_the_parameter(
    switching_ohlcv: pd.DataFrame,
) -> None:
    """PERF-1: ``max_window=None`` (the default) must be identical to
    omitting the parameter entirely — the equivalence guarantee that lets
    this knob ship without changing any existing caller's output."""
    default = ChangePointDetector(min_train=60, refit_every=21)
    explicit_none = ChangePointDetector(min_train=60, refit_every=21, max_window=None)
    pd.testing.assert_series_equal(
        default.classify(switching_ohlcv), explicit_none.classify(switching_ohlcv)
    )


def test_changepoint_max_window_bounds_the_window(switching_ohlcv: pd.DataFrame) -> None:
    """Setting ``max_window`` makes the window rolling instead of expanding
    (a real, working knob) while preserving the detector's warmup/no-NaN/
    valid-label contract. It is a genuine semantic change (PELT sees a
    different, shorter signal) so it is not compared for equality against
    the unbounded default here — only opt-in behavior is deferred to the
    caller, per PERF-1's identical-by-default constraint."""
    capped = ChangePointDetector(min_train=60, refit_every=21, max_window=100)
    labels = capped.classify(switching_ohlcv)

    assert not labels.isna().any()
    assert set(labels.unique()).issubset({Regime.TRENDING, Regime.CHOPPY})
    assert (labels.iloc[:60] == Regime.CHOPPY).all()


# --------------------------------------------------------------------------
# HMM
# --------------------------------------------------------------------------


def test_hmm_no_nans_valid_labels_and_warmup(switching_ohlcv: pd.DataFrame) -> None:
    detector = HMMDetector(min_train=252, refit_every=63, seed=0)
    labels = detector.classify(switching_ohlcv)

    assert not labels.isna().any()
    assert set(labels.unique()).issubset({Regime.TRENDING, Regime.CHOPPY})
    assert (labels.iloc[:252] == Regime.CHOPPY).all()


def test_hmm_deterministic(switching_ohlcv: pd.DataFrame) -> None:
    detector = HMMDetector(min_train=252, refit_every=63, seed=0)
    labels_a = detector.classify(switching_ohlcv)
    labels_b = detector.classify(switching_ohlcv)
    pd.testing.assert_series_equal(labels_a, labels_b)


def test_hmm_halves_differ_materially(switching_ohlcv: pd.DataFrame) -> None:
    detector = HMMDetector(min_train=252, refit_every=63, seed=0)
    labels = detector.classify(switching_ohlcv)

    first_half = labels.iloc[252:FIRST_HALF]
    second_half = labels.iloc[FIRST_HALF:]
    assert len(first_half) > 0
    assert len(second_half) > 0

    frac_trending_first = (first_half == Regime.TRENDING).mean()
    frac_trending_second = (second_half == Regime.TRENDING).mean()

    assert abs(frac_trending_first - frac_trending_second) >= 0.3


def test_hmm_warmup_below_min_train_is_choppy() -> None:
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2020-01-01", periods=100)
    close = 100.0 + np.cumsum(rng.normal(size=100))
    df = _to_ohlcv(index, close, rng)

    detector = HMMDetector(min_train=252, refit_every=63, seed=0)
    labels = detector.classify(df)
    assert (labels == Regime.CHOPPY).all()


# --------------------------------------------------------------------------
# regime_conditioned_metrics
# --------------------------------------------------------------------------


def test_regime_conditioned_metrics_matches_masked_subsets() -> None:
    index = pd.bdate_range("2021-01-01", periods=20)
    rng = np.random.default_rng(5)
    returns = pd.Series(rng.normal(loc=0.001, scale=0.01, size=20), index=index)
    labels = pd.Series([Regime.TRENDING] * 10 + [Regime.CHOPPY] * 10, index=index, dtype=object)

    result = regime_conditioned_metrics(returns, labels)

    from funnel.backtest.metrics import max_drawdown, sharpe

    trending_subset = returns.iloc[:10]
    choppy_subset = returns.iloc[10:]

    assert result[Regime.TRENDING].sharpe == pytest.approx(sharpe(trending_subset))
    assert result[Regime.TRENDING].max_drawdown == pytest.approx(max_drawdown(trending_subset))
    assert result[Regime.TRENDING].n_days == 10

    assert result[Regime.CHOPPY].sharpe == pytest.approx(sharpe(choppy_subset))
    assert result[Regime.CHOPPY].max_drawdown == pytest.approx(max_drawdown(choppy_subset))
    assert result[Regime.CHOPPY].n_days == 10


def test_regime_conditioned_metrics_profitable_only_in_trending() -> None:
    index = pd.bdate_range("2021-01-01", periods=40)
    labels = pd.Series([Regime.TRENDING] * 20 + [Regime.CHOPPY] * 20, index=index, dtype=object)
    # Positive-drift, noisy returns while TRENDING (nonzero std -> a real
    # Sharpe, not the constant-series 0.0 special case), alternating
    # (net ~flat/negative) while CHOPPY.
    trending_returns = [0.012, 0.008, 0.011, 0.009] * 5
    choppy_returns = [0.01, -0.012] * 10
    returns = pd.Series(trending_returns + choppy_returns, index=index)

    result = regime_conditioned_metrics(returns, labels)

    assert result[Regime.TRENDING].sharpe > 0
    assert result[Regime.CHOPPY].sharpe <= 0


def test_regime_conditioned_metrics_ffills_misaligned_labels() -> None:
    """Regime labels on a coarser/offset index are ffilled onto the returns index."""
    returns_index = pd.bdate_range("2021-01-01", periods=5)
    returns = pd.Series([0.01, 0.01, -0.01, 0.02, -0.005], index=returns_index)

    label_index = returns_index[[0, 2]]
    labels = pd.Series([Regime.TRENDING, Regime.CHOPPY], index=label_index, dtype=object)

    result = regime_conditioned_metrics(returns, labels)

    # Day 0-1 ffill to TRENDING, days 2-4 ffill to CHOPPY.
    assert result[Regime.TRENDING].n_days == 2
    assert result[Regime.CHOPPY].n_days == 3


def test_regime_conditioned_metrics_omits_regime_with_zero_days() -> None:
    index = pd.bdate_range("2021-01-01", periods=5)
    returns = pd.Series([0.01, 0.02, -0.01, 0.005, 0.01], index=index)
    labels = pd.Series([Regime.TRENDING] * 5, index=index, dtype=object)

    result = regime_conditioned_metrics(returns, labels)
    assert Regime.TRENDING in result
    assert Regime.CHOPPY not in result


# --------------------------------------------------------------------------
# compare / agreement
# --------------------------------------------------------------------------


def test_agreement_matrix_self_agreement_is_one() -> None:
    index = pd.bdate_range("2021-01-01", periods=10)
    labels = pd.Series([Regime.TRENDING, Regime.CHOPPY] * 5, index=index, dtype=object)
    matrix = agreement_matrix({"a": labels})
    assert matrix.loc["a", "a"] == pytest.approx(1.0)


def test_agreement_matrix_pairwise() -> None:
    index = pd.bdate_range("2021-01-01", periods=4)
    a = pd.Series(
        [Regime.TRENDING, Regime.TRENDING, Regime.CHOPPY, Regime.CHOPPY], index=index, dtype=object
    )
    b = pd.Series(
        [Regime.TRENDING, Regime.CHOPPY, Regime.CHOPPY, Regime.CHOPPY], index=index, dtype=object
    )
    matrix = agreement_matrix({"a": a, "b": b})
    assert matrix.loc["a", "b"] == pytest.approx(0.75)
    assert matrix.loc["b", "a"] == pytest.approx(0.75)
    assert matrix.loc["a", "a"] == pytest.approx(1.0)
    assert matrix.loc["b", "b"] == pytest.approx(1.0)


def test_compare_detectors_switch_counts_on_hand_built_labels() -> None:
    index = pd.bdate_range("2021-01-01", periods=6)
    labels = pd.Series(
        [
            Regime.CHOPPY,
            Regime.CHOPPY,
            Regime.TRENDING,
            Regime.TRENDING,
            Regime.CHOPPY,
            Regime.TRENDING,
        ],
        index=index,
        dtype=object,
    )

    class _FixedDetector:
        def classify(self, df: pd.DataFrame) -> pd.Series:
            return labels

    df = pd.DataFrame({"close": np.arange(6.0)}, index=index)
    result = compare_detectors(df, {"fixed": _FixedDetector()})

    row = result.iloc[0]
    assert row["detector"] == "fixed"
    assert row["n_switches"] == 3  # choppy->trending, trending->choppy, choppy->trending
    assert row["fraction_trending"] == pytest.approx(3 / 6)
    assert row["mean_spell_length"] == pytest.approx(6 / 4)


def test_compare_detectors_from_labels_matches_compare_detectors() -> None:
    """PERF-1: ``funnel.pipeline`` calls ``classify()`` once per detector and
    derives the comparison table via ``compare_detectors_from_labels``
    instead of calling ``compare_detectors`` (which would call ``classify``
    a second, redundant time — expensive for detectors like
    ``ChangePointDetector``). The two entry points must produce identical
    output for the same labels."""
    index = pd.bdate_range("2021-01-01", periods=6)
    labels = pd.Series(
        [
            Regime.CHOPPY,
            Regime.CHOPPY,
            Regime.TRENDING,
            Regime.TRENDING,
            Regime.CHOPPY,
            Regime.TRENDING,
        ],
        index=index,
        dtype=object,
    )

    class _FixedDetector:
        def classify(self, df: pd.DataFrame) -> pd.Series:
            return labels

    df = pd.DataFrame({"close": np.arange(6.0)}, index=index)
    via_detectors = compare_detectors(df, {"fixed": _FixedDetector()})
    via_labels = compare_detectors_from_labels({"fixed": labels})

    pd.testing.assert_frame_equal(via_detectors, via_labels)


def test_assemble_regime_performance_schema() -> None:
    index = pd.bdate_range("2021-01-01", periods=10)
    returns = pd.Series(np.full(10, 0.01), index=index)
    labels = pd.Series([Regime.TRENDING] * 5 + [Regime.CHOPPY] * 5, index=index, dtype=object)

    frame = assemble_regime_performance({"strategy_a": (returns, labels)})

    assert list(frame.columns) == [
        "name",
        "regime",
        "sharpe",
        "max_drawdown",
        "n_days",
        "total_return",
    ]
    assert set(frame["name"]) == {"strategy_a"}
    assert set(frame["regime"]) == {"trending", "choppy"}
