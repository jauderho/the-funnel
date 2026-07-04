"""Tests for the portfolio-level correlation matrix and redundancy flags (PRD §11.4)."""

import numpy as np
import pandas as pd
import pytest

from funnel.portfolio.correlation import correlation_matrix, redundancy_flags


def test_correlation_matrix_known_values() -> None:
    index = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(1)
    a = pd.Series(rng.normal(size=100), index=index)
    b = -a  # perfectly anti-correlated
    c = a.copy()  # perfectly correlated (identical)

    corr = correlation_matrix({"a": a, "b": b, "c": c}, min_overlap=10)

    assert corr.loc["a", "a"] == pytest.approx(1.0)
    assert corr.loc["a", "b"] == pytest.approx(-1.0)
    assert corr.loc["a", "c"] == pytest.approx(1.0)
    assert corr.loc["b", "c"] == pytest.approx(-1.0)
    # Symmetric
    assert corr.loc["b", "a"] == pytest.approx(corr.loc["a", "b"])


def test_correlation_matrix_below_min_overlap_is_nan() -> None:
    index_a = pd.bdate_range("2020-01-01", periods=100)
    index_b = pd.bdate_range("2020-01-01", periods=5)  # only 5 rows overlap
    a = pd.Series(np.arange(100, dtype="float64"), index=index_a)
    b = pd.Series(np.arange(5, dtype="float64"), index=index_b)

    corr = correlation_matrix({"a": a, "b": b}, min_overlap=60)
    assert pd.isna(corr.loc["a", "b"])
    assert pd.isna(corr.loc["b", "a"])
    # Self-correlation is unaffected by min_overlap.
    assert corr.loc["a", "a"] == pytest.approx(1.0)


def test_correlation_matrix_default_min_overlap_is_60() -> None:
    index = pd.bdate_range("2020-01-01", periods=59)
    rng = np.random.default_rng(2)
    a = pd.Series(rng.normal(size=59), index=index)
    b = pd.Series(rng.normal(size=59), index=index)

    corr = correlation_matrix({"a": a, "b": b})
    assert pd.isna(corr.loc["a", "b"])


def test_redundancy_flags_threshold_behavior() -> None:
    corr = pd.DataFrame(
        {
            "a": [1.0, 0.95, 0.5],
            "b": [0.95, 1.0, 0.86],
            "c": [0.5, 0.86, 1.0],
        },
        index=["a", "b", "c"],
    )

    flags = redundancy_flags(corr, threshold=0.85)
    pairs = set(zip(flags["a"], flags["b"], strict=False))
    assert ("a", "b") in pairs
    assert ("b", "c") in pairs
    assert len(flags) == 2
    # Sorted by corr descending.
    assert flags.iloc[0]["corr"] >= flags.iloc[1]["corr"]


def test_redundancy_flags_excludes_nan_pairs() -> None:
    corr = pd.DataFrame(
        {
            "a": [1.0, float("nan")],
            "b": [float("nan"), 1.0],
        },
        index=["a", "b"],
    )
    flags = redundancy_flags(corr, threshold=0.85)
    assert flags.empty


def test_redundancy_flags_below_threshold_excluded() -> None:
    corr = pd.DataFrame(
        {"a": [1.0, 0.5], "b": [0.5, 1.0]},
        index=["a", "b"],
    )
    flags = redundancy_flags(corr, threshold=0.85)
    assert flags.empty
