"""Tests for funnel.regime.label_cache: regime-label cache (PERF-2)."""

from pathlib import Path

import pandas as pd
import pytest

import funnel.regime.label_cache as label_cache_module
from funnel.compute_cache import evict_oldest
from funnel.regime.base import RegimeDetector
from funnel.regime.label_cache import classify_all_cached, fingerprint_regime_inputs
from funnel.regime.ma_filter import MAFilterDetector
from funnel.regime.realized_vol import RealizedVolDetector


def _detectors() -> dict[str, RegimeDetector]:
    return {"ma_filter": MAFilterDetector(), "realized_vol": RealizedVolDetector()}


def test_cache_miss_then_hit_matches_fresh_classify(
    tmp_path: Path, trending_ohlcv: pd.DataFrame
) -> None:
    detectors = _detectors()
    cache_dir = tmp_path / "cache"

    miss = classify_all_cached(trending_ohlcv, detectors, cache_dir)
    assert miss.cache_hit is False

    hit = classify_all_cached(trending_ohlcv, detectors, cache_dir)
    assert hit.cache_hit is True
    assert hit.fingerprint == miss.fingerprint

    for name, detector in detectors.items():
        fresh = detector.classify(trending_ohlcv)
        assert (hit.labels_by_detector[name].to_numpy() == fresh.to_numpy()).all()
        assert (miss.labels_by_detector[name].to_numpy() == fresh.to_numpy()).all()


def test_cache_writes_parquet_and_json_sidecar(
    tmp_path: Path, trending_ohlcv: pd.DataFrame
) -> None:
    cache_dir = tmp_path / "cache"
    result = classify_all_cached(trending_ohlcv, _detectors(), cache_dir)
    assert (cache_dir / f"regime_labels_{result.fingerprint}.parquet").is_file()
    assert (cache_dir / f"regime_labels_{result.fingerprint}.json").is_file()


def test_fingerprint_deterministic(trending_ohlcv: pd.DataFrame) -> None:
    detectors = _detectors()
    assert fingerprint_regime_inputs(trending_ohlcv, detectors) == fingerprint_regime_inputs(
        trending_ohlcv, detectors
    )


def test_fingerprint_changes_on_data_perturbation(trending_ohlcv: pd.DataFrame) -> None:
    detectors = _detectors()
    base = fingerprint_regime_inputs(trending_ohlcv, detectors)
    perturbed = trending_ohlcv.copy()
    perturbed.iloc[10, perturbed.columns.get_loc("close")] += 0.01
    assert fingerprint_regime_inputs(perturbed, detectors) != base


def test_fingerprint_changes_on_detector_param_change(trending_ohlcv: pd.DataFrame) -> None:
    base = fingerprint_regime_inputs(trending_ohlcv, {"ma_filter": MAFilterDetector(window=200)})
    changed = fingerprint_regime_inputs(trending_ohlcv, {"ma_filter": MAFilterDetector(window=50)})
    assert base != changed


def test_fingerprint_changes_on_schema_bump(
    monkeypatch: pytest.MonkeyPatch, trending_ohlcv: pd.DataFrame
) -> None:
    detectors = _detectors()
    base = fingerprint_regime_inputs(trending_ohlcv, detectors)
    monkeypatch.setattr(label_cache_module, "COMPUTE_CACHE_SCHEMA", 999)
    bumped = fingerprint_regime_inputs(trending_ohlcv, detectors)
    assert bumped != base


def test_fingerprint_changes_on_engine_version_bump(
    monkeypatch: pytest.MonkeyPatch, trending_ohlcv: pd.DataFrame
) -> None:
    detectors = _detectors()
    base = fingerprint_regime_inputs(trending_ohlcv, detectors)
    monkeypatch.setattr(label_cache_module, "FUNNEL_VERSION", "999.0.0")
    bumped = fingerprint_regime_inputs(trending_ohlcv, detectors)
    assert bumped != base


def test_evict_oldest_entries_past_cap(
    tmp_path: Path,
    trending_ohlcv: pd.DataFrame,
    mean_reverting_ohlcv: pd.DataFrame,
    flat_ohlcv: pd.DataFrame,
) -> None:
    cache_dir = tmp_path / "cache"
    detectors = _detectors()
    for df in (trending_ohlcv, mean_reverting_ohlcv, flat_ohlcv):
        classify_all_cached(df, detectors, cache_dir)

    evict_oldest(cache_dir, "regime_labels_*.parquet", keep=2)
    remaining = list(cache_dir.glob("regime_labels_*.parquet"))
    assert len(remaining) == 2
