"""Regime-label cache (PERF-2): same logic as
``funnel.backtest.sweep_cache`` applied to the regime stage.

Every detector's ``classify(proxy_df)`` output depends only on the proxy
symbol's OHLCV content, the detector's own parameters, and the engine
version/schema (``funnel.compute_cache``) -- never on ``FunnelThresholds``
or the strategy grid. On a profile/slider-only re-run (same proxy data),
the change-point PELT refit (the dominant regime cost) can be skipped
entirely and the previously-computed labels re-read from disk.

Caches the four detectors' label Series (indexed by date) to a single
parquet file keyed by a content fingerprint of the proxy frame plus each
detector's constructor parameters (so a detector param change is also a
guaranteed miss, never a stale hit).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from funnel import __version__ as FUNNEL_VERSION
from funnel.compute_cache import (
    COMPUTE_CACHE_SCHEMA,
    evict_oldest,
    hash_dataframe,
    write_cache_metadata,
)
from funnel.regime.base import RegimeDetector

logger = logging.getLogger(__name__)


def fingerprint_regime_inputs(proxy_df: pd.DataFrame, detectors: dict[str, RegimeDetector]) -> str:
    """Fingerprint of everything the detectors' classify() output depends on."""
    h = hashlib.sha256()
    h.update(f"schema={COMPUTE_CACHE_SCHEMA}".encode())
    h.update(f"funnel_version={FUNNEL_VERSION}".encode())
    h.update(hash_dataframe(proxy_df).encode())
    for name in sorted(detectors):
        det = detectors[name]
        h.update(name.encode())
        h.update(repr(vars(det)).encode())
    return h.hexdigest()[:32]


@dataclass(slots=True, frozen=True)
class RegimeCacheResult:
    labels_by_detector: dict[str, pd.Series]
    cache_hit: bool
    fingerprint: str
    elapsed_s: float


def _cache_path(cache_dir: Path, fingerprint: str) -> Path:
    return cache_dir / f"regime_labels_{fingerprint}.parquet"


def classify_all_cached(
    proxy_df: pd.DataFrame,
    detectors: dict[str, RegimeDetector],
    cache_dir: Path,
) -> RegimeCacheResult:
    """Drop-in alternative to ``{name: d.classify(proxy_df) for name, d in detectors.items()}``
    with an on-disk cache under ``cache_dir``.

    On a miss, evicts the oldest entry if the cache is at capacity
    (``funnel.compute_cache.MAX_ENTRIES_PER_KIND``) after writing.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = fingerprint_regime_inputs(proxy_df, detectors)
    path = _cache_path(cache_dir, fingerprint)

    if path.exists():
        t0 = time.perf_counter()
        wide = pd.read_parquet(path)
        labels_by_detector = {col: wide[col] for col in wide.columns}
        dt = time.perf_counter() - t0
        logger.info("regime cache HIT fingerprint=%s (%.3fs)", fingerprint, dt)
        return RegimeCacheResult(labels_by_detector, True, fingerprint, dt)

    t0 = time.perf_counter()
    labels_by_detector = {name: d.classify(proxy_df) for name, d in detectors.items()}
    dt = time.perf_counter() - t0
    wide = pd.DataFrame(labels_by_detector)
    wide.to_parquet(path)
    write_cache_metadata(
        path.with_suffix(".json"),
        fingerprint,
        {"funnel_version": FUNNEL_VERSION, "n_rows": len(wide)},
    )
    evict_oldest(cache_dir, "regime_labels_*.parquet")
    logger.info("regime cache MISS fingerprint=%s (%.3fs)", fingerprint, dt)
    return RegimeCacheResult(labels_by_detector, False, fingerprint, dt)
