"""Shared on-disk compute-cache infrastructure (PERF-2).

Backs two threshold-independent caches: the sweep-metrics cache
(``funnel.backtest.sweep_cache``) and the regime-label cache
(``funnel.regime.label_cache``). Both are keyed by a fingerprint of the
actual computation inputs plus an engine-version salt (``funnel.__version__``
and ``COMPUTE_CACHE_SCHEMA`` below) so that a code change to the computation
itself -- not just its inputs -- reliably invalidates every existing entry
rather than serving stale numbers under semantics that have since changed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

COMPUTE_CACHE_SCHEMA = 1
"""Bump this whenever a change to sweep / walk-forward / strategy / regime
semantics would change the *value* of a cached metric or label for
identical inputs -- e.g. a fix to how returns are stitched or compounded is
exactly the class of change that must bump this. Without a schema bump, a
stale cache entry computed under the old semantics would be served as if it
were still correct. This constant is mixed into every fingerprint in these
caches alongside ``funnel.__version__``, so any release that bumps either
one invalidates all prior entries -- a guaranteed miss, never a stale hit."""

MAX_ENTRIES_PER_KIND = 20
"""Eviction cap: each cache "kind" (sweep metrics, regime labels) keeps at
most this many most-recently-written entries on disk; on every write past
the cap, the oldest (by file mtime) entries are deleted."""


def default_compute_cache_dir() -> Path:
    """Resolve the compute-cache directory.

    Honors ``FUNNEL_COMPUTE_CACHE_DIR`` if set; otherwise
    ``<FUNNEL_DATA_DIR or repo data>/compute_cache`` -- mirrors
    ``funnel.data.sources.default_cache_dir``'s override precedence.
    """
    override = os.environ.get("FUNNEL_COMPUTE_CACHE_DIR")
    if override:
        return Path(override)
    data_dir_override = os.environ.get("FUNNEL_DATA_DIR")
    if data_dir_override:
        return Path(data_dir_override) / "compute_cache"
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "compute_cache"


def hash_dataframe(df: pd.DataFrame) -> str:
    """Content hash of a frame's index and values (order-sensitive).

    Datetime indexes are canonicalized to int64 nanoseconds before hashing:
    the same instants stored at different precisions (yfinance returns
    ``datetime64[s]``, a parquet round-trip yields ``datetime64[ms]``) must
    hash identically, otherwise the first re-run after a data refresh pays
    a spurious cache miss. Genuine value differences still change the hash.
    """
    h = hashlib.sha256()
    if isinstance(df.index, pd.DatetimeIndex):
        h.update(np.ascontiguousarray(df.index.astype("datetime64[ns]").asi8).tobytes())
    else:
        h.update(pd.util.hash_pandas_object(df.index).to_numpy(dtype=np.uint64).tobytes())
    for col in sorted(df.columns):
        h.update(col.encode())
        h.update(np.ascontiguousarray(df[col].to_numpy(dtype=np.float64)).tobytes())
    return h.hexdigest()


def write_cache_metadata(path: Path, fingerprint: str, extra: dict[str, object]) -> None:
    """Write the JSON sidecar for one cache entry (used for provenance and eviction)."""
    payload: dict[str, object] = {"fingerprint": fingerprint, "schema": COMPUTE_CACHE_SCHEMA}
    payload.update(extra)
    path.write_text(json.dumps(payload, indent=2))


def evict_oldest(cache_dir: Path, glob_pattern: str, keep: int = MAX_ENTRIES_PER_KIND) -> None:
    """Delete the oldest (by mtime) entries of one cache kind beyond ``keep``.

    ``glob_pattern`` matches the parquet files of one cache kind (e.g.
    ``"sweep_metrics_*.parquet"``); each match's sibling ``.json`` metadata
    file, if present, is deleted alongside it.
    """
    entries = sorted(cache_dir.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in entries[keep:]:
        stale.unlink(missing_ok=True)
        stale.with_suffix(".json").unlink(missing_ok=True)
