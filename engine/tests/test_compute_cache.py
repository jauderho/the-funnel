"""Tests for funnel.compute_cache: shared PERF-2 cache infrastructure
(cache-dir resolution, eviction)."""

from pathlib import Path

import pytest

from funnel.compute_cache import default_compute_cache_dir, evict_oldest, write_cache_metadata


def test_default_compute_cache_dir_honors_funnel_compute_cache_dir_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    override = tmp_path / "explicit-cache"
    monkeypatch.setenv("FUNNEL_COMPUTE_CACHE_DIR", str(override))
    assert default_compute_cache_dir() == override


def test_default_compute_cache_dir_falls_back_to_funnel_data_dir_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FUNNEL_COMPUTE_CACHE_DIR", raising=False)
    data_dir = tmp_path / "data-dir"
    monkeypatch.setenv("FUNNEL_DATA_DIR", str(data_dir))
    assert default_compute_cache_dir() == data_dir / "compute_cache"


def test_default_compute_cache_dir_falls_back_to_repo_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FUNNEL_COMPUTE_CACHE_DIR", raising=False)
    monkeypatch.delenv("FUNNEL_DATA_DIR", raising=False)
    result = default_compute_cache_dir()
    assert result.name == "compute_cache"
    assert result.parent.name == "data"


def _touch_entry(cache_dir: Path, name: str) -> Path:
    path = cache_dir / f"{name}.parquet"
    path.write_bytes(b"x")
    write_cache_metadata(path.with_suffix(".json"), fingerprint=name, extra={})
    return path


def test_evict_oldest_keeps_only_the_n_most_recent(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    paths = [_touch_entry(cache_dir, f"entry{i}") for i in range(5)]
    # Force distinct, increasing mtimes (some filesystems have coarse mtime
    # resolution) so "oldest" is unambiguous.
    import os
    import time

    now = time.time()
    for i, path in enumerate(paths):
        ts = now + i
        os.utime(path, (ts, ts))
        os.utime(path.with_suffix(".json"), (ts, ts))

    evict_oldest(cache_dir, "entry*.parquet", keep=2)

    remaining = sorted(p.stem for p in cache_dir.glob("*.parquet"))
    assert remaining == ["entry3", "entry4"]
    # Sidecar JSON metadata is deleted alongside its parquet file.
    remaining_json = sorted(p.stem for p in cache_dir.glob("*.json"))
    assert remaining_json == ["entry3", "entry4"]


def test_evict_oldest_is_noop_when_under_the_cap(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _touch_entry(cache_dir, "only-one")

    evict_oldest(cache_dir, "only-one*.parquet", keep=20)

    assert (cache_dir / "only-one.parquet").exists()
