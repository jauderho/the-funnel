"""Tests for ``funnel.api.jobs.JobRegistry``, focused on run_id path safety.

``JobRegistry`` joins caller-supplied ``run_id`` values into filesystem
paths (``runs_dir / run_id / ...``) in both ``submit`` and ``get``. These
tests exercise that validation directly, independent of the FastAPI routing
layer (which has its own, separately-tested guard in
``funnel.api.app._validate_run_id``).
"""

from pathlib import Path

import pytest

from funnel.api.jobs import JobRegistry


def test_get_returns_none_for_traversal_run_id_without_touching_disk(tmp_path: Path) -> None:
    # Plant a status file outside the configured runs dir; if `get` ever
    # joined an unvalidated run_id into a path, this is what it would leak.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "status.json").write_text('{"run_id": "leaked", "state": "done", "stage": "done"}')

    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    registry = JobRegistry(runs_dir)

    assert registry.get("../outside") is None
    assert registry.get("..%2Foutside") is None


def test_submit_rejects_traversal_run_id(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    registry = JobRegistry(runs_dir)

    with pytest.raises(ValueError, match="invalid run_id"):
        registry.submit("../evil", lambda progress: None)


def test_submit_and_get_roundtrip_for_valid_run_id(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    registry = JobRegistry(runs_dir)

    registry.submit("20260704T000000000000", lambda progress: progress("done"))

    # Poll briefly; the executor runs on a background thread.
    import time

    deadline = time.monotonic() + 5.0
    status = None
    while time.monotonic() < deadline:
        status = registry.get("20260704T000000000000")
        if status is not None and status.state in ("done", "error"):
            break
        time.sleep(0.05)

    assert status is not None
    assert status.state == "done"
