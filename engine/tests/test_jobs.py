"""Tests for ``funnel.api.jobs.JobRegistry``: run_id path safety, run_type
plumbing, cooperative cancellation, and the disk-scan-once listing cache.

``JobRegistry`` joins caller-supplied ``run_id`` values into filesystem
paths (``runs_dir / run_id / ...``) in both ``submit`` and ``get``. These
tests exercise that validation directly, independent of the FastAPI routing
layer (which has its own, separately-tested guard in
``funnel.api.app._validate_run_id``).

Also covers ``should_stop`` plumbing into the sweep runners
(``funnel.backtest.sweep.run_sweep``, ``funnel.options.sweep.run_overlay_sweep``)
directly, since ``RunCancelledError`` (``funnel.cancellation``) is the shared
exception type both the registry and the sweeps use.
"""

import json
import threading
import time
from pathlib import Path

import pandas as pd
import pytest

import funnel.api.jobs as jobs_module
from funnel.api.jobs import JobRegistry, RunCancelledError
from funnel.backtest.sweep import run_sweep
from funnel.cancellation import RunCancelledError as CancellationRunCancelledError
from funnel.config import CostModel, FunnelThresholds, WalkForwardConfig
from funnel.data.universe import AssetClass
from funnel.options.grid import OverlayConfig
from funnel.options.overlays import OverlayCosts, OverlaySpec, OverlayStructure, StrikeSelector
from funnel.options.pricing import VolProxyConfig
from funnel.options.sweep import run_overlay_sweep
from funnel.strategies.base import Category
from funnel.strategies.grid import StrategyConfig
from funnel.strategies.trend import ma_crossover


def _wait_until(predicate, timeout_s: float = 5.0, interval_s: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


# ---------------------------------------------------------------------------
# run_id path-traversal hardening (pre-existing)
# ---------------------------------------------------------------------------


def test_get_returns_none_for_traversal_run_id_without_touching_disk(tmp_path: Path) -> None:
    # Plant a status file outside the configured runs dir; if `get` ever
    # joined an unvalidated run_id into a path, this is what it would leak.
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "status.json").write_text(
        '{"run_id": "leaked", "state": "done", "stage": "done", "run_type": "strategy"}'
    )

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
        registry.submit("../evil", lambda progress, should_stop: None, run_type="strategy")


def test_submit_and_get_roundtrip_for_valid_run_id(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    registry = JobRegistry(runs_dir)

    registry.submit(
        "20260704T000000000000",
        lambda progress, should_stop: progress("done"),
        run_type="strategy",
    )

    assert _wait_until(
        lambda: (
            (s := registry.get("20260704T000000000000")) is not None
            and s.state in ("done", "error")
        )
    )
    status = registry.get("20260704T000000000000")
    assert status is not None
    assert status.state == "done"


# ---------------------------------------------------------------------------
# run_type plumbing
# ---------------------------------------------------------------------------


def test_submit_threads_run_type_into_status(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    registry = JobRegistry(runs_dir)

    registry.submit(
        "run-overlay-1", lambda progress, should_stop: progress("done"), run_type="overlay"
    )
    assert _wait_until(
        lambda: (s := registry.get("run-overlay-1")) is not None and s.state == "done"
    )
    status = registry.get("run-overlay-1")
    assert status is not None
    assert status.run_type == "overlay"

    on_disk = json.loads((runs_dir / "run-overlay-1" / "status.json").read_text())
    assert on_disk["run_type"] == "overlay"


def test_legacy_status_json_without_run_type_defaults_to_strategy(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "legacy-run"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps({"run_id": "legacy-run", "state": "done", "stage": "done"})
    )

    registry = JobRegistry(runs_dir)
    status = registry.get("legacy-run")
    assert status is not None
    assert status.run_type == "strategy"


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_cancel_queued_run_prevents_execution(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    registry = JobRegistry(runs_dir)

    first_started = threading.Event()
    release_first = threading.Event()

    def first_work(progress, should_stop) -> None:
        first_started.set()
        release_first.wait(timeout=5.0)
        progress("done")

    registry.submit("run-first", first_work, run_type="strategy")
    assert first_started.wait(timeout=5.0)

    executed_second: list[bool] = []

    def second_work(progress, should_stop) -> None:
        executed_second.append(True)
        progress("done")

    registry.submit("run-second", second_work, run_type="strategy")

    # The single worker thread is still busy with `first_work`, so
    # "run-second" is guaranteed to still be queued (not started) here.
    assert registry.cancel("run-second") is True

    status = registry.get("run-second")
    assert status is not None
    assert status.state == "cancelled"
    assert status.stage == "cancelled"
    assert status.error is None

    release_first.set()
    assert _wait_until(
        lambda: (s := registry.get("run-first")) is not None and s.state in ("done", "error")
    )

    # Give the executor a moment in case it were (incorrectly) about to run
    # "run-second"; it must never execute the work function at all.
    time.sleep(0.2)
    assert executed_second == []

    on_disk = json.loads((runs_dir / "run-second" / "status.json").read_text())
    assert on_disk["state"] == "cancelled"


def test_cancel_running_job_transitions_to_cancelled_via_progress(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    registry = JobRegistry(runs_dir)

    started = threading.Event()

    def work(progress, should_stop) -> None:
        for i in range(200):
            started.set()
            progress(f"step-{i}")
            time.sleep(0.02)

    registry.submit("run-slow", work, run_type="strategy")
    assert started.wait(timeout=5.0)

    assert registry.cancel("run-slow") is True

    assert _wait_until(
        lambda: (
            (s := registry.get("run-slow")) is not None
            and s.state in ("cancelled", "error", "done")
        )
    )
    status = registry.get("run-slow")
    assert status is not None
    assert status.state == "cancelled"
    assert status.stage == "cancelled"
    assert status.error is None

    # The disk mirror is written just after the in-memory update, on the
    # worker thread; wait for it rather than racing it.
    status_path = runs_dir / "run-slow" / "status.json"
    assert _wait_until(lambda: json.loads(status_path.read_text())["state"] == "cancelled")


def test_cancel_running_job_via_should_stop(tmp_path: Path) -> None:
    """The ``should_stop`` callback (not just ``progress``) also observes cancellation."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    registry = JobRegistry(runs_dir)

    started = threading.Event()

    def work(progress, should_stop) -> None:
        started.set()
        for _ in range(200):
            if should_stop():
                raise RunCancelledError("cancelled by should_stop")
            time.sleep(0.02)

    registry.submit("run-should-stop", work, run_type="strategy")
    assert started.wait(timeout=5.0)

    assert registry.cancel("run-should-stop") is True
    assert _wait_until(
        lambda: (s := registry.get("run-should-stop")) is not None and s.state == "cancelled"
    )


def test_cancel_after_done_is_noop(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    registry = JobRegistry(runs_dir)

    registry.submit("run-done", lambda progress, should_stop: progress("done"), run_type="strategy")
    assert _wait_until(lambda: (s := registry.get("run-done")) is not None and s.state == "done")

    assert registry.cancel("run-done") is False
    status = registry.get("run-done")
    assert status is not None
    assert status.state == "done"


def test_cancel_unknown_run_is_noop(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    registry = JobRegistry(runs_dir)

    assert registry.cancel("never-existed") is False


def test_run_cancelled_error_is_same_type_as_cancellation_module() -> None:
    """``funnel.api.jobs`` re-exports the single shared exception type."""
    assert RunCancelledError is CancellationRunCancelledError


# ---------------------------------------------------------------------------
# should_stop plumbed into the sweep runners
# ---------------------------------------------------------------------------


def test_run_sweep_raises_run_cancelled_error_promptly() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0], "open": [1.0, 2.0, 3.0]})
    data = {"AAA": df}
    configs = [
        StrategyConfig(
            name="ma_crossover_10_50",
            family="ma_crossover",
            fn=ma_crossover,
            params={"fast": 10, "slow": 50},
            category=Category.TREND,
        )
    ]
    asset_classes = {"AAA": AssetClass.LARGE_CAP}

    with pytest.raises(CancellationRunCancelledError):
        run_sweep(
            data,
            configs,
            asset_classes,
            WalkForwardConfig(),
            FunnelThresholds(),
            CostModel(),
            should_stop=lambda: True,
        )


def test_run_overlay_sweep_raises_run_cancelled_error_promptly() -> None:
    df = pd.DataFrame({"close": [1.0, 2.0, 3.0]})
    data = {"AAA": df}
    configs = [
        OverlayConfig(
            name="covered_call_test",
            spec=OverlaySpec(
                structure=OverlayStructure.COVERED_CALL,
                dte_target=30,
                strike_selector=StrikeSelector(mode="delta", value=0.25),
            ),
            description="test covered call",
        )
    ]

    with pytest.raises(CancellationRunCancelledError):
        run_overlay_sweep(
            data,
            configs,
            None,
            WalkForwardConfig(),
            VolProxyConfig(),
            OverlayCosts(),
            0.03,
            FunnelThresholds(),
            should_stop=lambda: True,
        )


# ---------------------------------------------------------------------------
# Stale "running" status recovery (dead-process detection)
# ---------------------------------------------------------------------------


def test_stale_running_status_reported_as_error_on_first_list(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "stale-run"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {
                "run_id": "stale-run",
                "state": "running",
                "stage": "stage: sweep",
                "run_type": "strategy",
                "started_at": "2026-01-01T00:00:00+00:00",
            }
        )
    )

    registry = JobRegistry(runs_dir)
    statuses = registry.list_all()

    assert len(statuses) == 1
    assert statuses[0].run_id == "stale-run"
    assert statuses[0].state == "error"
    assert statuses[0].error == "process restarted mid-run"

    # The correction is persisted too, not just reported in memory.
    on_disk = json.loads((run_dir / "status.json").read_text())
    assert on_disk["state"] == "error"
    assert on_disk["error"] == "process restarted mid-run"


# ---------------------------------------------------------------------------
# Cheap listing: disk scanned only once
# ---------------------------------------------------------------------------


def test_list_all_scans_disk_only_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_dir = tmp_path / "runs"
    run_dir = runs_dir / "on-disk-run"
    run_dir.mkdir(parents=True)
    (run_dir / "status.json").write_text(
        json.dumps(
            {"run_id": "on-disk-run", "state": "done", "stage": "done", "run_type": "strategy"}
        )
    )

    registry = JobRegistry(runs_dir)

    read_calls: list[Path] = []
    real_read_status = jobs_module._read_status

    def spy_read_status(run_dir: Path):
        read_calls.append(run_dir)
        return real_read_status(run_dir)

    monkeypatch.setattr(jobs_module, "_read_status", spy_read_status)

    first = registry.list_all()
    assert len(first) == 1
    assert len(read_calls) == 1

    second = registry.list_all()
    third = registry.list_all()
    assert len(second) == 1
    assert len(third) == 1
    # No further disk reads for calls after the first.
    assert len(read_calls) == 1
