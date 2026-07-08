"""Minimal background-job registry for pipeline runs.

A sweep is CPU-heavy, so jobs run one at a time on a single-worker
``ThreadPoolExecutor`` — serial by design, not an oversight. Status is kept
in memory (for fast polling) AND mirrored to ``runs/<id>/status.json`` on
every update, so a process restart can still list and inspect past runs by
reading their status files off disk even though the in-memory registry
starts empty again.

Cancellation is cooperative: each run gets a ``threading.Event`` that a work
function's ``progress``/``should_stop`` callbacks observe. There is no way to
forcibly kill a running thread in Python, so a run only actually stops once
its work function notices the event — either at the next stage boundary
(``progress`` raises ``RunCancelledError``) or, for the long sweep loops,
at the next (config, asset) iteration (``should_stop()`` checked directly).
"""

import json
import logging
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from funnel.cancellation import RunCancelledError

logger = logging.getLogger(__name__)

STATUS_FILENAME = "status.json"

_TERMINAL_STATES = frozenset({"done", "error", "cancelled"})

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
"""Same strict run-id shape as ``funnel.api.app._RUN_ID_RE`` — kept as a
second, independent gate here so ``JobRegistry`` never joins an
externally-supplied ``run_id`` into a filesystem path unchecked, even if a
future caller forgets to validate it first (defense in depth: the API layer
already rejects bad ids before calling into this class)."""


@dataclass(slots=True, frozen=True)
class JobStatus:
    """Point-in-time status of one pipeline run."""

    run_id: str
    state: str
    """One of ``"queued"``, ``"running"``, ``"done"``, ``"error"``, ``"cancelled"``."""

    stage: str
    run_type: str
    """``"strategy"`` or ``"overlay"`` — which pipeline this run is. Always
    set explicitly by ``JobRegistry.submit``; only the on-disk read fallback
    (``_read_status``) defaults a missing key to ``"strategy"`` for legacy
    status files written before this field existed."""

    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


def _write_status(run_dir: Path, status: JobStatus) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / STATUS_FILENAME
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(asdict(status), indent=2))
    tmp_path.replace(path)


class JobRegistry:
    """Runs pipeline jobs serially and tracks their status in memory + on disk."""

    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._lock = threading.Lock()
        self._statuses: dict[str, JobStatus] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._started: set[str] = set()
        self._disk_scanned = False

    def submit(
        self,
        run_id: str,
        work: Callable[[Callable[[str], None], Callable[[], bool]], None],
        run_type: str,
    ) -> None:
        """Queue ``work`` under ``run_id``.

        ``work`` receives two callbacks: ``progress(stage: str)`` updates
        this job's status (both in-memory and on disk) as it runs, and
        raises ``RunCancelledError`` if cancellation has been requested;
        ``should_stop() -> bool`` reports the same cancellation request
        without raising, for callers (the sweep loops) that need to check it
        many times per stage rather than only at stage boundaries.
        """
        if not _RUN_ID_RE.match(run_id):
            raise ValueError(f"invalid run_id {run_id!r}")
        run_dir = self._runs_dir / run_id
        status = JobStatus(run_id=run_id, state="queued", stage="queued", run_type=run_type)
        cancel_event = threading.Event()
        with self._lock:
            self._statuses[run_id] = status
            self._cancel_events[run_id] = cancel_event
        _write_status(run_dir, status)

        self._executor.submit(self._run, run_id, run_dir, work, run_type, cancel_event)

    def cancel(self, run_id: str) -> bool:
        """Request cancellation of ``run_id``.

        Returns ``True`` if the request was accepted (the run was queued or
        running) and ``False`` if the run is unknown or already terminal
        (``done``/``error``/``cancelled``) — a no-op in that case. A queued
        run (never started) transitions to ``"cancelled"`` immediately, here,
        under the lock — ``_run``'s started-flag check then refuses to
        execute ``work`` at all for it. A running run keeps its ``"running"``
        state until its ``progress``/``should_stop`` callback next observes
        the event and unwinds via ``RunCancelledError``.
        """
        run_dir = self._runs_dir / run_id
        new_status: JobStatus | None = None
        with self._lock:
            status = self._statuses.get(run_id)
            if status is None or status.state in _TERMINAL_STATES:
                return False
            event = self._cancel_events[run_id]
            event.set()
            if run_id not in self._started:
                new_status = replace(
                    status,
                    state="cancelled",
                    stage="cancelled",
                    error=None,
                    finished_at=datetime.now(UTC).isoformat(),
                )
                self._statuses[run_id] = new_status
        if new_status is not None:
            _write_status(run_dir, new_status)
        return True

    def _run(
        self,
        run_id: str,
        run_dir: Path,
        work: Callable[[Callable[[str], None], Callable[[], bool]], None],
        run_type: str,
        cancel_event: threading.Event,
    ) -> None:
        with self._lock:
            # A queued-cancel (see `cancel`) may have already marked this run
            # "cancelled" before the executor got around to it; if so, never
            # run `work` at all.
            if self._statuses[run_id].state == "cancelled":
                return
            self._started.add(run_id)

        started_at = datetime.now(UTC).isoformat()
        self._update(
            run_id,
            run_dir,
            state="running",
            stage="starting",
            started_at=started_at,
            run_type=run_type,
        )

        def progress(stage: str) -> None:
            if cancel_event.is_set():
                raise RunCancelledError(f"run {run_id!r} cancelled")
            self._update(
                run_id,
                run_dir,
                state="running",
                stage=stage,
                started_at=started_at,
                run_type=run_type,
            )

        def should_stop() -> bool:
            return cancel_event.is_set()

        try:
            work(progress, should_stop)
        except RunCancelledError:
            logger.info("job %s cancelled", run_id)
            self._update(
                run_id,
                run_dir,
                state="cancelled",
                stage="cancelled",
                started_at=started_at,
                finished_at=datetime.now(UTC).isoformat(),
                error=None,
                run_type=run_type,
            )
            return
        except Exception as exc:  # noqa: BLE001 - report to job status, don't crash the worker
            logger.exception("job %s failed", run_id)
            self._update(
                run_id,
                run_dir,
                state="error",
                stage="error",
                started_at=started_at,
                finished_at=datetime.now(UTC).isoformat(),
                error=str(exc),
                run_type=run_type,
            )
            return

        self._update(
            run_id,
            run_dir,
            state="done",
            stage="done",
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
            run_type=run_type,
        )

    def _update(
        self,
        run_id: str,
        run_dir: Path,
        *,
        state: str,
        stage: str,
        run_type: str,
        started_at: str | None = None,
        finished_at: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            prev = self._statuses.get(run_id)
            resolved_started_at = started_at
            if resolved_started_at is None and prev is not None:
                resolved_started_at = prev.started_at
            status = replace(
                prev
                if prev is not None
                else JobStatus(run_id=run_id, state=state, stage=stage, run_type=run_type),
                state=state,
                stage=stage,
                run_type=run_type,
                started_at=resolved_started_at,
                finished_at=finished_at,
                error=error,
            )
            self._statuses[run_id] = status
        _write_status(run_dir, status)

    def get(self, run_id: str) -> JobStatus | None:
        """Return the in-memory status for ``run_id``, falling back to its on-disk file.

        A status found on disk this way is cached into the in-memory dict
        before being returned, so a repeated lookup of the same unknown id
        does not re-read the file every time.

        Returns ``None`` (never raises) for a malformed ``run_id`` — the
        in-memory registry simply won't have an entry for it, and the
        on-disk fallback is skipped rather than joining an unvalidated id
        into a filesystem path.
        """
        with self._lock:
            status = self._statuses.get(run_id)
        if status is not None:
            return status
        if not _RUN_ID_RE.match(run_id):
            return None
        on_disk = _read_status(self._runs_dir / run_id)
        if on_disk is None:
            return None
        with self._lock:
            status = self._statuses.setdefault(run_id, on_disk)
        return status

    def list_all(self) -> list[JobStatus]:
        """List every known run: in-memory jobs plus any on-disk statuses not yet in memory.

        The runs dir is scanned exactly once, on the first call — every
        subsequent call serves entirely from memory. All in-process status
        changes already flow through ``_update``, so re-scanning disk on
        every ~1s UI poll would be pure waste; the only statuses that can
        exist on disk and not in memory are from a previous process. A
        ``"running"`` status found during that one-time scan is therefore
        necessarily stale (its process is gone and will never finish it), so
        it is corrected to ``"error"`` here rather than left to look like a
        run in progress forever.
        """
        with self._lock:
            if not self._disk_scanned:
                self._scan_disk_locked()
                self._disk_scanned = True
            statuses = dict(self._statuses)

        return sorted(statuses.values(), key=lambda s: s.run_id, reverse=True)

    def _scan_disk_locked(self) -> None:
        """Merge on-disk statuses not yet known in memory. Caller must hold ``self._lock``."""
        if not self._runs_dir.exists():
            return
        for run_dir in self._runs_dir.iterdir():
            if not run_dir.is_dir() or run_dir.name in self._statuses:
                continue
            on_disk = _read_status(run_dir)
            if on_disk is None:
                continue
            if on_disk.state == "running":
                on_disk = replace(
                    on_disk,
                    state="error",
                    stage="error",
                    error="process restarted mid-run",
                    finished_at=datetime.now(UTC).isoformat(),
                )
                _write_status(run_dir, on_disk)
            self._statuses[run_dir.name] = on_disk


def _read_status(run_dir: Path) -> JobStatus | None:
    path = run_dir / STATUS_FILENAME
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return JobStatus(
        run_id=data["run_id"],
        state=data["state"],
        stage=data["stage"],
        run_type=data.get("run_type", "strategy"),
        error=data.get("error"),
        started_at=data.get("started_at"),
        finished_at=data.get("finished_at"),
    )
