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
import os
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
    pid: int | None = None
    """OS pid of the process that submitted this run, recorded by ``submit``
    and carried forward by every later ``_update``. Used only as a liveness
    signal when a *different* process's ``JobRegistry`` finds this run's
    ``status.json`` on disk with ``state == "running"`` and no matching
    in-memory entry (see ``_pid_alive`` / ``_scan_disk_locked``). ``None``
    for legacy status files written before this field existed."""


def _write_status(run_dir: Path, status: JobStatus) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / STATUS_FILENAME
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(asdict(status), indent=2))
    tmp_path.replace(path)


def _pid_alive(pid: int) -> bool:
    """Best-effort, same-host liveness check for a status.json's ``pid``.

    ``os.kill(pid, 0)`` sends no signal — it only asks the OS whether ``pid``
    currently names a process: ``ProcessLookupError`` means it does not
    (dead); a successful call or a ``PermissionError`` (a process with that
    pid exists but is owned by another user) both mean a process with that
    pid is alive right now.

    Caveat, deliberately: this is same-host only, and pids are recycled by
    the OS after a process exits. With a runs dir shared across hosts or
    containers (e.g. a mounted volume), a foreign pid cannot be checked
    against this host's process table at all, and even on one host a dead
    process's old pid may already belong to something new. Both failure
    modes point the same direction — this function may report a truly-dead
    run as "alive". That is intentional: the caller treats "alive" as "leave
    the run's status untouched", so the failure mode of this best-effort
    check is a stale-but-harmless "running" status that a human eventually
    notices, never a live run getting falsely overwritten with "error".
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    else:
        return True


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
        self._foreign_running: set[str] = set()
        """run_ids of on-disk ``"running"`` statuses that belong to another
        process and look alive (``_pid_alive`` true, or unknown). These are
        deliberately never cached in ``self._statuses`` — see
        ``_classify_on_disk_running`` — so ``get``/``list_all`` re-read their
        status.json on every call for as long as they remain in this set."""

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
        status = JobStatus(
            run_id=run_id, state="queued", stage="queued", run_type=run_type, pid=os.getpid()
        )
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
        does not re-read the file every time — *unless* the on-disk status
        is ``"running"`` and belongs to another process that looks alive
        (see ``_classify_on_disk_running``): such a status is never
        permanently cached, so every ``get()`` call re-reads it from disk
        until it stops being ``"running"``. That is the only case that pays
        a repeated disk read.

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
        run_dir = self._runs_dir / run_id
        on_disk = _read_status(run_dir)
        if on_disk is None:
            return None
        with self._lock:
            if on_disk.state == "running":
                return self._classify_on_disk_running(run_dir, on_disk)
            self._foreign_running.discard(run_id)
            status = self._statuses.setdefault(run_id, on_disk)
        return status

    def list_all(self) -> list[JobStatus]:
        """List every known run: in-memory jobs plus any on-disk statuses not yet in memory.

        The runs dir is scanned exactly once, on the first call — every
        subsequent call serves entirely from memory, EXCEPT for runs
        classified as "foreign running" during that scan (see
        ``_classify_on_disk_running``): a foreign, live-looking "running"
        status is re-read from disk on every ``list_all`` call for as long
        as it remains "running", since it belongs to another process and
        may still be updating. This bounded extra cost (one disk read per
        foreign-running run per call) is the price of never permanently
        caching — and thereby freezing — a possibly-live run's status.

        All other on-disk "running" statuses found during the scan (no
        recorded pid, or a pid that is provably dead — see ``_pid_alive``)
        are necessarily stale and are corrected to ``"error"`` immediately,
        as before.
        """
        with self._lock:
            if not self._disk_scanned:
                self._scan_disk_locked()
                self._disk_scanned = True
            statuses = dict(self._statuses)
            foreign_ids = list(self._foreign_running)

        for run_id in foreign_ids:
            run_dir = self._runs_dir / run_id
            on_disk = _read_status(run_dir)
            if on_disk is None:
                continue
            with self._lock:
                if on_disk.state == "running":
                    statuses[run_id] = self._classify_on_disk_running(run_dir, on_disk)
                else:
                    self._foreign_running.discard(run_id)
                    self._statuses[run_id] = on_disk
                    statuses[run_id] = on_disk

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
                self._classify_on_disk_running(run_dir, on_disk)
                continue
            self._statuses[run_dir.name] = on_disk

    def _classify_on_disk_running(self, run_dir: Path, on_disk: JobStatus) -> JobStatus:
        """Classify one on-disk ``"running"`` status not present in memory.

        Caller must hold ``self._lock``. Two registry instances (e.g.
        ``uvicorn --workers>1``, or two containers) may share one runs dir;
        a "running" status found on disk with no matching in-memory entry
        could be this process's own dead predecessor, OR a run that is
        genuinely in progress right now in a sibling process. Distinguishing
        those without a shared liveness channel is exactly what ``pid`` +
        ``_pid_alive`` is for:

        - No recorded pid (legacy status file), or a pid that is provably
          dead on this host: the run is stale. Corrected to ``"error"``,
          written back to disk, and cached permanently in memory — the
          original recovery behavior.
        - A pid that looks alive (or can't be checked, e.g. a different
          host): the run might genuinely still be running. Left completely
          untouched on disk and NOT cached in ``self._statuses`` (that would
          freeze its status at whatever this one glance saw); instead its
          run_id is recorded in ``self._foreign_running`` so ``get``/
          ``list_all`` re-read it from disk on every future call while it
          remains "running". This is the safe direction: a live run is
          never falsely overwritten with "error", at the cost that a truly
          dead run with a since-recycled pid can appear "running" until a
          human notices — the lesser evil.
        """
        if on_disk.pid is None or not _pid_alive(on_disk.pid):
            corrected = replace(
                on_disk,
                state="error",
                stage="error",
                error="process restarted mid-run",
                finished_at=datetime.now(UTC).isoformat(),
            )
            _write_status(run_dir, corrected)
            self._statuses[run_dir.name] = corrected
            self._foreign_running.discard(run_dir.name)
            return corrected
        self._foreign_running.add(run_dir.name)
        return on_disk


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
        pid=data.get("pid"),
    )
