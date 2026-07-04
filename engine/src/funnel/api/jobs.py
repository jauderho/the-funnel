"""Minimal background-job registry for pipeline runs.

A sweep is CPU-heavy, so jobs run one at a time on a single-worker
``ThreadPoolExecutor`` — serial by design, not an oversight. Status is kept
in memory (for fast polling) AND mirrored to ``runs/<id>/status.json`` on
every update, so a process restart can still list and inspect past runs by
reading their status files off disk even though the in-memory registry
starts empty again.
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

logger = logging.getLogger(__name__)

STATUS_FILENAME = "status.json"

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
    """One of ``"queued"``, ``"running"``, ``"done"``, ``"error"``."""

    stage: str
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

    def submit(self, run_id: str, work: Callable[[Callable[[str], None]], None]) -> None:
        """Queue ``work`` under ``run_id``.

        ``work`` receives a ``progress(stage: str)`` callback that updates
        this job's status (both in-memory and on disk) as it runs.
        """
        if not _RUN_ID_RE.match(run_id):
            raise ValueError(f"invalid run_id {run_id!r}")
        run_dir = self._runs_dir / run_id
        status = JobStatus(run_id=run_id, state="queued", stage="queued")
        with self._lock:
            self._statuses[run_id] = status
        _write_status(run_dir, status)

        self._executor.submit(self._run, run_id, run_dir, work)

    def _run(
        self, run_id: str, run_dir: Path, work: Callable[[Callable[[str], None]], None]
    ) -> None:
        started_at = datetime.now(UTC).isoformat()
        self._update(run_id, run_dir, state="running", stage="starting", started_at=started_at)

        def progress(stage: str) -> None:
            self._update(run_id, run_dir, state="running", stage=stage, started_at=started_at)

        try:
            work(progress)
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
            )
            return

        self._update(
            run_id,
            run_dir,
            state="done",
            stage="done",
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
        )

    def _update(
        self,
        run_id: str,
        run_dir: Path,
        *,
        state: str,
        stage: str,
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
                prev if prev is not None else JobStatus(run_id=run_id, state=state, stage=stage),
                state=state,
                stage=stage,
                started_at=resolved_started_at,
                finished_at=finished_at,
                error=error,
            )
            self._statuses[run_id] = status
        _write_status(run_dir, status)

    def get(self, run_id: str) -> JobStatus | None:
        """Return the in-memory status for ``run_id``, falling back to its on-disk file.

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
        return _read_status(self._runs_dir / run_id)

    def list_all(self) -> list[JobStatus]:
        """List every known run: in-memory jobs plus any on-disk statuses not yet in memory."""
        with self._lock:
            statuses = dict(self._statuses)

        if self._runs_dir.exists():
            for run_dir in self._runs_dir.iterdir():
                if not run_dir.is_dir() or run_dir.name in statuses:
                    continue
                on_disk = _read_status(run_dir)
                if on_disk is not None:
                    statuses[run_dir.name] = on_disk

        return sorted(statuses.values(), key=lambda s: s.run_id, reverse=True)


def _read_status(run_dir: Path) -> JobStatus | None:
    path = run_dir / STATUS_FILENAME
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return JobStatus(
        run_id=data["run_id"],
        state=data["state"],
        stage=data["stage"],
        error=data.get("error"),
        started_at=data.get("started_at"),
        finished_at=data.get("finished_at"),
    )
