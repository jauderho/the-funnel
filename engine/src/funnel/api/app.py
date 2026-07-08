"""FastAPI application factory for The Funnel engine."""

import json
import os
import re
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from funnel import __version__
from funnel.api.jobs import JobRegistry, JobStatus
from funnel.api.testing import SyntheticSource
from funnel.config import CostModel, FunnelThresholds, WalkForwardConfig
from funnel.data.sources import CachedSource, DataSource, YFinanceSource
from funnel.data.universe import ASSET_UNIVERSE
from funnel.options.grid import OverlayConfig
from funnel.options.pricing import VolProxyConfig
from funnel.pipeline import (
    ARTIFACT_NAMES,
    OverlayRunConfig,
    PipelineConfig,
    run_overlay_pipeline,
    run_pipeline,
)
from funnel.profiles.mapping import explain_mapping, ranking_weights, thresholds_for
from funnel.profiles.models import Profile, SliderValues
from funnel.profiles.store import delete_profile, list_profiles, load_profile, save_profile
from funnel.strategies.grid import StrategyConfig

_OVERLAY_SYMBOLS_CAP = 10
"""Max number of symbols accepted per POST /api/overlays request — an
overlay sweep is O(configs x symbols), so this keeps a single request's
runtime bounded."""

REPORT_ARTIFACT_NAME = "report.json"
_ARTIFACT_WHITELIST = frozenset(ARTIFACT_NAMES) - {REPORT_ARTIFACT_NAME}

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
"""Strict run-id format: no path separators, no ``.``, no leading ``..`` —
anything that isn't this shape is rejected before it ever touches a
filesystem path. ``create_run``'s generator (``%Y%m%dT%H%M%S%f``) always
matches this pattern."""


def _validate_run_id(run_id: str) -> None:
    """Raise 400 unless ``run_id`` is a safe, path-traversal-free token.

    Every run-scoped endpoint must call this before joining ``run_id`` into
    any filesystem path (directly, or indirectly via ``JobRegistry``).
    """
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail=f"invalid run_id {run_id!r}")


def _resolve_web_dir() -> Path:
    """Resolve the directory containing the frontend static assets.

    Honors the ``FUNNEL_WEB_DIR`` environment variable if set; otherwise
    falls back to the ``web/`` directory at the repo root (relative to this
    file's location in ``engine/src/funnel/api/app.py``).
    """
    override = os.environ.get("FUNNEL_WEB_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4] / "web"


def runs_dir() -> Path:
    """Resolve the on-disk runs directory.

    Honors the ``FUNNEL_RUNS_DIR`` environment variable if set; otherwise
    falls back to ``<repo root>/runs``.
    """
    override = os.environ.get("FUNNEL_RUNS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4] / "runs"


def get_data_source() -> DataSource:
    """Resolve the default data source for new runs.

    Honors ``FUNNEL_FAKE_DATA=1`` for a deterministic, network-free
    synthetic source (useful for local UI dev); otherwise a cached
    yfinance-backed source. Overridable in tests via monkeypatching this
    module-level function or ``app.dependency_overrides``-style injection at
    the call site (``create_app`` reads it once per request via the closure
    below, so tests typically monkeypatch ``funnel.api.app.get_data_source``
    directly before calling ``create_app()``).
    """
    if os.environ.get("FUNNEL_FAKE_DATA") == "1":
        return SyntheticSource()
    return CachedSource(YFinanceSource())


def get_strategy_configs() -> list[StrategyConfig] | None:
    """Resolve the strategy grid override for new runs.

    ``None`` (the production default) runs the full ``build_all_configs()``
    grid inside ``run_pipeline``. Tests monkeypatch
    ``funnel.api.app.get_strategy_configs`` to return a small, explicit list
    so a POST /api/runs test completes in seconds instead of running the
    full production sweep.
    """
    return None


def get_overlay_configs() -> list[OverlayConfig] | None:
    """Resolve the overlay grid override for new overlay runs.

    ``None`` (the production default) runs the full ``build_overlay_grid()``
    grid inside ``run_overlay_pipeline``. Tests monkeypatch
    ``funnel.api.app.get_overlay_configs`` to return a small, explicit list
    so a POST /api/overlays test completes in seconds instead of running the
    full production overlay grid.
    """
    return None


class SliderValuesModel(BaseModel):
    capital: int
    risk_tolerance: int
    time_horizon: int
    drawdown_tolerance: int


class SaveProfileRequest(BaseModel):
    name: str
    capital: int
    risk_tolerance: int
    time_horizon: int
    drawdown_tolerance: int


class CreateRunRequest(BaseModel):
    profile_name: str | None = None
    sliders: SliderValuesModel | None = None


class CreateOverlayRunRequest(BaseModel):
    symbols: list[str]


def _profile_to_dict(profile: Profile) -> dict[str, Any]:
    return {
        "name": profile.name,
        "sliders": asdict(profile.sliders),
        "created_at": profile.created_at,
        "preset": profile.preset,
    }


def _status_to_dict(status: JobStatus) -> dict[str, Any]:
    return asdict(status)


def _resolve_profile(request: CreateRunRequest) -> Profile:
    if request.profile_name is not None:
        try:
            return load_profile(request.profile_name)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    if request.sliders is not None:
        sliders = SliderValues(
            capital=request.sliders.capital,
            risk_tolerance=request.sliders.risk_tolerance,
            time_horizon=request.sliders.time_horizon,
            drawdown_tolerance=request.sliders.drawdown_tolerance,
        )
        return Profile(
            name="ad-hoc",
            sliders=sliders,
            created_at=datetime.now(UTC).date().isoformat(),
            preset=False,
        )
    raise HTTPException(status_code=400, detail="either profile_name or sliders is required")


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(title="The Funnel", version=__version__)
    registry = JobRegistry(runs_dir())

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/api/profiles")
    def get_profiles() -> list[dict[str, Any]]:
        return [_profile_to_dict(p) for p in list_profiles()]

    @app.post("/api/profiles")
    def post_profile(request: SaveProfileRequest) -> dict[str, Any]:
        try:
            sliders = SliderValues(
                capital=request.capital,
                risk_tolerance=request.risk_tolerance,
                time_horizon=request.time_horizon,
                drawdown_tolerance=request.drawdown_tolerance,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        profile = Profile(
            name=request.name,
            sliders=sliders,
            created_at=datetime.now(UTC).date().isoformat(),
            preset=False,
        )
        save_profile(profile)
        return _profile_to_dict(profile)

    @app.delete("/api/profiles/{name}")
    def delete_profile_endpoint(name: str) -> dict[str, str]:
        try:
            delete_profile(name)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"status": "deleted"}

    @app.post("/api/runs")
    def create_run(request: CreateRunRequest) -> dict[str, str]:
        profile = _resolve_profile(request)
        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")

        source = get_data_source()
        pipeline_config = PipelineConfig(
            profile=profile,
            wf=WalkForwardConfig(),
            base_thresholds=FunnelThresholds(),
            costs=CostModel(),
            configs=get_strategy_configs(),
        )

        def work(progress: Any) -> None:
            run_pipeline(pipeline_config, source, runs_dir(), run_id, progress=progress)

        registry.submit(run_id, work)
        return {"run_id": run_id}

    @app.post("/api/overlays")
    def create_overlay_run(request: CreateOverlayRunRequest) -> dict[str, str]:
        if not request.symbols:
            raise HTTPException(status_code=400, detail="symbols must not be empty")
        if len(request.symbols) > _OVERLAY_SYMBOLS_CAP:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"at most {_OVERLAY_SYMBOLS_CAP} symbols allowed, got {len(request.symbols)}"
                ),
            )
        valid_symbols = {spec.symbol for spec in ASSET_UNIVERSE}
        unknown = [s for s in request.symbols if s not in valid_symbols]
        if unknown:
            raise HTTPException(status_code=400, detail=f"unknown symbol(s): {unknown}")

        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%f")

        source = get_data_source()
        overlay_config = OverlayRunConfig(
            symbols=request.symbols,
            wf=WalkForwardConfig(),
            vol_config=VolProxyConfig(),
            thresholds=FunnelThresholds(),
            configs=get_overlay_configs(),
        )

        def work(progress: Any) -> None:
            run_overlay_pipeline(overlay_config, source, runs_dir(), run_id, progress=progress)

        registry.submit(run_id, work)
        return {"run_id": run_id}

    @app.get("/api/overlays/universe")
    def get_overlay_universe() -> list[dict[str, str]]:
        return [
            {"symbol": spec.symbol, "asset_class": spec.asset_class.value}
            for spec in ASSET_UNIVERSE
        ]

    @app.get("/api/runs")
    def list_runs() -> list[dict[str, Any]]:
        return [_status_to_dict(s) for s in registry.list_all()]

    @app.get("/api/runs/{run_id}/status")
    def get_run_status(run_id: str) -> dict[str, Any]:
        _validate_run_id(run_id)
        status = registry.get(run_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id {run_id!r}")
        return _status_to_dict(status)

    @app.get("/api/runs/{run_id}/report")
    def get_run_report(run_id: str) -> dict[str, Any]:
        _validate_run_id(run_id)
        status = registry.get(run_id)
        if status is None:
            raise HTTPException(status_code=404, detail=f"unknown run_id {run_id!r}")
        report_path = runs_dir() / run_id / REPORT_ARTIFACT_NAME
        if status.state != "done" or not report_path.is_file():
            raise HTTPException(status_code=404, detail=f"report not ready for run {run_id!r}")
        return json.loads(report_path.read_text())

    @app.get("/api/runs/{run_id}/artifacts/{name}")
    def get_run_artifact(run_id: str, name: str) -> FileResponse:
        # run_id is validated first: it is a raw path segment used to build
        # a filesystem path below, so it must be checked regardless of the
        # artifact-name whitelist.
        _validate_run_id(run_id)
        # Whitelist check: rejects any path-traversal attempt via `name`
        # (e.g. "../evil") outright, since only exact, known artifact
        # filenames are ever accepted — no path component of `name` is
        # trusted either.
        if name not in _ARTIFACT_WHITELIST:
            raise HTTPException(status_code=400, detail=f"unknown artifact {name!r}")
        path = runs_dir() / run_id / name
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"artifact {name!r} not found")
        return FileResponse(path)

    @app.get("/api/mapping/preview")
    def mapping_preview(
        capital: int, risk_tolerance: int, time_horizon: int, drawdown_tolerance: int
    ) -> dict[str, Any]:
        try:
            sliders = SliderValues(
                capital=capital,
                risk_tolerance=risk_tolerance,
                time_horizon=time_horizon,
                drawdown_tolerance=drawdown_tolerance,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        base = FunnelThresholds()
        thresholds = thresholds_for(sliders, base)
        weights = ranking_weights(sliders)
        explanation = explain_mapping(sliders, base)

        return {
            "thresholds": asdict(thresholds),
            "ranking_weights": asdict(weights),
            "explain_mapping": explanation,
        }

    web_dir = _resolve_web_dir()
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app
