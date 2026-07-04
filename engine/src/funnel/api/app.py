"""FastAPI application factory for The Funnel engine."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from funnel import __version__


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


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(title="The Funnel", version=__version__)

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    web_dir = _resolve_web_dir()
    if web_dir.is_dir():
        app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")

    return app
