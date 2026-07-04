# The Funnel

The Funnel is a strategy screener and backtester for a single power-user, built
around a "test everything, then validate hard" discipline: sweep dozens of
strategies across parameter grids and a broad asset universe, then kill most of
them through walk-forward validation and a six-filter survival funnel, backed by
robustness checks (parameter sensitivity, bootstrap stress tests) so that only
genuinely durable edges are reported — attrition, deep drawdowns, and fragile
survivors are surfaced by design, never hidden.

## Quickstart

**Local dev:**

```bash
cd engine
uv sync
uv run uvicorn funnel.api.app:create_app --factory --reload
```

**Docker:**

```bash
docker compose up
```

Either way, the app serves the API at `/api/*` and the frontend at `/`.

## Repo layout

```
engine/       Python 3.14 / FastAPI backend (package: funnel)
web/          Cyberdeck SPA frontend (single-file, zero-build)
docs/         Architecture notes
runs/         Per-run pipeline artifacts (gitignored)
```

See [`PLAN.md`](PLAN.md) for the full implementation plan and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for an architecture overview.
