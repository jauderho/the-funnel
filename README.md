# The Funnel

The Funnel is a strategy screener and backtester for a single power-user, built
around a "test everything, then validate hard" discipline: sweep dozens of
strategies across parameter grids and a broad asset universe, then kill most of
them through walk-forward validation and a six-filter survival funnel, backed by
robustness checks (parameter sensitivity, bootstrap stress tests) so that only
genuinely durable edges are reported — attrition, deep drawdowns, and fragile
survivors are surfaced by design, never hidden.

v2 adds an options overlay module (covered calls, cash-secured puts, vertical
spreads, LEAPs) on top of any core holding: walk-forward-validated yield vs.
assignment probability vs. upside forgone, always shown against plain
buy-and-hold. Every option price is a synthetic Black-Scholes model price (no
real historical option chains are available from free sources), so treat the
yields and assignment rates as model estimates, not market-observed quotes.

## Honesty by design

Funnel thresholds live in one frozen config consumed by both the engine and the
report, so the report always states the thresholds actually used. Attrition counts
come from the raw sweep (all configs × assets), not a curated subset — a weak
strategy family's negative mean OOS Sharpe shows up in `sensitivity.csv` right next
to the survivors, never filtered out. Solid/fragile bootstrap flags and
regime-dependence are always rendered, never filterable-away in the UI. A
zero-survivor run is a valid, complete result, not an error.

## Quickstart

**Local dev:**

```bash
cd engine
uv sync
uv run uvicorn funnel.api.app:create_app --factory --reload
```

**Local dev, no network (synthetic data):**

```bash
cd engine
FUNNEL_FAKE_DATA=1 FUNNEL_RUNS_DIR=/tmp/funnel-dev-runs \
  uv run uvicorn funnel.api.app:create_app --factory --port 8731
```

`FUNNEL_FAKE_DATA=1` swaps in a deterministic synthetic data source — no yfinance
calls, no network required. This is what `.claude/launch.json` uses for preview/dev.

**Docker:**

```bash
docker compose up
```

Either way, the app serves the API at `/api/*` and the frontend at `/`.

## Repo layout

```
engine/       Python 3.14 / FastAPI backend (package: funnel)
web/          catfu SPA frontend (single-file, zero-build)
docs/         Architecture notes
runs/         Per-run pipeline artifacts (gitignored)
```

See [`PLAN.md`](PLAN.md) for the full implementation plan,
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for an architecture overview, and
[`docs/OPEN_ITEMS.md`](docs/OPEN_ITEMS.md) for known gaps.
