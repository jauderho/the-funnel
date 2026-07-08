# Architecture

The Funnel has two components:

- **`engine/`** — a Python 3.14 / FastAPI backend (package `funnel`). This is the
  quant core: data ingestion, the strategy library, the backtest engine, walk-forward
  validation, the six-filter survival funnel, robustness checks, cross-sectional
  momentum research, regime detection, the layer/portfolio stack, and profile/slider
  mapping. It also serves a small JSON API and hosts the frontend as static files.
- **`web/index.html`** — the catfu (cassette-futurism) SPA: a single-file, zero-build,
  vanilla HTML/CSS/JS frontend (per `AESTHETIC_CONTRACT.md`). It drives the pipeline
  via the API, polls run status, and renders profile faders, the funnel attrition
  view, sensitivity/bootstrap/regime/correlation views, and hand-built SVG charts.

## Module map (`engine/src/funnel/`)

| Module | Responsibility |
|---|---|
| `config.py` | Frozen dataclasses: `FunnelThresholds`, `WalkForwardConfig`, `CostModel`, universe config. |
| `data/` | `sources.py` — `DataSource` protocol, `YFinanceSource`, `CachedSource` (parquet on disk), `universe.py` — the ~30-asset universe and min-history filtering. |
| `strategies/` | `base.py` (`Strategy` protocol, no-look-ahead contract), `trend.py`, `meanrev.py`, `volume.py`, `volatility.py`, `pattern.py`, `composite.py`, `indicators.py`, `grid.py` (`build_all_configs()` — the full config × family grid). |
| `backtest/` | `engine.py` (position × next-day return − per-side cost), `metrics.py` (Sharpe, max DD, trade count, CAGR, DD duration, win rate), `walkforward.py` (5-window 70/30 stitched-OOS), `funnel.py` (six survival filters), `sweep.py` (full config × asset sweep → `sweep_results.csv`). |
| `robustness/` | `sensitivity.py` (per-family mean/std/positive-fraction of OOS Sharpe), `bootstrap.py` (200-reshuffle stress test, p5/p50/p95 Sharpe, worst-case DD, solid/fragile flag). |
| `momentum/` | `cross_sectional.py` — research-only long/short cross-sectional momentum check, reported side-by-side with single-asset momentum, clearly labeled non-tradeable. |
| `regime/` | `base.py` (`RegimeDetector` protocol), `hmm.py`, `ma_filter.py`, `realized_vol.py`, `changepoint.py`, `compare.py` (detector agreement + regime-conditioned performance). |
| `layers/` | `sizing.py` (vol targeting, ATR sizing, caps), `combine.py` (signal blending), `router.py` (regime→strategy routing), `stack.py` (per-layer on/off attribution). |
| `portfolio/` | `correlation.py` — cross-strategy correlation matrix + redundancy flags. |
| `profiles/` | `models.py` (`Profile`, `SliderValues`), `mapping.py` (slider→threshold + ranking-weight mapping, explicit and logged), `store.py` (JSON on disk), `screener.py` (profile-driven filtering incl. hard-constraint exclusions). |
| `options/` | Options overlay module (v2, PRD §11.3). `pricing.py` (BSM price/delta/P(ITM), causal realized-vol proxy + vol-risk-premium knob), `overlays.py` (covered call, cash-secured put, vertical spread, LEAPs structures on a daily roll grid; defined-risk validation), `grid.py` (`build_overlay_grid()` — structure × delta target × DTE × roll-rule config grid), `sweep.py` (overlay config × symbol sweep, walk-forward + bootstrap scoring vs. buy-and-hold → `overlay_results.csv`). |
| `reports/` | `attrition.py` — six-filter funnel attrition report (by category, by family). |
| `pipeline.py` | Orchestrates every stage above into one run; writes all artifacts + `report.json`. Pure glue — no scoring/funnel/robustness logic of its own. Also holds `run_overlay_pipeline`, the lighter sibling run type for options-overlay requests. |
| `api/` | `app.py` (FastAPI app factory + routes), `jobs.py` (background job registry), `testing.py` (`SyntheticSource`, network-free fake data for dev/tests). |

## Pipeline stage order

`run_pipeline` (`engine/src/funnel/pipeline.py`) runs, per run id, in this order:

1. **data** — fetch + filter the asset universe via the injected `DataSource`.
2. **thresholds** — map the profile's sliders onto `FunnelThresholds` (logged via `explain_mapping`).
3. **sweep** — run every strategy config × asset backtest through walk-forward OOS scoring.
4. **attrition** — apply the six-filter funnel to the raw sweep, build the attrition report.
5. **sensitivity** — per-family OOS Sharpe sensitivity across the full sweep (all families, not just survivors).
6. **bootstrap** — 200-reshuffle stress test on survivors' stitched OOS returns.
7. **cross-sectional** — research-only long/short momentum check, compared to single-asset momentum.
8. **regime** — HMM + 3 comparator detectors on the SPY proxy; regime-conditioned performance for survivors.
9. **layers** — layer-stack attribution (sizing/combine/routing) for the top survivor's asset.
10. **correlation** — correlation matrix + redundancy flags across the top survivors.
11. **screen** — profile-driven screening (hard constraints + soft ranking) for the UI's results view.
12. **report** — assemble `report.json`, the single JSON source of truth for the API/UI.

Every stage announces itself via a `progress` callback so the API's job registry can surface live status. A stage with nothing to report (e.g. zero survivors, missing regime proxy) records a warning and continues — a zero-survivor run is a valid, complete result, never suppressed.

### Overlay run type (`run_overlay_pipeline`, `pipeline.py`)

A separate, lighter run type for options-overlay requests (v2, PRD §11.3): fetches
only the caller-requested symbols (not the whole universe), runs 2 stages, and
writes its own `report.json` (`run_type: "overlay"`) alongside `overlay_results.csv`.

1. **data** — fetch only `config.symbols` via the injected `DataSource`, filter by
   min-history. Zero eligible symbols completes honestly with an empty sweep and a
   warning, exactly as a zero-survivor strategy run does.
2. **sweep** — every `OverlayConfig` (`build_overlay_grid()`, or an override) ×
   eligible symbol through `simulate_overlay`, walk-forward-scored against the
   underlying's buy-and-hold on identical windows, then bootstrap-stressed. Writes
   `overlay_results.csv`, then a `report.json` carrying `transparency` counts,
   the always-present `model_risk_caveat`, `grid_summary` (per-structure config
   counts), and every overlay row (survived or not).

## Artifacts (`runs/<run_id>/`, from `ARTIFACT_NAMES` in `pipeline.py`)

| File | Contents |
|---|---|
| `sweep_results.csv` | Every strategy config × asset backtest result (including skipped rows). |
| `funnel_report.csv` | Six-filter pass/fail attrition, by category and by family. |
| `sensitivity.csv` | Per-family parameter sensitivity (mean/std/positive-fraction of OOS Sharpe) — always all families. |
| `bootstrap.csv` | Bootstrap-reshuffle stress test for survivors (p5/p50/p95 Sharpe, worst-case DD, solid/fragile). |
| `cross_sectional.csv` | Cross-sectional momentum research/diagnostic results. |
| `regime_performance.csv` | Regime-conditioned performance breakdown for survivors. |
| `layer_attribution.csv` | Per-layer (sizing/combine/routing) marginal attribution for the top survivor. |
| `correlation_matrix.csv` | Cross-strategy correlation matrix + redundancy flags among top survivors. |
| `overlay_results.csv` | Options-overlay run only: one row per (overlay config, symbol) — overlay vs. buy-and-hold Sharpe/drawdown, premium yield, `mean_model_prob_itm`, assignment/roll counts, upside forgone, bootstrap verdict; skipped rows flagged, never dropped. |
| `report.json` | Everything above assembled into one JSON document, plus thresholds applied, mapping explanation, transparency counts, and warnings. For an overlay run, a separate, smaller `report.json` (`run_type: "overlay"`) with `transparency` counts, `model_risk_caveat`, `grid_summary`, and `overlay_rows`. |

All artifacts are directly downloadable from the API; the frontend polls run status, then renders from `report.json` and fetches individual CSVs on demand.

## API endpoints (`engine/src/funnel/api/app.py`)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness + version. |
| GET | `/api/profiles` | List saved profiles. |
| POST | `/api/profiles` | Save a named profile (sliders). |
| DELETE | `/api/profiles/{name}` | Delete a profile (presets are protected). |
| POST | `/api/runs` | Start a pipeline run for a profile name or ad-hoc sliders; returns `run_id`. |
| POST | `/api/overlays` | Start an options-overlay run for a list of underlying symbols (capped, validated against the asset universe); returns `run_id`. |
| GET | `/api/overlays/universe` | List symbols eligible for an overlay run (symbol + asset class). |
| GET | `/api/runs` | List all known runs and their status. |
| GET | `/api/runs/{run_id}/status` | Poll a run's status (queued/running/done/error). |
| GET | `/api/runs/{run_id}/report` | Fetch the assembled `report.json` for a finished run. |
| GET | `/api/runs/{run_id}/artifacts/{name}` | Download a whitelisted artifact CSV. |
| GET | `/api/mapping/preview` | Preview the slider→threshold/ranking-weight mapping without running the pipeline. |
| GET | `/` | Static SPA (`web/`), mounted last, only if the resolved web directory exists. |

`run_id` is validated against a strict `[A-Za-z0-9_-]+` pattern before it ever touches a filesystem path; artifact names are checked against a whitelist derived from `ARTIFACT_NAMES` (excluding `report.json`, which has its own endpoint).

## Environment variables

| Variable | Effect | Default |
|---|---|---|
| `FUNNEL_WEB_DIR` | Directory of frontend static assets to mount at `/`. | `<repo root>/web` (resolved relative to `app.py`) |
| `FUNNEL_DATA_DIR` | On-disk cache directory for fetched market data. | engine-relative data dir |
| `FUNNEL_RUNS_DIR` | Directory where run artifacts are written. | `<repo root>/runs` |
| `FUNNEL_PROFILES_DIR` | Directory where saved profiles (JSON) live. | engine-relative profiles dir |
| `FUNNEL_FAKE_DATA` | `1` selects `SyntheticSource` (deterministic, network-free) instead of the cached yfinance source — used for local UI dev and CI. | unset (real yfinance data) |

## Dev quickstart

```bash
cd engine
uv sync
FUNNEL_FAKE_DATA=1 FUNNEL_RUNS_DIR=/tmp/funnel-dev-runs \
  uv run uvicorn funnel.api.app:create_app --factory --port 8731
```

`FUNNEL_FAKE_DATA=1` avoids any real network calls (synthetic price series) — this
is what `.claude/launch.json` uses for preview/dev. Drop it (and point
`FUNNEL_RUNS_DIR` wherever you like) to run against real, cached yfinance data.

```bash
# tests
cd engine && uv run pytest -q

# lint / format / types
uv run ruff check .
uv run ruff format --check .
uv run ty check
```

## Further reading

- [`PLAN.md`](../PLAN.md) — the full milestone plan and architectural decisions.
- [`AESTHETIC_CONTRACT.md`](../AESTHETIC_CONTRACT.md) — the binding design contract
  for all UI work in this repo.
- [`docs/OPEN_ITEMS.md`](OPEN_ITEMS.md) — known gaps and deferred work.
