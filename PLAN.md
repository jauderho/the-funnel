# The Funnel — Strategy Screener & Backtester: v1 Implementation Plan

## Context

Greenfield build in `the-funnel` (repo currently contains only CI/template boilerplate — no application code). The PRD defines a regime-aware strategy screener and backtester for a single power-user, built around a "test everything, then validate hard" discipline: sweep ~49 strategies × parameter grids × ~30 assets (thousands of backtests), then kill most of them via walk-forward validation, a six-filter survival funnel, and robustness checks. Honesty-by-design is a hard constraint: the system must report attrition, deep drawdowns, and fragile survivors — it must not be tunable to look good.

I am the **orchestrator** (per AGENTS.md Model Contract). All implementation is delegated per-task; **Fable Low is the implementor ceiling**, with Sonnet/Haiku for mechanical work. I review all delegated output before accepting.

## Decisions (user-confirmed)

1. **Architecture:** Python (FastAPI) quant core + **vanilla single-file SPA** frontend (`web/index.html`), fully honoring AESTHETIC_CONTRACT.md (zero-build, no framework, hand-built CSS/SVG viz). Contract wins over PRD §12's React preference. Dockerized, local-first. **Design language (updated 2026-07-03): "catfu" (cassette-futurism)** — the user replaced the original Cyberdeck contract; the PRD's gradient-track sliders are superseded by catfu fader-style controls (no gradients, no border-radius, mechanical motion only).
2. **Options overlay (PRD §11.3): deferred to v2.** v1 keeps a pluggable seam (strategy/instrument abstractions) but builds nothing options-specific.
3. **Sliders: hybrid.** Drawdown-tolerance and risk-tolerance map directly onto funnel filter thresholds (max-DD floor, Sharpe ceiling, min trades); capital and time-horizon soft-filter/re-rank eligible families and universe. Both mappings are explicit, logged, and shown in the UI.
4. **Data: free EOD via yfinance** behind a pluggable `DataSource` protocol, with on-disk caching. Intraday slider positions labeled "unsupported in v1".

## Stated assumptions (remaining PRD §13 questions — resolved, not asked)

- **Regime detection** runs **globally on a market proxy (SPY)** for routing; detectors are pluggable so per-asset comes later. HMM is the baseline; MA-filter, realized-vol, and change-point (ruptures) run as comparators. All treated as research to validate, not presumed-correct.
- **Profiles** are fully user-defined named profiles from day one, shipping with two presets: "Retirement Core" and "Swing Sandbox".
- **Cross-sectional momentum (Layer 4)** is an internal **research/diagnostic tool**, clearly labeled non-tradeable (long/short conflicts with the no-short constraint), reported side-by-side with single-asset momentum.
- **Walk-forward config** (5 windows, 70/30 IS/OOS) ships as engine defaults in a config dataclass — adjustable via config/API, not surfaced as primary UI in v1.
- **Long-only enforcement:** tradeable-track signals are clipped to {0, 1}; short signals map to flat. Only the research-only cross-sectional module holds shorts.
- **Transaction costs:** 1bp/side default, higher (5bp) for crypto, per-asset-class configurable.

## Architecture & repo layout

```
the-funnel/
├── PLAN.md                      # this plan, committed at repo root (user requirement)
├── CHANGELOG.md                 # summarized changes per milestone (user requirement)
├── engine/                      # Python 3.14+, uv-managed project
│   ├── pyproject.toml           # deps: pandas, numpy, yfinance, hmmlearn, ruptures,
│   │                            #       fastapi, uvicorn, msgspec; dev: pytest, ruff, ty
│   ├── src/funnel/
│   │   ├── config.py            # frozen dataclasses(slots=True): FunnelThresholds,
│   │   │                        #   WalkForwardConfig, CostModel, UniverseConfig
│   │   ├── data/                # sources.py (DataSource protocol, YFinanceSource,
│   │   │                        #   ParquetCache), universe.py (~30 assets, min-history filter)
│   │   ├── strategies/          # base.py (Strategy protocol → position Series, no look-ahead),
│   │   │                        #   trend.py, meanrev.py, volume.py, volatility.py,
│   │   │                        #   pattern.py, composite.py, grid.py (config builder →
│   │   │                        #   (name, fn, params, category) tuples, hundreds of configs)
│   │   ├── backtest/            # engine.py (pos × next-day return − per-side cost),
│   │   │                        #   metrics.py (Sharpe √252, max DD, trade count, CAGR,
│   │   │                        #   DD duration, win rate), walkforward.py (5-window
│   │   │                        #   70/30 stitched-OOS), funnel.py (six filters),
│   │   │                        #   sweep.py (full config×asset sweep → sweep_results.csv)
│   │   ├── robustness/          # sensitivity.py (per-family mean/std/positive-fraction of
│   │   │                        #   OOS Sharpe), bootstrap.py (200 reshuffles, p5/p50/p95
│   │   │                        #   Sharpe, worst-case DD, solid/fragile flag)
│   │   ├── momentum/            # cross_sectional.py (21-day rebalance; 3m/6m/12-1 lookbacks;
│   │   │                        #   top/bottom-third L/S; costs on turnover; same WF scoring)
│   │   ├── regime/              # base.py (RegimeDetector protocol), hmm.py, realized_vol.py,
│   │   │                        #   ma_filter.py, changepoint.py; regime-conditioned metrics
│   │   ├── layers/              # sizing.py (vol targeting, ATR sizing, caps),
│   │   │                        #   combine.py (uncorrelated-signal blending),
│   │   │                        #   router.py (regime→strategy routing); each independently
│   │   │                        #   toggleable with marginal Sharpe/DD/win-rate attribution
│   │   ├── portfolio/           # correlation.py (matrix + redundancy flags)
│   │   ├── profiles/            # models.py (Profile), mapping.py (slider→threshold +
│   │   │                        #   soft-ranking, explicit and logged), store.py (JSON on disk)
│   │   ├── reports/             # export.py (all CSVs), attrition.py (funnel report)
│   │   └── api/                 # FastAPI app: run pipeline as background job,
│   │                            #   poll status, fetch results; serves web/ statics
│   └── tests/                   # pytest — see Verification
├── web/index.html               # catfu SPA: profile faders, screener, funnel report, charts
├── Dockerfile, compose.yaml
└── docs/ARCHITECTURE.md
```

**Pipeline data flow:** run = `runs/<id>/` on disk (parquet + the PRD's CSVs: `sweep_results.csv`, `funnel_report.csv`, `sensitivity.csv`, `bootstrap.csv`, `cross_sectional.csv`, `regime_performance.csv`, `correlation_matrix.csv`). Frontend polls job status, then renders from JSON endpoints backed by those artifacts. Everything exportable (PRD §11.5).

**Honesty by design (enforced structurally):** funnel thresholds live in one frozen dataclass consumed by both engine and report, so the report always states the thresholds actually used; attrition counts are computed from the raw sweep file, not a curated subset; solid/fragile and regime-dependence flags are always rendered, never filterable-away in the UI.

## Milestones & delegation (orchestrator: me; implementor model per task)

| # | Milestone | Key deliverables | Implementor |
|---|---|---|---|
| M0 | Scaffolding | uv project, FastAPI skeleton, Docker, PLAN.md + CHANGELOG.md at root, CI hookup | **Sonnet** |
| M1 | Data + strategy library (PRD §4) | DataSource/cache, universe, all ~49 strategies + grid builder, total-backtest-count report | **Fable Low** (no-look-ahead correctness is subtle) |
| M2 | Backtest engine + funnel (PRD §5) | Vectorized engine, metrics, walk-forward stitching, six-filter funnel, sweep runner, attrition report | **Fable Low** (centerpiece; validation-critical) |
| M3 | Robustness (PRD §6) | Parameter sensitivity, bootstrap stress test, solid/fragile flags, CSV outputs | **Sonnet** (well-specified given M2), orchestrator review with extra scrutiny |
| M4 | Cross-sectional momentum (PRD §7) | Standalone L/S check, comparable WF scoring, side-by-side report, research-only labeling | **Fable Low** (turnover-cost + comparability subtleties) |
| M5 | Regime layer (PRD §9) | Detector protocol, HMM + 3 comparators, regime tagging, regime-conditioned performance in all reports | **Fable Low** |
| M6 | Layer stack + portfolio (PRD §10, §11.4) | Sizing/combine/routing layers, per-layer on/off with attribution, correlation matrix + redundancy flags | **Fable Low** |
| M7 | Profiles + slider mapping (PRD §8, §11.1) | Profile model/store, hybrid slider→threshold/ranking mapping, presets, screener filtering incl. hard-constraint exclusions | **Sonnet** |
| M8 | catfu SPA (PRD §8, §11.5, contract) | Single-file UI: fader-style profile sliders w/ snap points, profile save/load, run + poll, funnel attrition viz, sensitivity/bootstrap/comparison views, instrument-panel readouts, dark/light rocker toggle, CSV download links | **Fable Low** (contract compliance is high-craft) |
| M9 | Integration + hardening | End-to-end run, Docker verification, docs, CHANGELOG finalization, full test/lint/type pass | **Sonnet** |

Each milestone: implementor works from a written task spec (scope, interfaces, acceptance checks); I review the diff against the spec and PRD before accepting; one signed logical commit per milestone (or finer). CHANGELOG.md updated at each milestone.

## Verification

- **Unit/property tests (pytest), written with each milestone:**
  - **Look-ahead guard (critical):** for every strategy, assert position at *t* is unchanged when data after *t* is truncated/perturbed — a generic test parameterized over all ~49 strategies.
  - Metrics golden tests on synthetic series (known Sharpe/DD/trade counts).
  - Walk-forward: window boundaries, 70/30 split, stitched series length/ordering.
  - Funnel: each of the six filters exercised with pass/fail fixtures; attrition counts sum correctly.
  - Bootstrap determinism under fixed seed; cross-sectional rebalance dates and cost application; long-only clipping on the tradeable track.
  - Slider mapping: monotonicity (e.g., shallower DD tolerance ⇒ strictly tighter DD filter) and preset round-trips.
- **Quality gates:** `ruff check` + `ruff format --check` + `ty` clean; all functions typed; `uv run pytest` green — run per milestone, results reported honestly.
- **End-to-end:** `docker compose up`, run full pipeline on the real 30-asset universe (cached data), confirm thousands of backtests reported, all seven CSVs written, funnel attrition path printed and rendered.
- **UI:** preview at 375/768/1280 px, dark+light, and run the AESTHETIC_CONTRACT (catfu) §13 compliance checklist item-by-item before accepting M8.
- **Honesty spot-check:** verify negative/fragile results actually appear in the report for at least one known-weak family (e.g., naked single-asset momentum, which the PRD predicts scores near zero).

## Success criteria (from PRD §14, used as acceptance)

Single-session slider→screen→full-pipeline run; per-layer and per-regime attribution in the report; full attrition funnel with top-survivors table; no screened output violating hard constraints (no shorts on tradeable track, no uncapped-loss structures); correlation/redundancy surfaced; negative results visible by construction.
