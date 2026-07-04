# Architecture

The Funnel has two components:

- **`engine/`** — a Python 3.14 / FastAPI backend (package `funnel`). This is the
  quant core: data ingestion, the strategy library, the backtest engine, walk-forward
  validation, the six-filter survival funnel, robustness checks, regime detection,
  and the layer/portfolio stack. It also serves a small JSON API and hosts the
  frontend as static files.
- **`web/index.html`** — the Cyberdeck SPA: a single-file, zero-build, vanilla
  HTML/CSS/JS frontend (per `AESTHETIC_CONTRACT.md`). It drives the pipeline via the
  API, polls run status, and renders the funnel report, sensitivity/bootstrap views,
  and hand-built SVG charts. The current file is an M0 placeholder; M8 replaces it
  with the full SPA.

## Pipeline data flow

A single execution of the pipeline is a **run**, identified by an id and materialized
on disk under `runs/<id>/`. Each run writes seven CSV artifacts (plus cached parquet
data):

- `sweep_results.csv` — every strategy config × asset backtest result
- `funnel_report.csv` — six-filter pass/fail attrition per config
- `sensitivity.csv` — per-family parameter sensitivity (mean/std/positive-fraction)
- `bootstrap.csv` — bootstrap-reshuffle stress test (p5/p50/p95 Sharpe, worst-case DD)
- `cross_sectional.csv` — cross-sectional momentum research/diagnostic results
- `regime_performance.csv` — regime-conditioned performance breakdown
- `correlation_matrix.csv` — cross-strategy correlation / redundancy flags

The frontend polls the API for run status, then fetches JSON views backed by these
CSVs. All artifacts are also directly downloadable.

## Further reading

- [`PLAN.md`](../PLAN.md) — the full milestone plan and architectural decisions.
- [`AESTHETIC_CONTRACT.md`](../AESTHETIC_CONTRACT.md) — the binding design contract
  for all UI work in this repo.
