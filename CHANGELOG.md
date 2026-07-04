# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- M0: Repo scaffolding — `engine/` uv-managed FastAPI project skeleton, placeholder
  `web/index.html`, Docker/Compose deployment files, and project docs (`PLAN.md`,
  `docs/ARCHITECTURE.md`).
- M0: `GET /api/health` endpoint serving the engine version, and static hosting of `web/`.
- M1: Data layer — `DataSource` protocol, `YFinanceSource`, and a parquet-backed
  `CachedSource` (cache dir via `FUNNEL_DATA_DIR`, default `data/cache/`); the
  31-asset default universe (`funnel.data.universe`) with min-history filtering.
- M1: Strategy library — 47 strategy families (19 trend, 12 mean-reversion, 6
  volume, 3 volatility, 4 pattern, 3 composite) under a strict causal-only,
  no-look-ahead position contract, plus a shared hand-rolled indicator module
  (no TA library dependency).
- M1: `funnel.strategies.grid` config builder — 150 (strategy, parameter-set)
  configs across all six categories, with `total_backtest_count` and
  `summarize_grid` for transparent "N configs × M assets = K backtests" reporting.
- M2: Backtest engine (`funnel.backtest.engine`) — `strategy_returns` turns a
  position series into net daily returns (position_t applied to the t→t+1
  close-to-close return, transaction costs charged exactly on position changes
  via `|Δposition| × bps/1e4`), plus `cost_bps_for` mapping asset class to the
  configured per-side cost rate.
- M2: Metrics (`funnel.backtest.metrics`) — `sharpe`, `max_drawdown`,
  `drawdown_duration`, `trade_count`, `cagr`, and `win_rate`, all on daily
  return/position series with documented edge-case behavior (zero/near-zero
  variance and empty series return 0.0 rather than NaN/inf).
- M2: Walk-forward validation (`funnel.backtest.walkforward`) — `walk_forward_oos`
  splits an asset's history into 5 contiguous, near-equal windows, each
  independently recomputed from its own 70% in-sample / 30% out-of-sample
  split (indicators warm up strictly within each window, never leaking
  pre-window history), stitching all OOS tails and IS heads into combined
  series. Raises `InsufficientHistoryError` when any window's OOS segment
  is too short (< 30 rows), which the sweep runner catches and records as
  a skipped pair.
- M2: Six-filter survival funnel (`funnel.backtest.funnel`) — `apply_funnel`
  evaluates all six filters (max-DD floor, min/max OOS Sharpe, the
  OOS/IS overfit-signature gap, min OOS trades, positive IS Sharpe) from
  `FunnelThresholds`, always recording all six per-filter outcomes even
  after an earlier failure.
- M2: Sweep runner (`funnel.backtest.sweep`) — `run_sweep` runs every
  (config, asset) pair through walk-forward + the funnel into one DataFrame,
  printing the "N configs × M assets = K backtests" transparency line;
  `write_sweep_results` writes `sweep_results.csv`.
- M2: Attrition report (`funnel.reports.attrition`) — `build_attrition_report`
  computes total/skipped/survived counts and per-category/per-family survival
  rates and mean OOS Sharpe from the raw sweep DataFrame, embedding the exact
  `FunnelThresholds` applied; `render_text`, `to_dict`, and
  `write_funnel_report` (→ `funnel_report.csv`) expose it as text, dict, and CSV.
