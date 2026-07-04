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
