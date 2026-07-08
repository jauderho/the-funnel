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
- M3: Parameter sensitivity (`funnel.robustness.sensitivity`) — per-family
  mean/std/positive-fraction of OOS Sharpe across parameter configs
  (curve-fit red-flag detection); `write_sensitivity` → `sensitivity.csv`.
- M3: Bootstrap stress test (`funnel.robustness.bootstrap`) — drawdown
  percentiles and worst-case drawdown from order permutations of each
  survivor's stitched OOS returns (sequencing risk), Sharpe p5/p50/p95 from
  with-replacement resamples, deterministic under a fixed seed; solid/fragile
  verdict against the funnel's max-DD floor; `write_bootstrap` → `bootstrap.csv`.
- M4: Cross-sectional momentum research check (`funnel.momentum.cross_sectional`)
  — monthly (21-day) rank of the universe by trailing return at 3m/6m/12-1
  lookbacks, long top third / short bottom third with turnover costs, scored
  with the identical 5-window walk-forward discipline; side-by-side comparison
  against single-asset momentum families and a plain-language verdict; flagged
  research-only (long/short conflicts with the no-short tradeable track);
  `write_cross_sectional` → `cross_sectional.csv`.
- M5: Regime detection (`funnel.regime`) — `Regime` (trending/choppy),
  `RegimeDetector` protocol, and four causal detectors: 200-day MA filter,
  expanding-quantile realized-vol, ruptures change-point (expanding refit),
  and a 2-state Gaussian HMM on [return, vol] with periodic expanding-window
  refits (never fit-on-full-then-label-the-past); detector comparison and
  agreement matrix; `regime_conditioned_metrics` and
  `write_regime_performance` → `regime_performance.csv`.
- M6: Strategy layer stack (`funnel.layers`) — position sizing (volatility
  targeting and ATR risk sizing, both causal, leverage-capped at 1.0),
  uncorrelated-signal combining with greedy correlation-bounded selection,
  and regime routing (zero exposure outside a strategy's preferred regime);
  `run_stack` with independent `LayerToggles` per layer and
  `attribution_table` isolating each layer's marginal Sharpe/drawdown/
  win-rate contribution → `layer_attribution.csv`.
- M6: Portfolio view (`funnel.portfolio.correlation`) — pairwise strategy
  correlation matrix with a minimum-overlap guard and redundancy flags for
  highly correlated pairs → `correlation_matrix.csv`.

- M7: Risk/reward profiles (`funnel.profiles`) — four validated 0–100 sliders
  with the hybrid mapping: drawdown-tolerance and risk-tolerance hard-map
  onto funnel thresholds (max-DD floor -0.15…-0.50, Sharpe ceiling 2.0…4.0,
  min trades 40…20, all linear and monotone), capital and time-horizon
  soft-re-rank only (crypto niche penalty, turnover preference); per-slider
  `explain_mapping` for UI display; two presets ("Retirement Core",
  "Swing Sandbox") plus atomic JSON profile persistence; profile-aware
  screener with hard-constraint flags (research-only long/short excluded
  from the tradeable track) and an intraday-unsupported warning.
- M7.5: Pipeline orchestration (`funnel.pipeline`) — one call runs
  data → profile thresholds → sweep → attrition → sensitivity → bootstrap →
  cross-sectional → regime → layer attribution → correlation → screen,
  writing all artifacts (seven CSVs + `layer_attribution.csv` +
  `report.json`) under `runs/<id>/`; zero-survivor runs complete honestly
  with warnings recorded, never swallowed.
- M7.5: API — profiles CRUD, `POST /api/runs` (serial background jobs with
  per-stage status mirrored to `status.json`), status/report/artifact
  endpoints (artifact-name whitelist and strict run-id validation guarding
  against path traversal), live `GET /api/mapping/preview` for slider
  feedback, and a `FUNNEL_FAKE_DATA=1` synthetic source for offline UI dev.

- M8: catfu single-file SPA (`web/index.html`) — zero-build instrument-panel
  UI per AESTHETIC_CONTRACT.md: profile deck hero (four keyboard-operable
  faders with LCD readouts, preset snap, live mapping readout from
  `/api/mapping/preview`, intraday warning), run console with per-stage LED
  progress, and report views for funnel attrition (direct-labeled stepped
  bars + exact thresholds applied), robustness (sensitivity + solid/fragile
  bootstrap verdicts), cross-sectional (research-only banner), regime
  (single-hue agreement matrix), layer attribution, and correlation
  redundancy — every section with CSV artifact downloads, honest zero-state
  handling, dark/light rocker theming, and WCAG AA-verified contrast.
- M8: `.claude/launch.json` dev-preview config (uvicorn on port 8731 with
  synthetic data via `FUNNEL_FAKE_DATA=1`).

- M9: Docs — `docs/ARCHITECTURE.md` (module map, 12-stage pipeline, artifact
  and endpoint inventory, env vars), `docs/OPEN_ITEMS.md` (honest gaps:
  real-data path untested in the network-blocked dev env, Docker build
  unverified without a daemon, options overlay deferred to v2, intraday
  unsupported in v1), and a README refresh with the fake-data dev mode.
- M9: `.dockerignore` now excludes `data/` (local cache/profiles) from
  image builds.
- V2-M1: Options pricing core (`funnel.options.pricing`) — Black-Scholes-
  Merton price/delta/risk-neutral P(ITM at expiry) with documented q=0
  total-return-frame ground rules (model prices, never market prices;
  "assignment probability" is always the labeled model P(ITM) proxy),
  closed-form strike-for-delta via Acklam's inverse-normal approximation,
  and a strictly causal realized-vol proxy (`synthetic_iv`) with a
  configurable volatility-risk-premium multiplier and floor. Golden-value,
  put-call-parity, monotonicity, edge-case, round-trip, and
  truncation-invariance tests.

### Changed

- Design language switched from "Cyberdeck" to "catfu" (cassette-futurism):
  AESTHETIC_CONTRACT.md replaced; PLAN.md M8 updated (fader-style profile
  sliders supersede the PRD's gradient sliders; contract §13 checklist).

### Fixed

- Docker build: multi-stage Dockerfile — a `build-essential` builder stage
  compiles hmmlearn/ruptures from source (no cp314/linux-aarch64 wheels),
  the runtime stage stays compiler-free; container CMD preloads libstdc++
  (resolved via ldconfig, amd64/arm64-portable) to fix an RTTI
  symbol-visibility crash in hmmlearn's compiled extension under Python's
  RTLD_LOCAL dlopen, and invokes uvicorn from the venv directly (no uv
  re-resolution at container start).
- `.dockerignore` now uses `**/` globs so nested `engine/.venv`, caches,
  and bytecode are excluded — build context dropped from 419 MB to ~5 kB.
- Live data: `YFinanceSource.fetch` now falls back once (per process) to
  yfinance's "csrf" cookie strategy when a download comes back empty —
  yfinance 1.4.1's default "basic" strategy bootstraps via `fc.yahoo.com`,
  which refuses connections on some networks, previously yielding empty
  frames and honest-but-useless zero-asset pipeline runs. The fallback is
  logged, guarded against private-API drift, and covered by offline tests.
- `.gitignore`: scoped `data/` and `runs/` to the repo root (`/data/`,
  `/runs/`) — the unanchored `data/` pattern had been silently excluding
  the entire `engine/src/funnel/data/` package (DataSource protocol,
  yfinance source, cache, universe) from version control since M1; the
  package is now actually tracked.
- Zero-backtest runs on DNS-filtered networks diagnosed and documented:
  `fc.yahoo.com` (yfinance's cookie-bootstrap host) is on common DNS
  blocklists; when sinkholed, every fetch returns empty and the pipeline
  honestly reports a zero-asset run. `compose.yaml` now carries a commented
  `extra_hosts` stopgap and the durable fix (DNS whitelist) is documented in
  `docs/OPEN_ITEMS.md`. First real-data run verified end-to-end: 31 assets,
  4650 backtests, 10 survivors.
