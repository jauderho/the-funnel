# Open Items

Known gaps and deferred work, stated plainly per the honesty-by-design principle in
`PLAN.md` — nothing here is hidden or silently worked around.

- **Real yfinance data path is now verified end-to-end (resolved), with one
  network caveat.** A watched "Retirement Core" run inside the compose container
  fetched all 31 assets and completed 4650 real backtests (2217 positive OOS →
  596 cleared the Sharpe floor → 10 survivors, no warnings). Caveat discovered
  on the way: yfinance bootstraps its auth cookie from `fc.yahoo.com`, which is
  on common DNS-blocklist feeds — on this LAN the resolver sinkholes it to
  `0.0.0.0`, every fetch returns empty, and the pipeline honestly reports a
  zero-asset run. Its fallback "csrf" consent flow yields no token from US IPs,
  so both yfinance cookie strategies dead-end when that host is blocked.
  Durable fix: whitelist `fc.yahoo.com` in the DNS blocker — applied on this
  LAN 2026-07-08 and re-verified with a clean (pin-free) container: 31 assets,
  4650 backtests, 16 survivors, no warnings. The commented `extra_hosts` pin in
  `compose.yaml` remains as a documented stopgap for other DNS-filtered
  networks, and the `YFinanceSource` csrf fallback (with offline tests) stays
  as belt-and-braces.
- **The host's docker CLI dispatch is now fixed (resolved, 2026-07-08).** The
  OrbStack CLI wedge described below (every personality answering as Docker
  Compose) is gone after the user completed the pending OrbStack update:
  `docker build -t the-funnel-funnel .` and `docker run` both work normally
  again on this machine. Left here for context in case it recurs: after an
  OrbStack update/handoff on 2026-07-07, the `docker-tools` multiplexer
  answered as Docker Compose ("v5.3.0") for every personality (`docker`,
  `docker-buildx`, `docker-compose`), so `docker build` / `docker compose`
  failed with mangled-argument errors while the daemon itself stayed healthy
  (Engine 29.4.0 on `~/.orbstack/run/docker.sock`; the v2 image rebuild used
  the daemon REST API as a workaround before the fix).
- **Options overlay (v2, PRD §11.3) is model-priced, not market-priced.** No
  historical option-chain data is available from free sources, so every price,
  Greek, and assignment probability in `overlay_results.csv` / the OVERLAYS UI
  is a synthetic Black-Scholes-Merton value computed from the underlying's
  adjusted-close series (q=0) and a causal realized-vol proxy — never an
  observed market quote or implied vol. Consequences worth stating plainly:
  (1) the vol-risk-premium multiplier used to scale realized vol into a
  proxy-IV is a crude stand-in for the market's actual IV risk premium, tuned
  by feel rather than fit to observed option prices; (2) dividend-driven early
  assignment is not modeled at all (BSM with q=0 on adjusted closes has no
  concept of an ex-dividend date), so real-world assignment timing on
  dividend-paying names will differ from `mean_model_prob_itm`; (3) every
  reported yield, assignment rate, and overlay-vs-hold comparison is a model
  estimate conditioned on these simplifications, not a tradeable guarantee —
  the `model_risk_caveat` string is carried on every overlay `report.json` and
  rendered permanently in the OVERLAYS UI banner precisely because of this.
- **Intraday is unsupported in v1.** Only EOD data is fetched/backtested; slider
  positions implying intraday behavior are labeled "unsupported in v1" in the UI,
  not silently downgraded.
- **Docker build is now verified end-to-end (resolved).** A real `docker build .`
  with OrbStack running surfaced two problems the earlier static review couldn't
  catch, both now fixed: (1) `hmmlearn`/`ruptures` have no prebuilt wheel for
  cp314/linux-aarch64 and source-build, but the single-stage image had no
  compiler — fixed with a multi-stage `Dockerfile` (`build-essential` in a
  `builder` stage only, copied `.venv` into a slim runtime stage); (2) the
  compiled `hmmlearn._hmmc` extension resolves libstdc++ RTTI/vtable symbols
  (`_ZTVN10__cxxabiv120__function_type_infoE`) via `RTLD_LOCAL` dlopen, which
  fails to interpose across independently-loaded extension modules on this base
  image — fixed by preloading libstdc++ globally in the container `CMD`
  (`LD_PRELOAD` resolved via `ldconfig` so it works on both amd64 and arm64).
  `.dockerignore` also switched from top-level (`.venv/`, `data/`) to `**/`-glob
  patterns, since context-root-relative patterns didn't match `engine/.venv` —
  this cut build context from 419MB to 4.72kB. `docker build -t funnel:test .`
  and `docker compose up -d` both succeed; `curl localhost:8000/api/health`
  returns `{"status":"ok","version":"0.1.0"}` and `/` serves the SPA HTML.
- **Real-data options-overlay run is now verified end-to-end (resolved).** A
  watched `POST /api/overlays {"symbols":["AAPL","MSFT"]}` run inside a
  rebuilt v2 container fetched fresh real yfinance data (2010-01-01 to
  2025-12-31, confirmed via new parquet cache files timestamped at run start)
  and completed all 36 overlay configs × 2 symbols = 72 backtests in ~7.5s,
  no warnings, transparency counts 72/72 as expected, `model_risk_caveat`
  present. Results plainly show non-rosy outcomes as intended: 37 of 72 rows
  (51%) have negative `oos_sharpe_vs_hold`, all 16 vertical-spread rows are
  negative (mean vs-hold Sharpe -0.46), and 62 of 72 rows (86%) got a
  `fragile` bootstrap verdict vs. 10 `solid` — the module does not paper over
  weak overlay results. Worst row: an AAPL bear-put vertical (30 DTE, -0.30
  delta short leg) at oos_sharpe_vs_hold = -0.94 (overlay Sharpe -1.13 vs.
  underlying -0.19). Best row: an MSFT cash-secured put (21 DTE, -0.15 delta)
  at +0.62 (overlay Sharpe +0.83 vs. underlying +0.21). `premium_collected_annualized`
  ranged 0.00-2.42 (mean 0.43); `mean_model_prob_itm` ranged 0.12-0.72 (mean
  0.29). `overlay_results.csv` downloaded at HTTP 200 with 72 data rows,
  matching the report exactly. Two things noticed along the way, worth
  tracking: (1) the underlying's own walk-forward-stitched OOS Sharpe was
  negative for AAPL (-0.19, max DD -67.5%) despite the 2010-2025 window's
  well-known bull run — the stitched-OOS metric (v1's windowing, reused
  as-is here) evidently weights volatile sub-periods more than a naive
  full-period buy-and-hold return would, so "beats hold" in this report means
  "beats hold's own walk-forward OOS Sharpe," not "beats a simple buy-and-hold
  total return" — RESOLVED: a one-line clarification now sits directly above
  the comparison table in the OVERLAYS view. (2) `n_assignments` was 0 across
  all 72 rows, structurally: every grid config used the default
  `roll_at_dte=5`, so `simulate_overlay`'s scheduled-roll check always fired
  before true expiry and the assignment/settlement path was unreachable in
  production — a user could wrongly conclude zero assignment risk in 15 years
  of data. RESOLVED: `build_overlay_grid()` now ships 10 hold-to-expiry
  (`roll_at_dte=0`) variants (4 covered-call, 4 cash-secured-put, 2 vertical;
  46 configs total), including the avoid_assignment=True + hold-to-expiry
  combination that matches the "avoid assignment when possible" use case, and
  a sweep-level test proves a hold-to-expiry covered call reports nonzero
  `n_assignments` (32 events on the test fixture). Note the default-grid
  roll-at-5-DTE configs still legitimately report 0 assignments — that now
  reflects a real policy choice, contrasted in the same report against the
  hold-to-expiry rows.
- **Repo-level GitHub Actions workflows are currently failing** (unrelated to the
  funnel engine itself): Codespell, Gitlab Sync, Lint Code Base, and Security
  Scorecard all show `failure` as of this pass (`bash scripts/checkWorkflows.sh
  --dry-run`). Not investigated here — out of scope for the engine/UI work, but
  worth triaging before relying on CI green as a signal.
