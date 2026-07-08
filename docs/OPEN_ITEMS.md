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
- **The host's docker CLI dispatch is currently broken (OrbStack, machine-level,
  not this repo).** After an OrbStack update/handoff on 2026-07-07, the
  `docker-tools` multiplexer answers as Docker Compose ("v5.3.0") for every
  personality (`docker`, `docker-buildx`, `docker-compose`), so `docker build`
  / `docker compose` fail with mangled-argument errors while the daemon itself
  is healthy (Engine 29.4.0 answers on `~/.orbstack/run/docker.sock`; this
  repo's image was rebuilt via the daemon REST API as a workaround). An app
  restart didn't clear it; `orbctl update` opened the GUI updater — completing
  that update (or reinstalling OrbStack) should restore the CLI.
- **Options overlay (PRD §11.3) is deferred to v2.** A user decision — v1 keeps a
  pluggable strategy/instrument seam but implements nothing options-specific.
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
  Still pending: a real yfinance-data compose run (see the item above) has not
  been watched inside this container.
- **Repo-level GitHub Actions workflows are currently failing** (unrelated to the
  funnel engine itself): Codespell, Gitlab Sync, Lint Code Base, and Security
  Scorecard all show `failure` as of this pass (`bash scripts/checkWorkflows.sh
  --dry-run`). Not investigated here — out of scope for the engine/UI work, but
  worth triaging before relying on CI green as a signal.
