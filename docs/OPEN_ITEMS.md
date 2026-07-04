# Open Items

Known gaps and deferred work, stated plainly per the honesty-by-design principle in
`PLAN.md` — nothing here is hidden or silently worked around.

- **Real yfinance data path is untested end-to-end.** This dev environment has no
  outbound network access, so the full pipeline has only been verified against
  `FUNNEL_FAKE_DATA=1` (`SyntheticSource`). The first real run against live yfinance
  data (real universe fetch, `CachedSource` parquet caching, rate limits/outages)
  should be watched manually before being trusted unattended.
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
