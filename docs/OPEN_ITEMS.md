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
- **Docker build could not be verified in this environment.** `docker` CLI is
  present but no daemon is running (no Docker Desktop/OrbStack installed;
  `docker build .` fails with `dial unix .../docker.sock: connect: no such file or
  directory`). Verified instead via static review: `Dockerfile` copy layout, `uv
  sync --frozen --no-dev`, the `funnel.api.app:create_app` factory string, and
  `FUNNEL_WEB_DIR=/app/web` all line up with `_resolve_web_dir()` in
  `engine/src/funnel/api/app.py`. First real `docker build . && docker compose up`
  should still be watched once a daemon is available.
- **`.dockerignore` was missing `data/`** (cached yfinance parquet + saved profiles)
  — fixed in this pass; re-verify the built image doesn't bundle stale local cache
  data once a real build is possible.
- **Repo-level GitHub Actions workflows are currently failing** (unrelated to the
  funnel engine itself): Codespell, Gitlab Sync, Lint Code Base, and Security
  Scorecard all show `failure` as of this pass (`bash scripts/checkWorkflows.sh
  --dry-run`). Not investigated here — out of scope for the engine/UI work, but
  worth triaging before relying on CI green as a signal.
