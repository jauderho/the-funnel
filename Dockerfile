FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/engine
COPY engine/ .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --compile-bytecode

FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

# Least privilege (Checkov CKV_DOCKER_3): run as a dedicated non-root user.
# The runs/ and data/ bind mounts must be writable by uid 1000 on generic
# Linux hosts (chown them or override `user:` in compose); OrbStack/Docker
# Desktop map host-user permissions transparently.
RUN groupadd --gid 1000 funnel && useradd --uid 1000 --gid funnel --no-create-home funnel

WORKDIR /app
COPY --from=builder --chown=funnel:funnel /app/engine /app/engine
COPY --chown=funnel:funnel web/ web/
RUN mkdir -p /app/runs /app/data && chown funnel:funnel /app/runs /app/data

ENV PATH="/app/engine/.venv/bin:${PATH}"
ENV FUNNEL_WEB_DIR="/app/web"

WORKDIR /app/engine

EXPOSE 8000

USER funnel

# CKV_DOCKER_2: liveness probe against the app's own health endpoint (no
# curl in the slim image; stdlib urllib keeps the image dependency-free).
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4).status == 200 else 1)"]

# hmmlearn's compiled _hmmc extension resolves libstdc++ RTTI/vtable symbols
# (e.g. _ZTVN10__cxxabiv120__function_type_infoE) via RTLD_LOCAL dlopen, which
# fails to interpose across independently-loaded extension modules on this
# base image. Preloading libstdc++ globally (resolved via ldconfig, so this
# works on both amd64 and arm64) fixes symbol resolution without touching
# engine code.
CMD ["sh", "-c", "LD_PRELOAD=$(ldconfig -p | grep -m1 libstdc++.so.6 | awk '{print $NF}') exec uvicorn funnel.api.app:create_app --factory --host 0.0.0.0 --port 8000"]
