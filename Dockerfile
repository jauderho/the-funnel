FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/engine
COPY engine/ .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --compile-bytecode

FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

WORKDIR /app
COPY --from=builder /app/engine /app/engine
COPY web/ web/

ENV PATH="/app/engine/.venv/bin:${PATH}"
ENV FUNNEL_WEB_DIR="/app/web"

WORKDIR /app/engine

EXPOSE 8000

# hmmlearn's compiled _hmmc extension resolves libstdc++ RTTI/vtable symbols
# (e.g. _ZTVN10__cxxabiv120__function_type_infoE) via RTLD_LOCAL dlopen, which
# fails to interpose across independently-loaded extension modules on this
# base image. Preloading libstdc++ globally (resolved via ldconfig, so this
# works on both amd64 and arm64) fixes symbol resolution without touching
# engine code.
CMD ["sh", "-c", "LD_PRELOAD=$(ldconfig -p | grep -m1 libstdc++.so.6 | awk '{print $NF}') exec uvicorn funnel.api.app:create_app --factory --host 0.0.0.0 --port 8000"]
