FROM ghcr.io/astral-sh/uv:python3.14-trixie-slim

WORKDIR /app

COPY engine/ engine/
COPY web/ web/

WORKDIR /app/engine
RUN uv sync --frozen --no-dev

ENV PATH="/app/engine/.venv/bin:${PATH}"
ENV FUNNEL_WEB_DIR="/app/web"

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "funnel.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
