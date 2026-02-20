FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    HC_ARTIFACT_PATH=/app/artifacts/model_bundle.joblib

COPY pyproject.toml README.md ./
COPY src ./src
RUN uv sync --no-dev
RUN mkdir -p /app/artifacts

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "homecredit_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
