FROM node:22-slim AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY frontend ./
RUN npm run build


FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=5051

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --no-dev

COPY . .
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

EXPOSE 5051

CMD ["sh", "-c", "exec uv run --no-dev uvicorn anti_fraud_explorer.api:app --host ${HOST:-0.0.0.0} --port ${PORT:-5051}"]
