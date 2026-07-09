# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install dependencies first (better layer caching)
COPY pyproject.toml README.md ./
RUN uv venv /opt/venv && uv pip install --python /opt/venv/bin/python -e .

# Copy the rest of the application
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

ENV PATH="/opt/venv/bin:${PATH}"

RUN mkdir -p data/raw data/processed models metrics logs

ENTRYPOINT ["python", "scripts/train.py"]
