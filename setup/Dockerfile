# The production image for the API: self-contained, runtime dependencies only, non-root, no reload.
# docker-compose.yml runs this same image with two development overrides (a mounted src/ and --reload).
FROM python:3.13-slim

# uv, pinned to the version on our laptops.
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime dependencies plus the api extra; --no-dev leaves out Jupyter, pytest and black.
COPY pyproject.toml uv.lock README.md .python-version ./
RUN uv sync --frozen --no-dev --no-install-project --extra api

# The code is copied in, not mounted: the image is complete on its own.
COPY src ./src
RUN uv sync --frozen --no-dev --extra api

# A service never runs as root.
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# Lets Docker tell a hung service from a healthy one. Uses the standard library: no curl in the image.
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# One process, no --reload: nothing changes on disk in production, and a restart would drop requests.
CMD ["uvicorn", "aueb_agents_ca.api:app", "--host", "0.0.0.0", "--port", "8000"]
