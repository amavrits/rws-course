FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first for better caching
COPY pyproject.toml uv.lock* ./

# Install dependencies
RUN uv sync --frozen --no-dev --no-install-project

# Copy application code
COPY apps/ ./apps/
COPY src/ ./src/
COPY main/ ./main/

# Install the project itself
RUN uv sync --frozen --no-dev

EXPOSE 8050

# Run with gunicorn for production
CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:8050", "--workers", "2", "apps.main_app:server"]
