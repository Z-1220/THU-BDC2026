FROM python:3.12-slim-bookworm

# ------------------------------------------------------------------
# Install uv (fast Python package manager)
# ------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# ------------------------------------------------------------------
# Copy dependency files and install Python dependencies
# ------------------------------------------------------------------
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# ------------------------------------------------------------------
# Copy the entire application code (including scripts, src, etc.)
# ------------------------------------------------------------------
COPY . .

# ------------------------------------------------------------------
# Environment variables
# ------------------------------------------------------------------
ENV PATH="/app/.venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/lib:/usr/local/lib"

# ------------------------------------------------------------------
# Ensure mandatory scripts are executable
# ------------------------------------------------------------------
RUN chmod +x /app/init.sh /app/train.sh /app/test.sh /app/run.sh

# ------------------------------------------------------------------
# Set ENTRYPOINT to automatically run the full pipeline
# This takes priority over docker-compose.yml "command"
# ------------------------------------------------------------------
ENTRYPOINT ["/bin/bash", "/app/run.sh"]
