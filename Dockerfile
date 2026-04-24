FROM python:3.12-slim-bookworm

# ------------------------------------------------------------------
# Install system-level build dependencies (for compiling TA-Lib)
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# Install TA-Lib C library from source
# ------------------------------------------------------------------
RUN wget https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make -j1 && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

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