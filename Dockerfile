# Multi-stage build for a smaller and more secure final image
# ===== Builder stage =====
FROM python:3.11-slim AS builder

# Set environment variables to reduce Python behavior
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create a non-privileged user
RUN groupadd -g 1001 appuser && \
    useradd -m -u 1001 -g appuser appuser

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install uvicorn[standard] && \
    # Optimize Python bytecode
    python -m compileall /opt/venv

# ===== Final stage =====
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    TF_CPP_MIN_LOG_LEVEL=2 \
    SAVEMODEL_PATH="/app/models/disease_prediction" \
    PORT=8000 \
    WORKERS=4 \
    MAX_CONNECTIONS=1000

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Add non-privileged user
RUN groupadd -g 1001 appuser && \
    useradd -m -u 1001 -g appuser appuser && \
    # Create directories with proper permissions
    mkdir -p /app/models/disease_prediction /app/logs && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser src/ /app/src/
COPY --chown=appuser:appuser gunicorn_conf.py /app/

# Create volume mount points
VOLUME ["/app/models/disease_prediction", "/app/logs"]

# Expose port
EXPOSE 8000

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Use entrypoint script for better signal handling and startup configuration
ENTRYPOINT ["sh", "-c", "gunicorn -k uvicorn.workers.UvicornWorker -c gunicorn_conf.py src.deploy:app"]