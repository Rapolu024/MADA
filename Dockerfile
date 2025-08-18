# MADA - Medical AI Diagnosis Assistant
# Multi-stage Docker build for production deployment

# Base Python image with security updates
FROM python:3.9-slim-bullseye as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user for security
RUN groupadd --gid 1000 mada && \
    useradd --uid 1000 --gid mada --shell /bin/bash --create-home mada

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY . .

# Change ownership to mada user
RUN chown -R mada:mada /app

# Switch to non-root user
USER mada

# Expose ports
EXPOSE 8501 8000

# Command for development
CMD ["streamlit", "run", "src/dashboard/mada_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Production stage
FROM base as production

# Install production requirements only
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=mada:mada . .

# Create necessary directories
RUN mkdir -p /app/data/models /app/data/raw /app/data/processed /app/logs && \
    chown -R mada:mada /app

# Switch to non-root user
USER mada

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose port
EXPOSE 8501

# Production command
CMD ["streamlit", "run", "src/dashboard/mada_dashboard.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false"]
