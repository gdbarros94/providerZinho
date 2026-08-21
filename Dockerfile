# Use a slim Python 3.11 image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODELS_DIR=/app/models \
    HOST=0.0.0.0 \
    PORT=9880

# Install system dependencies required for building C extensions (llama-cpp, etc.) and GPU detection tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    pciutils \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy dependency specifications first for Docker caching
COPY pyproject.toml /app/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools \
    && pip install --no-cache-dir .

# Copy application source code
COPY edge_ai_provider /app/edge_ai_provider

# Create volume mount directory for GGUF models
RUN mkdir -p /app/models

# Expose server port
EXPOSE 9880

# Healthcheck endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:9880/v1/health || exit 1

# Start provider server
CMD ["python", "-m", "edge_ai_provider.main"]
