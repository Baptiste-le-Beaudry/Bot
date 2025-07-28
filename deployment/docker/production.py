# Multi-stage Dockerfile for AI Trading Robot
# Optimized for production with minimal image size and security

# Build stage for Python dependencies
FROM python:3.11-slim-bookworm AS python-builder

# Build arguments
ARG PYTHON_VERSION=3.11
ARG PIP_NO_CACHE_DIR=1
ARG PIP_DISABLE_PIP_VERSION_CHECK=1
ARG PYTHONDONTWRITEBYTECODE=1
ARG PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    curl \
    libpq-dev \
    libssl-dev \
    libffi-dev \
    libxml2-dev \
    libxslt1-dev \
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    pkg-config \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Copy requirements files
COPY requirements.txt requirements-ml.txt* ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    if [ -f requirements-ml.txt ]; then \
        pip install --no-cache-dir -r requirements-ml.txt; \
    fi

# Build stage for application
FROM python:3.11-slim-bookworm AS app-builder

# Copy application code
WORKDIR /build
COPY . .

# Remove unnecessary files
RUN find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find . -type f -name "*.pyc" -delete && \
    find . -type f -name "*.pyo" -delete && \
    find . -type f -name ".DS_Store" -delete && \
    rm -rf .git .github tests docs deployment scripts/*.log

# Compile Python files for better performance
RUN python -m compileall -b .

# Final production stage
FROM python:3.11-slim-bookworm

# Metadata
LABEL maintainer="AI Trading Robot Team"
LABEL version="1.0.0"
LABEL description="High-performance AI Trading Robot"

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=UTC

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libssl3 \
    libxml2 \
    libxslt1.1 \
    libblas3 \
    liblapack3 \
    libgomp1 \
    curl \
    ca-certificates \
    tzdata \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# Create non-root user
RUN groupadd -r trader -g 1000 && \
    useradd -r -u 1000 -g trader -s /sbin/nologin -c "Trading Bot User" trader && \
    mkdir -p /app /app/logs /app/data /app/cache /app/models && \
    chown -R trader:trader /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=python-builder --chown=trader:trader /opt/venv /opt/venv

# Copy application from builder
COPY --from=app-builder --chown=trader:trader /build .

# Set Python path
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH"

# Create necessary directories and set permissions
RUN mkdir -p /app/config /app/logs /app/data /app/cache /app/models /tmp && \
    chown -R trader:trader /app /tmp && \
    chmod -R 755 /app && \
    chmod -R 777 /tmp

# Install additional tools for debugging (can be removed for minimal image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    htop \
    net-tools \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER trader

# Expose ports
EXPOSE 8080 8081 9090

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

# Use tini as entrypoint to handle signals properly
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command
CMD ["python", "-m", "core.engine"]

# ==============================================================================
# GPU Support Stage (optional build)
# Build with: docker build --target gpu -t trading-bot:gpu .
# ==============================================================================
FROM python:3.11-slim-bookworm AS gpu-builder

# NVIDIA CUDA base image arguments
ARG CUDA_VERSION=11.8.0
ARG CUDNN_VERSION=8
ARG UBUNTU_VERSION=22.04

# Install CUDA dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    curl \
    ca-certificates \
    && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add - \
    && echo "deb https://nvidia.github.io/libnvidia-container/stable/ubuntu22.04/\$(ARCH) /" > /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    cuda-libraries-${CUDA_VERSION} \
    libcudnn${CUDNN_VERSION} \
    && rm -rf /var/lib/apt/lists/*

# Copy everything from the main build
COPY --from=app-builder /build /app
COPY --from=python-builder /opt/venv /opt/venv

# Install additional GPU Python packages
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Final GPU stage
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS gpu

# Copy everything from GPU builder
COPY --from=gpu-builder /app /app
COPY --from=gpu-builder /opt/venv /opt/venv

# Install Python and runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    libpq5 \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Create user and set permissions
RUN groupadd -r trader -g 1000 && \
    useradd -r -u 1000 -g trader -s /sbin/nologin -c "Trading Bot User" trader && \
    chown -R trader:trader /app

WORKDIR /app
USER trader

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH="/app:$PYTHONPATH" \
    CUDA_VISIBLE_DEVICES=0

EXPOSE 8080 8081 9090

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health/live || exit 1

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "core.engine", "--gpu"]

# ==============================================================================
# Development Stage (for local development)
# Build with: docker build --target development -t trading-bot:dev .
# ==============================================================================
FROM python:3.11-slim-bookworm AS development

# Install development tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    vim \
    less \
    iputils-ping \
    telnet \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment
COPY --from=python-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install development dependencies
RUN pip install --no-cache-dir \
    ipython \
    ipdb \
    pytest \
    pytest-asyncio \
    pytest-cov \
    black \
    flake8 \
    mypy \
    bandit \
    pre-commit

# Create user
RUN groupadd -r trader -g 1000 && \
    useradd -r -u 1000 -g trader -m -s /bin/bash trader

WORKDIR /app

# Mount point for local development
VOLUME ["/app"]

USER trader

EXPOSE 8080 8081 9090

# Development command with auto-reload
CMD ["python", "-m", "core.engine", "--debug", "--reload"]