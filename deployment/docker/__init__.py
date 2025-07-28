"""
Docker Deployment Sub-package
============================

Configuration et helpers pour le déploiement Docker.
Inclut Dockerfile templates et docker-compose configurations.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

# Configuration Docker par défaut
DOCKER_CONFIG = {
    "base_image": "python:3.11-slim",
    "cuda_base_image": "nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04",
    "working_dir": "/app",
    "user": "trader",
    "uid": 1000,
    "expose_ports": [8080, 9090],  # API et Prometheus
    "volumes": [
        "/app/config",
        "/app/logs",
        "/app/data"
    ],
    "build_args": {
        "PYTHON_VERSION": "3.11",
        "PIP_NO_CACHE_DIR": "1",
        "PYTHONUNBUFFERED": "1"
    }
}


def generate_dockerfile(
    use_cuda: bool = False,
    production: bool = True
) -> str:
    """Génère un Dockerfile optimisé"""
    
    base_image = DOCKER_CONFIG["cuda_base_image"] if use_cuda else DOCKER_CONFIG["base_image"]
    
    dockerfile = f"""# Trading Bot Dockerfile
# Generated automatically - DO NOT EDIT MANUALLY

FROM {base_image} AS builder

# Build arguments
ARG PYTHON_VERSION={DOCKER_CONFIG['build_args']['PYTHON_VERSION']}
ARG PIP_NO_CACHE_DIR={DOCKER_CONFIG['build_args']['PIP_NO_CACHE_DIR']}

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    curl \\
    git \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

"""
    
    if use_cuda:
        dockerfile += """# Install Python for CUDA image
RUN apt-get update && apt-get install -y python${PYTHON_VERSION} python3-pip

"""
    
    dockerfile += f"""# Create app user
RUN useradd -m -u {DOCKER_CONFIG['uid']} {DOCKER_CONFIG['user']}

# Set working directory
WORKDIR {DOCKER_CONFIG['working_dir']}

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && \\
    pip install -r requirements.txt

# Copy application code
COPY --chown={DOCKER_CONFIG['user']}:{DOCKER_CONFIG['user']} . .

"""
    
    if production:
        dockerfile += """# Production optimizations
RUN python -m compileall .
RUN find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

"""
    
    dockerfile += f"""# Switch to non-root user
USER {DOCKER_CONFIG['user']}

# Expose ports
"""
    
    for port in DOCKER_CONFIG['expose_ports']:
        dockerfile += f"EXPOSE {port}\n"
    
    dockerfile += """
# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8080/health')"

# Default command
CMD ["python", "-m", "scripts.run_trading_bot"]
"""
    
    return dockerfile


def generate_dockerignore() -> str:
    """Génère un .dockerignore"""
    return """# Docker ignore file
.git/
.github/
.pytest_cache/
.mypy_cache/
__pycache__/
*.pyc
*.pyo
*.pyd
.env
.env.*
venv/
env/
.venv/
logs/
data/
models/
*.log
*.db
*.sqlite
.DS_Store
.idea/
.vscode/
*.swp
*.swo
deployment/terraform/
deployment/k8s/
tests/
docs/
README.md
.dockerignore
Dockerfile
docker-compose*.yml
"""


def generate_docker_compose_dev() -> Dict[str, Any]:
    """Génère docker-compose pour développement"""
    return {
        "version": "3.8",
        "services": {
            "trading-bot": {
                "build": {
                    "context": ".",
                    "dockerfile": "deployment/docker/Dockerfile",
                    "args": {
                        "PYTHON_VERSION": "3.11"
                    }
                },
                "environment": {
                    "TRADING_ENV": "development",
                    "LOG_LEVEL": "DEBUG",
                    "DATABASE_URL": "postgresql://trader:password@postgres:5432/trading",
                    "REDIS_URL": "redis://redis:6379/0"
                },
                "volumes": [
                    "./config:/app/config:ro",
                    "./logs:/app/logs",
                    "./data:/app/data",
                    ".:/app:ro"  # Mount code for hot reload
                ],
                "ports": [
                    "8080:8080",
                    "9090:9090"
                ],
                "depends_on": [
                    "postgres",
                    "redis"
                ],
                "restart": "unless-stopped",
                "command": "python -m scripts.run_trading_bot --dev"
            },
            "postgres": {
                "image": "timescale/timescaledb:latest-pg15",
                "environment": {
                    "POSTGRES_USER": "trader",
                    "POSTGRES_PASSWORD": "password",
                    "POSTGRES_DB": "trading"
                },
                "volumes": [
                    "postgres-data:/var/lib/postgresql/data",
                    "./deployment/docker/init-db.sql:/docker-entrypoint-initdb.d/init.sql:ro"
                ],
                "ports": [
                    "5432:5432"
                ],
                "healthcheck": {
                    "test": ["CMD-SHELL", "pg_isready -U trader"],
                    "interval": "10s",
                    "timeout": "5s",
                    "retries": 5
                }
            },
            "redis": {
                "image": "redis:7-alpine",
                "command": "redis-server --appendonly yes",
                "volumes": [
                    "redis-data:/data"
                ],
                "ports": [
                    "6379:6379"
                ],
                "healthcheck": {
                    "test": ["CMD", "redis-cli", "ping"],
                    "interval": "10s",
                    "timeout": "5s",
                    "retries": 5
                }
            },
            "prometheus": {
                "image": "prom/prometheus:latest",
                "volumes": [
                    "./deployment/docker/prometheus.yml:/etc/prometheus/prometheus.yml:ro",
                    "prometheus-data:/prometheus"
                ],
                "ports": [
                    "9091:9090"
                ],
                "command": [
                    "--config.file=/etc/prometheus/prometheus.yml",
                    "--storage.tsdb.path=/prometheus",
                    "--web.console.libraries=/usr/share/prometheus/console_libraries",
                    "--web.console.templates=/usr/share/prometheus/consoles"
                ]
            },
            "grafana": {
                "image": "grafana/grafana:latest",
                "environment": {
                    "GF_SECURITY_ADMIN_PASSWORD": "admin",
                    "GF_USERS_ALLOW_SIGN_UP": "false"
                },
                "volumes": [
                    "grafana-data:/var/lib/grafana",
                    "./deployment/docker/grafana-dashboards:/etc/grafana/provisioning/dashboards:ro",
                    "./deployment/docker/grafana-datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml:ro"
                ],
                "ports": [
                    "3000:3000"
                ],
                "depends_on": [
                    "prometheus"
                ]
            }
        },
        "volumes": {
            "postgres-data": {},
            "redis-data": {},
            "prometheus-data": {},
            "grafana-data": {}
        },
        "networks": {
            "default": {
                "name": "trading-network"
            }
        }
    }


def create_docker_files(output_dir: Path):
    """Crée tous les fichiers Docker nécessaires"""
    docker_dir = output_dir / "deployment" / "docker"
    docker_dir.mkdir(parents=True, exist_ok=True)
    
    # Dockerfile principal
    (docker_dir / "Dockerfile").write_text(generate_dockerfile())
    
    # Dockerfile GPU
    (docker_dir / "Dockerfile.cuda").write_text(generate_dockerfile(use_cuda=True))
    
    # Dockerfile dev
    (docker_dir / "Dockerfile.dev").write_text(generate_dockerfile(production=False))
    
    # .dockerignore
    (output_dir / ".dockerignore").write_text(generate_dockerignore())
    
    # docker-compose files
    import yaml
    
    with open(output_dir / "docker-compose.yml", "w") as f:
        yaml.dump(generate_docker_compose_dev(), f, default_flow_style=False)
    
    # Scripts helper
    build_script = """#!/bin/bash
# Build Docker images

set -e

echo "Building Trading Bot Docker images..."

# Build production image
docker build -f deployment/docker/Dockerfile -t trading-bot:latest .

# Build GPU image if NVIDIA runtime available
if docker info | grep -q nvidia; then
    echo "Building GPU-enabled image..."
    docker build -f deployment/docker/Dockerfile.cuda -t trading-bot:latest-gpu .
fi

# Build dev image
docker build -f deployment/docker/Dockerfile.dev -t trading-bot:dev .

echo "Build complete!"
"""
    
    (docker_dir / "build.sh").write_text(build_script)
    (docker_dir / "build.sh").chmod(0o755)


# Exports
__all__ = [
    "DOCKER_CONFIG",
    "generate_dockerfile",
    "generate_dockerignore",
    "generate_docker_compose_dev",
    "create_docker_files"
]