"""
Deployment Package
==================

Configuration et scripts pour le déploiement du robot de trading.
Support pour Docker, Kubernetes et Terraform.

Structure:
    - docker/: Dockerfiles et configuration Docker
    - k8s/: Manifests Kubernetes
    - terraform/: Infrastructure as Code

Usage:
    # Build Docker image
    docker build -f deployment/docker/Dockerfile -t trading-bot:latest .
    
    # Deploy to Kubernetes
    kubectl apply -f deployment/k8s/
    
    # Provision infrastructure
    terraform apply deployment/terraform/
"""

import os
import subprocess
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

# Configuration de déploiement
DEPLOYMENT_CONFIG = {
    "docker": {
        "registry": "your-registry.io",
        "repository": "trading-bot",
        "tags": ["latest", "stable"],
        "build_args": {
            "PYTHON_VERSION": "3.11",
            "CUDA_VERSION": "11.8"
        }
    },
    "kubernetes": {
        "namespace": "trading",
        "replicas": {
            "dev": 1,
            "staging": 2,
            "production": 3
        },
        "resources": {
            "dev": {
                "requests": {"cpu": "1", "memory": "2Gi"},
                "limits": {"cpu": "2", "memory": "4Gi"}
            },
            "production": {
                "requests": {"cpu": "4", "memory": "8Gi"},
                "limits": {"cpu": "8", "memory": "16Gi"}
            }
        }
    },
    "terraform": {
        "backend": "s3",
        "backend_config": {
            "bucket": "trading-bot-tfstate",
            "key": "infrastructure/terraform.tfstate",
            "region": "us-east-1"
        }
    },
    "environments": {
        "dev": {
            "domain": "dev.trading-bot.io",
            "ssl": False,
            "monitoring": True
        },
        "staging": {
            "domain": "staging.trading-bot.io",
            "ssl": True,
            "monitoring": True
        },
        "production": {
            "domain": "trading-bot.io",
            "ssl": True,
            "monitoring": True,
            "high_availability": True
        }
    }
}


class Environment(Enum):
    """Environnements de déploiement"""
    DEVELOPMENT = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


class DeploymentTarget(Enum):
    """Cibles de déploiement"""
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    AWS_ECS = "aws_ecs"
    AWS_LAMBDA = "aws_lambda"
    BARE_METAL = "bare_metal"


@dataclass
class DeploymentConfig:
    """Configuration de déploiement"""
    environment: Environment
    target: DeploymentTarget
    version: str
    config_overrides: Dict[str, Any] = None
    dry_run: bool = False
    force: bool = False


class DeploymentManager:
    """Gestionnaire de déploiement"""
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.project_root = Path(__file__).parent.parent
        
    def deploy(self) -> bool:
        """Déploie selon la configuration"""
        if self.config.target == DeploymentTarget.DOCKER:
            return self.deploy_docker()
        elif self.config.target == DeploymentTarget.KUBERNETES:
            return self.deploy_kubernetes()
        elif self.config.target == DeploymentTarget.AWS_ECS:
            return self.deploy_aws_ecs()
        else:
            raise NotImplementedError(f"Target {self.config.target} not implemented")
    
    def deploy_docker(self) -> bool:
        """Déploie avec Docker"""
        # Build image
        dockerfile = self.project_root / "deployment/docker/Dockerfile"
        tag = f"{DEPLOYMENT_CONFIG['docker']['repository']}:{self.config.version}"
        
        build_cmd = [
            "docker", "build",
            "-f", str(dockerfile),
            "-t", tag,
            "."
        ]
        
        # Add build args
        for arg, value in DEPLOYMENT_CONFIG['docker']['build_args'].items():
            build_cmd.extend(["--build-arg", f"{arg}={value}"])
        
        if self.config.dry_run:
            print(f"Would run: {' '.join(build_cmd)}")
            return True
            
        result = subprocess.run(build_cmd, cwd=self.project_root)
        return result.returncode == 0
    
    def deploy_kubernetes(self) -> bool:
        """Déploie sur Kubernetes"""
        k8s_dir = self.project_root / "deployment/k8s"
        namespace = DEPLOYMENT_CONFIG['kubernetes']['namespace']
        
        # Create namespace if needed
        create_ns_cmd = ["kubectl", "create", "namespace", namespace, "--dry-run=client", "-o", "yaml"]
        if not self.config.dry_run:
            create_ns_cmd.append("| kubectl apply -f -")
            
        # Apply manifests
        apply_cmd = [
            "kubectl", "apply",
            "-n", namespace,
            "-f", str(k8s_dir),
            "--recursive"
        ]
        
        if self.config.dry_run:
            apply_cmd.append("--dry-run=client")
            
        result = subprocess.run(apply_cmd)
        return result.returncode == 0
    
    def deploy_aws_ecs(self) -> bool:
        """Déploie sur AWS ECS"""
        # TODO: Implémenter
        raise NotImplementedError("AWS ECS deployment not yet implemented")
    
    def rollback(self, version: str) -> bool:
        """Rollback vers une version précédente"""
        if self.config.target == DeploymentTarget.KUBERNETES:
            rollback_cmd = [
                "kubectl", "rollout", "undo",
                "deployment/trading-bot",
                "-n", DEPLOYMENT_CONFIG['kubernetes']['namespace']
            ]
            
            if version:
                rollback_cmd.extend(["--to-revision", version])
                
            result = subprocess.run(rollback_cmd)
            return result.returncode == 0
            
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Obtient le statut du déploiement"""
        if self.config.target == DeploymentTarget.KUBERNETES:
            status_cmd = [
                "kubectl", "get", "all",
                "-n", DEPLOYMENT_CONFIG['kubernetes']['namespace'],
                "-o", "json"
            ]
            
            result = subprocess.run(status_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return json.loads(result.stdout)
                
        return {}


# Helpers pour génération de configuration
def generate_docker_compose(environment: Environment) -> str:
    """Génère un docker-compose.yml"""
    config = {
        "version": "3.8",
        "services": {
            "trading-bot": {
                "image": f"{DEPLOYMENT_CONFIG['docker']['repository']}:latest",
                "environment": {
                    "TRADING_ENV": environment.value,
                    "LOG_LEVEL": "INFO" if environment == Environment.PRODUCTION else "DEBUG"
                },
                "volumes": [
                    "./config:/app/config:ro",
                    "./logs:/app/logs"
                ],
                "restart": "unless-stopped",
                "networks": ["trading-network"]
            },
            "redis": {
                "image": "redis:7-alpine",
                "command": "redis-server --appendonly yes",
                "volumes": ["redis-data:/data"],
                "networks": ["trading-network"]
            },
            "postgres": {
                "image": "timescale/timescaledb:latest-pg15",
                "environment": {
                    "POSTGRES_DB": "trading",
                    "POSTGRES_USER": "trader",
                    "POSTGRES_PASSWORD": "${DB_PASSWORD}"
                },
                "volumes": ["postgres-data:/var/lib/postgresql/data"],
                "networks": ["trading-network"]
            }
        },
        "networks": {
            "trading-network": {
                "driver": "bridge"
            }
        },
        "volumes": {
            "redis-data": {},
            "postgres-data": {}
        }
    }
    
    if environment == Environment.PRODUCTION:
        # Add monitoring
        config["services"]["prometheus"] = {
            "image": "prom/prometheus:latest",
            "volumes": [
                "./prometheus.yml:/etc/prometheus/prometheus.yml:ro",
                "prometheus-data:/prometheus"
            ],
            "networks": ["trading-network"]
        }
        
        config["services"]["grafana"] = {
            "image": "grafana/grafana:latest",
            "environment": {
                "GF_SECURITY_ADMIN_PASSWORD": "${GRAFANA_PASSWORD}"
            },
            "volumes": ["grafana-data:/var/lib/grafana"],
            "networks": ["trading-network"],
            "ports": ["3000:3000"]
        }
        
        config["volumes"]["prometheus-data"] = {}
        config["volumes"]["grafana-data"] = {}
    
    return yaml.dump(config, default_flow_style=False)


def generate_kubernetes_manifest(
    name: str,
    image: str,
    environment: Environment
) -> Dict[str, Any]:
    """Génère un manifest Kubernetes"""
    env_config = DEPLOYMENT_CONFIG['environments'][environment.value]
    k8s_config = DEPLOYMENT_CONFIG['kubernetes']
    
    manifest = {
        "apiVersion": "apps/v1",
        "kind": "Deployment",
        "metadata": {
            "name": name,
            "namespace": k8s_config['namespace'],
            "labels": {
                "app": name,
                "environment": environment.value
            }
        },
        "spec": {
            "replicas": k8s_config['replicas'][environment.value],
            "selector": {
                "matchLabels": {
                    "app": name
                }
            },
            "template": {
                "metadata": {
                    "labels": {
                        "app": name,
                        "environment": environment.value
                    }
                },
                "spec": {
                    "containers": [{
                        "name": name,
                        "image": image,
                        "resources": k8s_config['resources'].get(
                            environment.value,
                            k8s_config['resources']['dev']
                        ),
                        "env": [
                            {"name": "TRADING_ENV", "value": environment.value}
                        ],
                        "livenessProbe": {
                            "httpGet": {
                                "path": "/health",
                                "port": 8080
                            },
                            "initialDelaySeconds": 30,
                            "periodSeconds": 10
                        },
                        "readinessProbe": {
                            "httpGet": {
                                "path": "/ready",
                                "port": 8080
                            },
                            "initialDelaySeconds": 5,
                            "periodSeconds": 5
                        }
                    }]
                }
            }
        }
    }
    
    if env_config.get('high_availability'):
        manifest["spec"]["template"]["spec"]["affinity"] = {
            "podAntiAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": [{
                    "labelSelector": {
                        "matchExpressions": [{
                            "key": "app",
                            "operator": "In",
                            "values": [name]
                        }]
                    },
                    "topologyKey": "kubernetes.io/hostname"
                }]
            }
        }
    
    return manifest


# Scripts de déploiement
def create_deployment_scripts():
    """Crée les scripts de déploiement"""
    scripts_dir = Path(__file__).parent / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    
    # Script de déploiement principal
    deploy_script = '''#!/bin/bash
set -e

ENVIRONMENT=${1:-dev}
VERSION=${2:-latest}

echo "Deploying Trading Bot - Environment: $ENVIRONMENT, Version: $VERSION"

# Build and push Docker image
docker build -t trading-bot:$VERSION .
docker tag trading-bot:$VERSION $REGISTRY/trading-bot:$VERSION
docker push $REGISTRY/trading-bot:$VERSION

# Deploy to Kubernetes
kubectl set image deployment/trading-bot trading-bot=$REGISTRY/trading-bot:$VERSION -n trading

# Wait for rollout
kubectl rollout status deployment/trading-bot -n trading

echo "Deployment complete!"
'''
    
    (scripts_dir / "deploy.sh").write_text(deploy_script)
    (scripts_dir / "deploy.sh").chmod(0o755)
    
    # Script de rollback
    rollback_script = '''#!/bin/bash
set -e

REVISION=${1:-}

echo "Rolling back Trading Bot"

if [ -z "$REVISION" ]; then
    kubectl rollout undo deployment/trading-bot -n trading
else
    kubectl rollout undo deployment/trading-bot --to-revision=$REVISION -n trading
fi

kubectl rollout status deployment/trading-bot -n trading

echo "Rollback complete!"
'''
    
    (scripts_dir / "rollback.sh").write_text(rollback_script)
    (scripts_dir / "rollback.sh").chmod(0o755)


# Exports
__all__ = [
    "DEPLOYMENT_CONFIG",
    "Environment",
    "DeploymentTarget",
    "DeploymentConfig",
    "DeploymentManager",
    "generate_docker_compose",
    "generate_kubernetes_manifest",
    "create_deployment_scripts"
]

# Initialisation
if __name__ == "__main__":
    # Créer les scripts si exécuté directement
    create_deployment_scripts()
    print("Deployment scripts created successfully!")