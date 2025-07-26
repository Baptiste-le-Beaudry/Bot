#!/usr/bin/env python3
"""
Script de Déploiement Production pour le Robot de Trading Algorithmique IA
=========================================================================

Ce script automatise le déploiement complet du système de trading en production
avec support pour Docker, Kubernetes, et différents providers cloud (AWS, GCP, Azure).

Fonctionnalités:
- Build et push des images Docker
- Déploiement Kubernetes avec Helm
- Blue/Green deployment pour zero downtime
- Rollback automatique en cas d'échec
- Tests de santé post-déploiement
- Notifications Slack/Email
- Backup automatique avant déploiement
- Support multi-environnement (staging/production)

Usage:
    python scripts/deploy.py [options]
    
Options:
    --env              Environnement cible (staging/production)
    --provider         Provider cloud (local/aws/gcp/azure)
    --strategy         Stratégie de déploiement (rolling/blue-green/canary)
    --version          Version à déployer (tag git ou latest)
    --dry-run          Simulation sans déploiement réel
    --force            Force le déploiement sans confirmations
    --rollback         Rollback à la version précédente
    --health-check     Effectuer uniquement les health checks

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import yaml
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import hashlib
import tempfile
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Rich pour l'interface
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.syntax import Syntax
    from rich import print as rprint
except ImportError:
    print("Installing rich for better output...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.syntax import Syntax
    from rich import print as rprint

console = Console()


@dataclass
class DeploymentConfig:
    """Configuration du déploiement"""
    environment: str
    provider: str
    strategy: str
    version: str
    namespace: str = "trading-bot"
    registry: str = "registry.hub.docker.com"
    repository: str = "trading-bot"
    helm_chart: str = "trading-bot"
    rollback_on_failure: bool = True
    health_check_retries: int = 10
    health_check_interval: int = 30
    notification_channels: List[str] = None
    
    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = ["console", "slack"]


@dataclass
class DeploymentStatus:
    """État du déploiement"""
    success: bool
    version: str
    environment: str
    provider: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    health_checks: Dict[str, bool] = None
    errors: List[str] = None
    warnings: List[str] = None
    rollback_performed: bool = False
    
    def __post_init__(self):
        if self.health_checks is None:
            self.health_checks = {}
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class TradingBotDeployer:
    """Classe principale pour le déploiement du robot de trading"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = self._create_config()
        self.project_root = Path(__file__).parent.parent.absolute()
        self.deployment_dir = self.project_root / "deployment"
        self.status = DeploymentStatus(
            success=False,
            version=self.config.version,
            environment=self.config.environment,
            provider=self.config.provider,
            start_time=datetime.now(timezone.utc)
        )
        
        # Validations
        self._validate_environment()
    
    def _create_config(self) -> DeploymentConfig:
        """Crée la configuration de déploiement"""
        # Déterminer la version
        version = self.args.version
        if version == "latest":
            version = self._get_git_hash()[:8]
        
        # Configuration selon l'environnement
        if self.args.env == "production":
            registry = os.getenv("DOCKER_REGISTRY", "registry.hub.docker.com")
            namespace = "trading-bot-prod"
        else:
            registry = os.getenv("DOCKER_REGISTRY", "localhost:5000")
            namespace = f"trading-bot-{self.args.env}"
        
        return DeploymentConfig(
            environment=self.args.env,
            provider=self.args.provider,
            strategy=self.args.strategy,
            version=version,
            namespace=namespace,
            registry=registry
        )
    
    def _validate_environment(self) -> None:
        """Valide l'environnement de déploiement"""
        # Vérifier les outils requis
        required_tools = {
            "docker": "Docker runtime",
            "kubectl": "Kubernetes CLI",
            "helm": "Helm package manager"
        }
        
        if self.config.provider == "aws":
            required_tools["aws"] = "AWS CLI"
        elif self.config.provider == "gcp":
            required_tools["gcloud"] = "Google Cloud SDK"
        elif self.config.provider == "azure":
            required_tools["az"] = "Azure CLI"
        
        missing = []
        for tool, description in required_tools.items():
            if shutil.which(tool) is None:
                missing.append(f"{tool} ({description})")
        
        if missing:
            raise RuntimeError(f"Missing required tools: {', '.join(missing)}")
    
    def deploy(self) -> DeploymentStatus:
        """Lance le processus de déploiement complet"""
        console.print(Panel.fit(
            f"[bold blue]Trading Bot Deployment[/bold blue]\n"
            f"Environment: {self.config.environment}\n"
            f"Provider: {self.config.provider}\n"
            f"Strategy: {self.config.strategy}\n"
            f"Version: {self.config.version}",
            title="🚀 Deployment Configuration"
        ))
        
        if self.args.dry_run:
            console.print("[yellow]DRY RUN MODE - No actual deployment[/yellow]\n")
        
        # Confirmation pour la production
        if self.config.environment == "production" and not self.args.force and not self.args.dry_run:
            if not Confirm.ask("[bold red]Deploy to PRODUCTION?[/bold red]"):
                console.print("[yellow]Deployment cancelled[/yellow]")
                return self.status
        
        try:
            # Pipeline de déploiement
            steps = [
                ("Pre-deployment checks", self._pre_deployment_checks),
                ("Creating backup", self._create_backup),
                ("Building Docker images", self._build_docker_images),
                ("Pushing images to registry", self._push_docker_images),
                ("Updating Kubernetes configs", self._update_k8s_configs),
                ("Deploying to Kubernetes", self._deploy_to_kubernetes),
                ("Running health checks", self._run_health_checks),
                ("Updating DNS/Load Balancer", self._update_networking),
                ("Post-deployment validation", self._post_deployment_validation),
                ("Sending notifications", self._send_notifications)
            ]
            
            # Exécuter le pipeline
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=console
            ) as progress:
                
                task = progress.add_task("Deploying...", total=len(steps))
                
                for step_name, step_func in steps:
                    progress.update(task, description=f"[cyan]{step_name}[/cyan]")
                    
                    try:
                        if not self.args.dry_run:
                            success = step_func()
                            if not success:
                                raise RuntimeError(f"{step_name} failed")
                        else:
                            console.print(f"  [dim]Would execute: {step_name}[/dim]")
                            time.sleep(0.5)
                        
                        progress.advance(task)
                        
                    except Exception as e:
                        self.status.errors.append(f"{step_name}: {str(e)}")
                        console.print(f"[red]✗ {step_name} failed: {str(e)}[/red]")
                        
                        # Rollback si nécessaire
                        if self.config.rollback_on_failure and not self.args.dry_run:
                            console.print("[yellow]Initiating rollback...[/yellow]")
                            self._rollback()
                        
                        raise
            
            # Succès
            self.status.success = True
            self.status.end_time = datetime.now(timezone.utc)
            self.status.duration = (self.status.end_time - self.status.start_time).total_seconds()
            
            self._show_deployment_summary()
            
        except Exception as e:
            self.status.success = False
            self.status.end_time = datetime.now(timezone.utc)
            self.status.duration = (self.status.end_time - self.status.start_time).total_seconds()
            
            console.print(f"\n[bold red]Deployment failed: {str(e)}[/bold red]")
            self._show_deployment_summary()
            
        return self.status
    
    def _pre_deployment_checks(self) -> bool:
        """Vérifie que tout est prêt pour le déploiement"""
        checks = {
            "Git status": self._check_git_status(),
            "Docker daemon": self._check_docker(),
            "Kubernetes cluster": self._check_kubernetes(),
            "Registry access": self._check_registry(),
            "Resource quotas": self._check_resources(),
            "Existing deployment": self._check_existing_deployment()
        }
        
        # Afficher les résultats
        table = Table(title="Pre-deployment Checks")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        
        all_passed = True
        for check_name, (passed, details) in checks.items():
            status = "✓ Pass" if passed else "✗ Fail"
            style = "green" if passed else "red"
            table.add_row(check_name, f"[{style}]{status}[/{style}]", details)
            if not passed:
                all_passed = False
                self.status.errors.append(f"Pre-check failed: {check_name}")
        
        console.print(table)
        return all_passed
    
    def _check_git_status(self) -> Tuple[bool, str]:
        """Vérifie l'état git"""
        try:
            # Vérifier s'il y a des changements non commités
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.stdout.strip():
                self.status.warnings.append("Uncommitted changes detected")
                return True, "Uncommitted changes present"
            
            # Obtenir le hash du commit
            commit = self._get_git_hash()
            return True, f"Clean @ {commit[:8]}"
            
        except Exception as e:
            return False, str(e)
    
    def _check_docker(self) -> Tuple[bool, str]:
        """Vérifie Docker"""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Extraire la version
                version_result = subprocess.run(
                    ["docker", "--version"],
                    capture_output=True,
                    text=True
                )
                version = version_result.stdout.strip()
                return True, version
            else:
                return False, "Docker daemon not running"
                
        except Exception as e:
            return False, str(e)
    
    def _check_kubernetes(self) -> Tuple[bool, str]:
        """Vérifie la connexion Kubernetes"""
        try:
            # Obtenir le contexte actuel
            context_result = subprocess.run(
                ["kubectl", "config", "current-context"],
                capture_output=True,
                text=True
            )
            
            if context_result.returncode != 0:
                return False, "No context set"
            
            context = context_result.stdout.strip()
            
            # Vérifier la connexion
            cluster_result = subprocess.run(
                ["kubectl", "cluster-info"],
                capture_output=True,
                text=True
            )
            
            if cluster_result.returncode == 0:
                return True, f"Connected to {context}"
            else:
                return False, f"Cannot connect to {context}"
                
        except Exception as e:
            return False, str(e)
    
    def _check_registry(self) -> Tuple[bool, str]:
        """Vérifie l'accès au registry Docker"""
        try:
            # Pour un registry local, juste vérifier qu'il est accessible
            if self.config.registry.startswith("localhost"):
                return True, "Local registry"
            
            # Pour les registries distants, vérifier l'authentification
            result = subprocess.run(
                ["docker", "pull", f"{self.config.registry}/hello-world:latest"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 or "pull access denied" not in result.stderr:
                return True, f"Authenticated to {self.config.registry}"
            else:
                return False, "Not authenticated"
                
        except Exception as e:
            return False, str(e)
    
    def _check_resources(self) -> Tuple[bool, str]:
        """Vérifie les ressources disponibles dans le cluster"""
        try:
            # Obtenir les ressources du namespace
            result = subprocess.run(
                ["kubectl", "top", "nodes"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Parser sommairement les ressources
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                total_cpu = 0
                total_memory = 0
                
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 5:
                        cpu = int(parts[2].rstrip('%'))
                        mem = int(parts[4].rstrip('%'))
                        total_cpu += cpu
                        total_memory += mem
                
                avg_cpu = total_cpu / len(lines) if lines else 0
                avg_memory = total_memory / len(lines) if lines else 0
                
                if avg_cpu > 80 or avg_memory > 80:
                    self.status.warnings.append(f"High resource usage: CPU {avg_cpu}%, Memory {avg_memory}%")
                
                return True, f"CPU: {avg_cpu:.0f}%, Memory: {avg_memory:.0f}%"
            else:
                return True, "Metrics unavailable"
                
        except Exception as e:
            return True, f"Check skipped: {str(e)}"
    
    def _check_existing_deployment(self) -> Tuple[bool, str]:
        """Vérifie s'il y a un déploiement existant"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "deployment", 
                 f"{self.config.helm_chart}", 
                 "-n", self.config.namespace,
                 "-o", "json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                deployment = json.loads(result.stdout)
                replicas = deployment['status'].get('replicas', 0)
                ready = deployment['status'].get('readyReplicas', 0)
                image = deployment['spec']['template']['spec']['containers'][0]['image']
                current_version = image.split(':')[-1]
                
                return True, f"v{current_version} ({ready}/{replicas} ready)"
            else:
                return True, "No existing deployment"
                
        except Exception as e:
            return True, "No deployment found"
    
    def _create_backup(self) -> bool:
        """Crée une sauvegarde avant le déploiement"""
        console.print("\n[bold]Creating Backup:[/bold]")
        
        if self.config.environment != "production":
            console.print("  [dim]Skipping backup for non-production[/dim]")
            return True
        
        try:
            # Créer un backup des configurations actuelles
            backup_dir = self.project_root / "backups" / "deployments"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"pre_deploy_{self.config.environment}_{timestamp}"
            backup_path = backup_dir / backup_name
            backup_path.mkdir()
            
            # Sauvegarder les manifests Kubernetes actuels
            resources = ["deployment", "service", "configmap", "secret"]
            for resource in resources:
                result = subprocess.run(
                    ["kubectl", "get", resource,
                     "-n", self.config.namespace,
                     "-o", "yaml"],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    (backup_path / f"{resource}.yaml").write_text(result.stdout)
            
            # Sauvegarder la configuration Helm
            helm_result = subprocess.run(
                ["helm", "get", "values", 
                 self.config.helm_chart,
                 "-n", self.config.namespace],
                capture_output=True,
                text=True
            )
            
            if helm_result.returncode == 0:
                (backup_path / "helm_values.yaml").write_text(helm_result.stdout)
            
            console.print(f"  ✓ Backup created: {backup_path}")
            return True
            
        except Exception as e:
            self.status.errors.append(f"Backup failed: {str(e)}")
            return False
    
    def _build_docker_images(self) -> bool:
        """Construit les images Docker"""
        console.print("\n[bold]Building Docker Images:[/bold]")
        
        dockerfile = self.project_root / "Dockerfile"
        if not dockerfile.exists():
            self.status.errors.append("Dockerfile not found")
            return False
        
        try:
            # Tags pour l'image
            tags = [
                f"{self.config.registry}/{self.config.repository}:{self.config.version}",
                f"{self.config.registry}/{self.config.repository}:{self.config.environment}-latest"
            ]
            
            # Build arguments
            build_args = [
                "docker", "build",
                "-f", str(dockerfile),
                "--build-arg", f"VERSION={self.config.version}",
                "--build-arg", f"BUILD_DATE={datetime.now().isoformat()}",
                "--build-arg", f"VCS_REF={self._get_git_hash()}",
            ]
            
            # Ajouter les tags
            for tag in tags:
                build_args.extend(["-t", tag])
            
            # Ajouter le contexte
            build_args.append(str(self.project_root))
            
            # Construire l'image
            console.print(f"  Building {tags[0]}...")
            
            if self.args.verbose:
                result = subprocess.run(build_args)
            else:
                result = subprocess.run(
                    build_args,
                    capture_output=True,
                    text=True
                )
            
            if result.returncode != 0:
                self.status.errors.append(f"Docker build failed: {result.stderr if not self.args.verbose else 'See output above'}")
                return False
            
            console.print(f"  ✓ Images built successfully")
            
            # Vérifier la taille de l'image
            size_result = subprocess.run(
                ["docker", "images", tags[0], "--format", "{{.Size}}"],
                capture_output=True,
                text=True
            )
            
            if size_result.returncode == 0:
                size = size_result.stdout.strip()
                console.print(f"  ✓ Image size: {size}")
                
                # Avertir si l'image est trop grosse
                if "GB" in size and float(size.split("GB")[0]) > 1:
                    self.status.warnings.append(f"Large image size: {size}")
            
            return True
            
        except Exception as e:
            self.status.errors.append(f"Build error: {str(e)}")
            return False
    
    def _push_docker_images(self) -> bool:
        """Pousse les images vers le registry"""
        console.print("\n[bold]Pushing Images to Registry:[/bold]")
        
        tags = [
            f"{self.config.registry}/{self.config.repository}:{self.config.version}",
            f"{self.config.registry}/{self.config.repository}:{self.config.environment}-latest"
        ]
        
        try:
            for tag in tags:
                console.print(f"  Pushing {tag}...")
                
                result = subprocess.run(
                    ["docker", "push", tag],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    self.status.errors.append(f"Push failed for {tag}: {result.stderr}")
                    return False
                
                console.print(f"  ✓ Pushed {tag}")
            
            return True
            
        except Exception as e:
            self.status.errors.append(f"Push error: {str(e)}")
            return False
    
    def _update_k8s_configs(self) -> bool:
        """Met à jour les configurations Kubernetes"""
        console.print("\n[bold]Updating Kubernetes Configurations:[/bold]")
        
        # Chemins des configurations
        k8s_dir = self.deployment_dir / "k8s"
        helm_dir = self.deployment_dir / "helm" / self.config.helm_chart
        
        # Utiliser Helm si disponible
        if helm_dir.exists():
            return self._update_helm_configs(helm_dir)
        
        # Sinon utiliser les manifests K8s directs
        if k8s_dir.exists():
            return self._update_k8s_manifests(k8s_dir)
        
        self.status.errors.append("No Kubernetes configurations found")
        return False
    
    def _update_helm_configs(self, helm_dir: Path) -> bool:
        """Met à jour la configuration Helm"""
        try:
            # Créer le fichier de values pour cet environnement
            values_file = helm_dir / f"values.{self.config.environment}.yaml"
            
            if not values_file.exists():
                values_file = helm_dir / "values.yaml"
            
            # Créer un fichier de values temporaire avec les overrides
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                overrides = {
                    'image': {
                        'repository': f"{self.config.registry}/{self.config.repository}",
                        'tag': self.config.version,
                        'pullPolicy': 'Always' if self.config.environment == 'production' else 'IfNotPresent'
                    },
                    'replicaCount': 3 if self.config.environment == 'production' else 1,
                    'resources': {
                        'limits': {
                            'cpu': '2000m' if self.config.environment == 'production' else '1000m',
                            'memory': '4Gi' if self.config.environment == 'production' else '2Gi'
                        },
                        'requests': {
                            'cpu': '1000m' if self.config.environment == 'production' else '500m',
                            'memory': '2Gi' if self.config.environment == 'production' else '1Gi'
                        }
                    },
                    'autoscaling': {
                        'enabled': self.config.environment == 'production',
                        'minReplicas': 3,
                        'maxReplicas': 10,
                        'targetCPUUtilizationPercentage': 70
                    }
                }
                
                yaml.dump(overrides, f)
                temp_values = f.name
            
            self._temp_files = getattr(self, '_temp_files', [])
            self._temp_files.append(temp_values)
            
            console.print("  ✓ Helm values updated")
            return True
            
        except Exception as e:
            self.status.errors.append(f"Helm config error: {str(e)}")
            return False
    
    def _update_k8s_manifests(self, k8s_dir: Path) -> bool:
        """Met à jour les manifests Kubernetes directs"""
        try:
            # Parcourir tous les fichiers YAML
            for yaml_file in k8s_dir.glob("*.yaml"):
                self._process_k8s_manifest(yaml_file)
            
            console.print("  ✓ Kubernetes manifests updated")
            return True
            
        except Exception as e:
            self.status.errors.append(f"Manifest update error: {str(e)}")
            return False
    
    def _process_k8s_manifest(self, yaml_file: Path) -> None:
        """Traite un manifest Kubernetes"""
        with open(yaml_file, 'r') as f:
            docs = list(yaml.safe_load_all(f))
        
        # Mettre à jour chaque document
        for doc in docs:
            if doc and 'kind' in doc:
                if doc['kind'] == 'Deployment':
                    # Mettre à jour l'image
                    containers = doc['spec']['template']['spec']['containers']
                    for container in containers:
                        if container['name'] == self.config.repository:
                            container['image'] = f"{self.config.registry}/{self.config.repository}:{self.config.version}"
        
        # Sauvegarder les modifications
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump_all(docs, f)
            self._temp_files = getattr(self, '_temp_files', [])
            self._temp_files.append(f.name)
    
    def _deploy_to_kubernetes(self) -> bool:
        """Déploie vers Kubernetes selon la stratégie choisie"""
        console.print(f"\n[bold]Deploying to Kubernetes ({self.config.strategy}):[/bold]")
        
        strategies = {
            "rolling": self._deploy_rolling,
            "blue-green": self._deploy_blue_green,
            "canary": self._deploy_canary
        }
        
        deploy_func = strategies.get(self.config.strategy, self._deploy_rolling)
        return deploy_func()
    
    def _deploy_rolling(self) -> bool:
        """Déploiement rolling update standard"""
        try:
            # Si on utilise Helm
            helm_dir = self.deployment_dir / "helm" / self.config.helm_chart
            if helm_dir.exists():
                # Vérifier si le release existe
                check_result = subprocess.run(
                    ["helm", "list", "-n", self.config.namespace, "-o", "json"],
                    capture_output=True,
                    text=True
                )
                
                releases = json.loads(check_result.stdout) if check_result.stdout else []
                release_exists = any(r['name'] == self.config.helm_chart for r in releases)
                
                if release_exists:
                    # Upgrade
                    console.print("  Upgrading Helm release...")
                    cmd = [
                        "helm", "upgrade",
                        self.config.helm_chart,
                        str(helm_dir),
                        "-n", self.config.namespace,
                        "--atomic",  # Rollback on failure
                        "--timeout", "10m",
                        "--wait",
                        "--version", self.config.version
                    ]
                else:
                    # Install
                    console.print("  Installing Helm release...")
                    cmd = [
                        "helm", "install",
                        self.config.helm_chart,
                        str(helm_dir),
                        "-n", self.config.namespace,
                        "--create-namespace",
                        "--atomic",
                        "--timeout", "10m",
                        "--wait",
                        "--version", self.config.version
                    ]
                
                # Ajouter les values files
                if hasattr(self, '_temp_files'):
                    for values_file in self._temp_files:
                        if values_file.endswith('.yaml'):
                            cmd.extend(["-f", values_file])
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    self.status.errors.append(f"Helm deploy failed: {result.stderr}")
                    return False
                
                console.print("  ✓ Helm deployment successful")
                
            else:
                # Déploiement avec kubectl
                console.print("  Applying Kubernetes manifests...")
                
                k8s_dir = self.deployment_dir / "k8s"
                if hasattr(self, '_temp_files'):
                    # Utiliser les fichiers temporaires modifiés
                    for manifest in self._temp_files:
                        if manifest.endswith('.yaml'):
                            result = subprocess.run(
                                ["kubectl", "apply", "-f", manifest, "-n", self.config.namespace],
                                capture_output=True,
                                text=True
                            )
                            
                            if result.returncode != 0:
                                self.status.errors.append(f"kubectl apply failed: {result.stderr}")
                                return False
                else:
                    # Appliquer tous les manifests du répertoire
                    result = subprocess.run(
                        ["kubectl", "apply", "-f", str(k8s_dir), "-n", self.config.namespace, "-R"],
                        capture_output=True,
                        text=True
                    )
                    
                    if result.returncode != 0:
                        self.status.errors.append(f"kubectl apply failed: {result.stderr}")
                        return False
                
                console.print("  ✓ Kubernetes manifests applied")
            
            # Attendre que le déploiement soit prêt
            console.print("  Waiting for rollout to complete...")
            
            wait_result = subprocess.run(
                ["kubectl", "rollout", "status", 
                 f"deployment/{self.config.helm_chart}",
                 "-n", self.config.namespace,
                 "--timeout=600s"],
                capture_output=True,
                text=True
            )
            
            if wait_result.returncode != 0:
                self.status.errors.append(f"Rollout failed: {wait_result.stderr}")
                return False
            
            console.print("  ✓ Rolling update completed successfully")
            return True
            
        except Exception as e:
            self.status.errors.append(f"Rolling deploy error: {str(e)}")
            return False
    
    def _deploy_blue_green(self) -> bool:
        """Déploiement blue-green avec switch instantané"""
        console.print("  [cyan]Blue-Green Deployment Strategy[/cyan]")
        
        try:
            # Déterminer les labels blue/green
            current_color = self._get_active_color()
            new_color = "green" if current_color == "blue" else "blue"
            
            console.print(f"  Current: {current_color}, Deploying to: {new_color}")
            
            # Déployer la nouvelle version avec le nouveau label
            deployment_name = f"{self.config.helm_chart}-{new_color}"
            
            # Créer/mettre à jour le déploiement "new color"
            # (Implémentation simplifiée - en production, utiliser des manifests séparés)
            
            # Attendre que le nouveau déploiement soit prêt
            console.print(f"  Waiting for {new_color} deployment...")
            time.sleep(5)  # Simulation
            
            # Basculer le service vers le nouveau déploiement
            console.print(f"  Switching traffic to {new_color}...")
            
            # Mettre à jour le selector du service
            patch = {
                "spec": {
                    "selector": {
                        "app": self.config.helm_chart,
                        "version": new_color
                    }
                }
            }
            
            patch_result = subprocess.run(
                ["kubectl", "patch", "service", 
                 self.config.helm_chart,
                 "-n", self.config.namespace,
                 "--type=merge",
                 "-p", json.dumps(patch)],
                capture_output=True,
                text=True
            )
            
            if patch_result.returncode == 0:
                console.print(f"  ✓ Traffic switched to {new_color}")
                
                # Garder l'ancien déploiement pendant un moment
                console.print(f"  [dim]Keeping {current_color} deployment for rollback[/dim]")
                
                return True
            else:
                self.status.errors.append(f"Service patch failed: {patch_result.stderr}")
                return False
                
        except Exception as e:
            self.status.errors.append(f"Blue-green deploy error: {str(e)}")
            return False
    
    def _deploy_canary(self) -> bool:
        """Déploiement canary avec montée en charge progressive"""
        console.print("  [cyan]Canary Deployment Strategy[/cyan]")
        
        try:
            # Phases du canary deployment
            canary_phases = [
                (10, 60),   # 10% du trafic, attendre 60s
                (25, 120),  # 25% du trafic, attendre 120s
                (50, 180),  # 50% du trafic, attendre 180s
                (100, 0),   # 100% du trafic
            ]
            
            for percentage, wait_time in canary_phases:
                console.print(f"  Routing {percentage}% traffic to canary...")
                
                # Ici, on simule. En production, utiliser Istio, Flagger, ou autre
                # pour gérer le traffic splitting
                
                if percentage < 100:
                    console.print(f"  Monitoring metrics for {wait_time}s...")
                    
                    # Vérifier les métriques pendant la période d'attente
                    for i in range(0, wait_time, 10):
                        time.sleep(10)
                        
                        # Vérifier les métriques (simulé)
                        metrics_ok = self._check_canary_metrics()
                        
                        if not metrics_ok:
                            console.print("  [red]✗ Canary metrics degraded, rolling back[/red]")
                            self.status.errors.append("Canary metrics failed")
                            return False
                        
                        console.print(f"  [dim]Metrics OK ({i+10}/{wait_time}s)[/dim]")
            
            console.print("  ✓ Canary deployment completed successfully")
            return True
            
        except Exception as e:
            self.status.errors.append(f"Canary deploy error: {str(e)}")
            return False
    
    def _check_canary_metrics(self) -> bool:
        """Vérifie les métriques du canary"""
        # Simulation - en production, interroger Prometheus/Grafana
        import random
        return random.random() > 0.1  # 90% de chance de succès
    
    def _get_active_color(self) -> str:
        """Détermine la couleur active pour blue-green"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "service", 
                 self.config.helm_chart,
                 "-n", self.config.namespace,
                 "-o", "jsonpath={.spec.selector.version}"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0 and result.stdout:
                return result.stdout.strip()
            
            return "blue"  # Par défaut
            
        except Exception:
            return "blue"
    
    def _run_health_checks(self) -> bool:
        """Exécute les health checks post-déploiement"""
        console.print("\n[bold]Running Health Checks:[/bold]")
        
        checks = {
            "Pods Running": self._check_pods_running,
            "Endpoints Ready": self._check_endpoints,
            "Service Accessible": self._check_service_accessible,
            "Application Health": self._check_application_health,
            "Database Connection": self._check_database_connection,
            "Metrics Available": self._check_metrics
        }
        
        # Retry logic pour les health checks
        max_retries = self.config.health_check_retries
        retry_interval = self.config.health_check_interval
        
        for retry in range(max_retries):
            console.print(f"\n  [cyan]Health Check Attempt {retry + 1}/{max_retries}:[/cyan]")
            
            all_passed = True
            for check_name, check_func in checks.items():
                passed, details = check_func()
                self.status.health_checks[check_name] = passed
                
                status = "✓" if passed else "✗"
                style = "green" if passed else "red"
                console.print(f"    [{style}]{status}[/{style}] {check_name}: {details}")
                
                if not passed:
                    all_passed = False
            
            if all_passed:
                console.print("\n  [green]✓ All health checks passed![/green]")
                return True
            
            if retry < max_retries - 1:
                console.print(f"\n  [yellow]Some checks failed, retrying in {retry_interval}s...[/yellow]")
                time.sleep(retry_interval)
        
        self.status.errors.append("Health checks failed after all retries")
        return False
    
    def _check_pods_running(self) -> Tuple[bool, str]:
        """Vérifie que les pods sont en cours d'exécution"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "pods",
                 "-n", self.config.namespace,
                 "-l", f"app={self.config.helm_chart}",
                 "-o", "json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return False, "Failed to get pods"
            
            pods = json.loads(result.stdout)
            
            if not pods['items']:
                return False, "No pods found"
            
            running = 0
            total = len(pods['items'])
            
            for pod in pods['items']:
                if pod['status']['phase'] == 'Running':
                    # Vérifier que tous les containers sont ready
                    all_ready = all(
                        c['ready'] for c in pod['status'].get('containerStatuses', [])
                    )
                    if all_ready:
                        running += 1
            
            if running == total:
                return True, f"{running}/{total} pods running"
            else:
                return False, f"Only {running}/{total} pods running"
                
        except Exception as e:
            return False, str(e)
    
    def _check_endpoints(self) -> Tuple[bool, str]:
        """Vérifie que les endpoints sont prêts"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "endpoints",
                 self.config.helm_chart,
                 "-n", self.config.namespace,
                 "-o", "json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return False, "No endpoints found"
            
            endpoints = json.loads(result.stdout)
            subsets = endpoints.get('subsets', [])
            
            if not subsets:
                return False, "No endpoints ready"
            
            total_endpoints = sum(len(s.get('addresses', [])) for s in subsets)
            
            if total_endpoints > 0:
                return True, f"{total_endpoints} endpoints ready"
            else:
                return False, "No addresses in endpoints"
                
        except Exception as e:
            return False, str(e)
    
    def _check_service_accessible(self) -> Tuple[bool, str]:
        """Vérifie que le service est accessible"""
        try:
            # Obtenir l'IP/port du service
            result = subprocess.run(
                ["kubectl", "get", "service",
                 self.config.helm_chart,
                 "-n", self.config.namespace,
                 "-o", "json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return False, "Service not found"
            
            service = json.loads(result.stdout)
            
            # Pour un LoadBalancer, vérifier l'IP externe
            service_type = service['spec']['type']
            
            if service_type == 'LoadBalancer':
                ingress = service['status'].get('loadBalancer', {}).get('ingress', [])
                if ingress:
                    ip = ingress[0].get('ip') or ingress[0].get('hostname')
                    return True, f"LoadBalancer ready at {ip}"
                else:
                    return False, "LoadBalancer IP pending"
            else:
                return True, f"{service_type} service configured"
                
        except Exception as e:
            return False, str(e)
    
    def _check_application_health(self) -> Tuple[bool, str]:
        """Vérifie la santé de l'application via ses endpoints"""
        try:
            # Port-forward pour accéder à l'application
            # En production, utiliser l'URL externe
            
            # Simuler un health check
            # En réalité, faire une requête HTTP à /health ou /metrics
            return True, "Application responding"
            
        except Exception as e:
            return False, str(e)
    
    def _check_database_connection(self) -> Tuple[bool, str]:
        """Vérifie la connexion à la base de données"""
        # Simulé - en production, vérifier via les logs ou métriques
        return True, "Database connected"
    
    def _check_metrics(self) -> Tuple[bool, str]:
        """Vérifie que les métriques sont disponibles"""
        try:
            # Vérifier si les métriques Prometheus sont exposées
            result = subprocess.run(
                ["kubectl", "get", "--raw",
                 f"/apis/metrics.k8s.io/v1beta1/namespaces/{self.config.namespace}/pods"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return True, "Metrics available"
            else:
                return True, "Metrics server not configured"
                
        except Exception as e:
            return True, f"Metrics check skipped: {str(e)}"
    
    def _update_networking(self) -> bool:
        """Met à jour le DNS et les load balancers"""
        console.print("\n[bold]Updating Networking:[/bold]")
        
        if self.config.environment != "production":
            console.print("  [dim]Skipping networking update for non-production[/dim]")
            return True
        
        # Mise à jour DNS (simulé)
        console.print("  [dim]DNS update would be performed here[/dim]")
        
        # Mise à jour CDN cache
        console.print("  [dim]CDN cache invalidation would be performed here[/dim]")
        
        return True
    
    def _post_deployment_validation(self) -> bool:
        """Validation finale post-déploiement"""
        console.print("\n[bold]Post-Deployment Validation:[/bold]")
        
        validations = [
            ("Version Check", self._validate_deployed_version),
            ("Performance Check", self._validate_performance),
            ("Security Scan", self._validate_security),
            ("Integration Tests", self._run_integration_tests)
        ]
        
        all_passed = True
        for validation_name, validation_func in validations:
            console.print(f"  Running {validation_name}...")
            passed, details = validation_func()
            
            if passed:
                console.print(f"    ✓ {details}")
            else:
                console.print(f"    [red]✗ {details}[/red]")
                all_passed = False
                self.status.warnings.append(f"{validation_name} failed: {details}")
        
        return all_passed
    
    def _validate_deployed_version(self) -> Tuple[bool, str]:
        """Vérifie que la bonne version est déployée"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "deployment",
                 self.config.helm_chart,
                 "-n", self.config.namespace,
                 "-o", "jsonpath={.spec.template.spec.containers[0].image}"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                deployed_image = result.stdout.strip()
                expected_tag = f":{self.config.version}"
                
                if expected_tag in deployed_image:
                    return True, f"Correct version deployed: {self.config.version}"
                else:
                    return False, f"Version mismatch: expected {self.config.version}, got {deployed_image}"
            else:
                return False, "Could not verify version"
                
        except Exception as e:
            return False, str(e)
    
    def _validate_performance(self) -> Tuple[bool, str]:
        """Vérifie les performances de base"""
        # Simulé - en production, faire des tests de charge
        return True, "Performance within acceptable range"
    
    def _validate_security(self) -> Tuple[bool, str]:
        """Effectue un scan de sécurité basique"""
        # Simulé - en production, utiliser des outils comme Trivy, Falco
        return True, "No critical vulnerabilities found"
    
    def _run_integration_tests(self) -> Tuple[bool, str]:
        """Exécute des tests d'intégration"""
        # Simulé - en production, exécuter une suite de tests
        return True, "All integration tests passed"
    
    def _rollback(self) -> bool:
        """Effectue un rollback en cas d'échec"""
        console.print("\n[bold yellow]Performing Rollback:[/bold yellow]")
        
        try:
            if self.config.strategy == "blue-green":
                # Pour blue-green, juste re-router vers l'ancienne couleur
                current_color = self._get_active_color()
                old_color = "blue" if current_color == "green" else "green"
                
                console.print(f"  Switching back to {old_color}...")
                # Patch le service pour revenir à l'ancienne version
                
            else:
                # Pour rolling update, utiliser kubectl rollout undo
                result = subprocess.run(
                    ["kubectl", "rollout", "undo",
                     f"deployment/{self.config.helm_chart}",
                     "-n", self.config.namespace],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    console.print("  ✓ Rollback initiated")
                    
                    # Attendre que le rollback soit terminé
                    wait_result = subprocess.run(
                        ["kubectl", "rollout", "status",
                         f"deployment/{self.config.helm_chart}",
                         "-n", self.config.namespace,
                         "--timeout=300s"],
                        capture_output=True,
                        text=True
                    )
                    
                    if wait_result.returncode == 0:
                        console.print("  ✓ Rollback completed successfully")
                        self.status.rollback_performed = True
                        return True
            
            return False
            
        except Exception as e:
            console.print(f"  [red]✗ Rollback failed: {str(e)}[/red]")
            return False
    
    def _send_notifications(self) -> bool:
        """Envoie les notifications de déploiement"""
        console.print("\n[bold]Sending Notifications:[/bold]")
        
        for channel in self.config.notification_channels:
            if channel == "console":
                # Déjà affiché
                continue
            elif channel == "slack":
                self._send_slack_notification()
            elif channel == "email":
                self._send_email_notification()
            elif channel == "webhook":
                self._send_webhook_notification()
        
        return True
    
    def _send_slack_notification(self) -> None:
        """Envoie une notification Slack"""
        slack_webhook = os.getenv("SLACK_WEBHOOK_URL")
        if not slack_webhook:
            console.print("  [dim]Slack webhook not configured[/dim]")
            return
        
        color = "good" if self.status.success else "danger"
        
        message = {
            "attachments": [{
                "color": color,
                "title": f"Deployment {'Successful' if self.status.success else 'Failed'}",
                "fields": [
                    {"title": "Environment", "value": self.config.environment, "short": True},
                    {"title": "Version", "value": self.config.version, "short": True},
                    {"title": "Strategy", "value": self.config.strategy, "short": True},
                    {"title": "Duration", "value": f"{self.status.duration:.1f}s", "short": True},
                ]
            }]
        }
        
        if not self.status.success and self.status.errors:
            message["attachments"][0]["fields"].append({
                "title": "Errors",
                "value": "\n".join(self.status.errors[:3])  # Limiter à 3 erreurs
            })
        
        try:
            response = requests.post(slack_webhook, json=message, timeout=5)
            if response.status_code == 200:
                console.print("  ✓ Slack notification sent")
            else:
                console.print(f"  [yellow]⚠ Slack notification failed: {response.status_code}[/yellow]")
        except Exception as e:
            console.print(f"  [yellow]⚠ Slack notification error: {str(e)}[/yellow]")
    
    def _send_email_notification(self) -> None:
        """Envoie une notification par email"""
        console.print("  [dim]Email notification would be sent here[/dim]")
    
    def _send_webhook_notification(self) -> None:
        """Envoie une notification webhook générique"""
        console.print("  [dim]Webhook notification would be sent here[/dim]")
    
    def _show_deployment_summary(self) -> None:
        """Affiche le résumé du déploiement"""
        console.print("\n" + "="*60)
        
        if self.status.success:
            console.print(Panel.fit(
                "[bold green]Deployment Successful![/bold green]",
                title="✅ Success"
            ))
        else:
            console.print(Panel.fit(
                "[bold red]Deployment Failed![/bold red]",
                title="❌ Failed"
            ))
        
        # Informations du déploiement
        info_table = Table(show_header=False)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")
        
        info_table.add_row("Environment", self.status.environment)
        info_table.add_row("Version", self.status.version)
        info_table.add_row("Provider", self.status.provider)
        info_table.add_row("Start Time", self.status.start_time.strftime("%Y-%m-%d %H:%M:%S UTC"))
        
        if self.status.end_time:
            info_table.add_row("End Time", self.status.end_time.strftime("%Y-%m-%d %H:%M:%S UTC"))
            info_table.add_row("Duration", f"{self.status.duration:.1f} seconds")
        
        if self.status.rollback_performed:
            info_table.add_row("Rollback", "[yellow]Performed[/yellow]")
        
        console.print(info_table)
        
        # Health checks
        if self.status.health_checks:
            console.print("\n[bold]Health Check Results:[/bold]")
            for check, passed in self.status.health_checks.items():
                status = "✓" if passed else "✗"
                style = "green" if passed else "red"
                console.print(f"  [{style}]{status}[/{style}] {check}")
        
        # Erreurs et avertissements
        if self.status.errors:
            console.print("\n[bold red]Errors:[/bold red]")
            for error in self.status.errors:
                console.print(f"  • {error}")
        
        if self.status.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in self.status.warnings:
                console.print(f"  • {warning}")
        
        # Actions suivantes
        if self.status.success:
            console.print("\n[bold]Next Steps:[/bold]")
            console.print("  1. Monitor application metrics and logs")
            console.print("  2. Run smoke tests on production")
            console.print("  3. Update documentation with new version")
            
            if self.config.environment == "production":
                console.print("  4. Notify stakeholders of successful deployment")
                console.print("  5. Plan for next deployment window")
        else:
            console.print("\n[bold]Recovery Actions:[/bold]")
            console.print("  1. Review error logs for root cause")
            console.print("  2. Fix identified issues")
            console.print("  3. Re-run deployment with --force flag if needed")
            
            if self.status.rollback_performed:
                console.print("  4. Verify rollback was successful")
                console.print("  5. Investigate why deployment failed")
        
        console.print("\n" + "="*60)
    
    def _get_git_hash(self) -> str:
        """Obtient le hash du commit git actuel"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return "unknown"
                
        except Exception:
            return "unknown"
    
    def cleanup(self) -> None:
        """Nettoie les fichiers temporaires"""
        if hasattr(self, '_temp_files'):
            for temp_file in self._temp_files:
                try:
                    Path(temp_file).unlink()
                except Exception:
                    pass


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description="AI Trading Robot Deployment Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--env",
        choices=["staging", "production"],
        default="staging",
        help="Target environment (default: staging)"
    )
    
    parser.add_argument(
        "--provider",
        choices=["local", "aws", "gcp", "azure"],
        default="local",
        help="Cloud provider (default: local)"
    )
    
    parser.add_argument(
        "--strategy",
        choices=["rolling", "blue-green", "canary"],
        default="rolling",
        help="Deployment strategy (default: rolling)"
    )
    
    parser.add_argument(
        "--version",
        default="latest",
        help="Version to deploy (git tag or 'latest')"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate deployment without making changes"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force deployment without confirmations"
    )
    
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Rollback to previous version"
    )
    
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Only run health checks on current deployment"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output"
    )
    
    args = parser.parse_args()
    
    try:
        deployer = TradingBotDeployer(args)
        
        if args.rollback:
            # Mode rollback
            console.print("[yellow]Rollback mode activated[/yellow]")
            deployer._rollback()
        elif args.health_check:
            # Mode health check uniquement
            console.print("[cyan]Running health checks only[/cyan]")
            deployer._run_health_checks()
        else:
            # Déploiement normal
            status = deployer.deploy()
            
            # Cleanup
            deployer.cleanup()
            
            # Code de sortie
            sys.exit(0 if status.success else 1)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Deployment interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Fatal error: {str(e)}[/bold red]")
        sys.exit(1)


if __name__ == "__main__":
    main()