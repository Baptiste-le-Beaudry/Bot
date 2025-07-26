#!/usr/bin/env python3
"""
Script d'Installation Complet pour le Robot de Trading Algorithmique IA
======================================================================

Ce script automatise l'installation complète du système de trading incluant:
- Vérification des prérequis système
- Installation des dépendances Python
- Configuration des bases de données (PostgreSQL/TimescaleDB, Redis, InfluxDB)
- Création de la structure de répertoires
- Génération des fichiers de configuration
- Installation des services système
- Tests de validation post-installation

Usage:
    python scripts/setup.py [options]
    
Options:
    --env               Environnement (development/staging/production)
    --skip-db          Ne pas configurer les bases de données
    --skip-services    Ne pas installer les services système
    --gpu              Configurer le support GPU (CUDA)
    --exchange         Exchanges à configurer (binance,coinbase,etc)
    --dry-run          Afficher les actions sans les exécuter

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import secrets
import string
import urllib.request
import zipfile
import tarfile

# Rich pour une meilleure interface console
try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Installing rich for better output...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint

console = Console()


class TradingBotInstaller:
    """Classe principale pour l'installation du robot de trading"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.project_root = Path(__file__).parent.parent.absolute()
        self.errors = []
        self.warnings = []
        self.installed_components = []
        
        # Configuration par défaut
        self.python_version = sys.version_info
        self.os_type = platform.system()
        self.architecture = platform.machine()
        
        # Chemins importants
        self.venv_path = self.project_root / "venv"
        self.config_dir = self.project_root / "config"
        self.data_dir = self.project_root / "data"
        self.logs_dir = self.project_root / "logs"
        self.models_dir = self.project_root / "models"
        self.checkpoints_dir = self.project_root / "checkpoints"
        
    def run(self) -> bool:
        """Lance le processus d'installation complet"""
        console.print(Panel.fit(
            f"[bold blue]AI Trading Robot Installation Script[/bold blue]\n"
            f"Version: 1.0.0 | Environment: {self.args.env}\n"
            f"OS: {self.os_type} | Python: {self.python_version.major}.{self.python_version.minor}.{self.python_version.micro}",
            title="🤖 Trading Bot Installer"
        ))
        
        if self.args.dry_run:
            console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]\n")
        
        steps = [
            ("Checking system requirements", self.check_requirements),
            ("Creating directory structure", self.create_directories),
            ("Setting up Python virtual environment", self.setup_virtualenv),
            ("Installing Python dependencies", self.install_dependencies),
            ("Configuring databases", self.setup_databases),
            ("Setting up Redis cache", self.setup_redis),
            ("Configuring monitoring services", self.setup_monitoring),
            ("Generating configuration files", self.generate_configs),
            ("Installing system services", self.install_services),
            ("Setting up GPU support", self.setup_gpu),
            ("Running validation tests", self.run_validation),
            ("Creating startup scripts", self.create_scripts)
        ]
        
        # Filtrer les étapes selon les options
        if self.args.skip_db:
            steps = [(name, func) for name, func in steps if "database" not in name.lower() and "redis" not in name.lower()]
        if self.args.skip_services:
            steps = [(name, func) for name, func in steps if "service" not in name.lower()]
        if not self.args.gpu:
            steps = [(name, func) for name, func in steps if "GPU" not in name]
        
        # Exécuter chaque étape
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            
            task = progress.add_task("Installing...", total=len(steps))
            
            for step_name, step_func in steps:
                progress.update(task, description=f"[cyan]{step_name}[/cyan]")
                
                try:
                    if not self.args.dry_run:
                        success = step_func()
                        if not success and step_name in ["Checking system requirements"]:
                            console.print(f"[red]✗ {step_name} failed - aborting installation[/red]")
                            return False
                    else:
                        console.print(f"[dim]Would execute: {step_name}[/dim]")
                        time.sleep(0.5)  # Simulation delay
                    
                    self.installed_components.append(step_name)
                    progress.advance(task)
                    
                except Exception as e:
                    self.errors.append(f"{step_name}: {str(e)}")
                    console.print(f"[red]✗ Error in {step_name}: {str(e)}[/red]")
                    if step_name in ["Checking system requirements", "Installing Python dependencies"]:
                        return False
                    progress.advance(task)
        
        # Afficher le résumé
        self.show_summary()
        
        return len(self.errors) == 0
    
    def check_requirements(self) -> bool:
        """Vérifie les prérequis système"""
        console.print("\n[bold]Checking System Requirements:[/bold]")
        
        requirements = {
            "Python Version": self.check_python_version(),
            "Operating System": self.check_os(),
            "Available RAM": self.check_memory(),
            "Available Disk Space": self.check_disk_space(),
            "Network Connectivity": self.check_network(),
            "Required Tools": self.check_tools(),
            "Database Servers": self.check_databases() if not self.args.skip_db else (True, "Skipped"),
        }
        
        # Afficher les résultats
        table = Table(title="System Requirements Check")
        table.add_column("Requirement", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        
        all_passed = True
        for req_name, (passed, details) in requirements.items():
            status = "✓ Pass" if passed else "✗ Fail"
            style = "green" if passed else "red"
            table.add_row(req_name, f"[{style}]{status}[/{style}]", details)
            if not passed:
                all_passed = False
        
        console.print(table)
        return all_passed
    
    def check_python_version(self) -> Tuple[bool, str]:
        """Vérifie la version de Python"""
        min_version = (3, 8)
        current = (self.python_version.major, self.python_version.minor)
        
        if current >= min_version:
            return True, f"{current[0]}.{current[1]}.{self.python_version.micro}"
        else:
            self.errors.append(f"Python {min_version[0]}.{min_version[1]}+ required, found {current[0]}.{current[1]}")
            return False, f"Need {min_version[0]}.{min_version[1]}+, have {current[0]}.{current[1]}"
    
    def check_os(self) -> Tuple[bool, str]:
        """Vérifie le système d'exploitation"""
        supported = ["Linux", "Darwin", "Windows"]
        if self.os_type in supported:
            return True, f"{self.os_type} {platform.release()}"
        else:
            self.warnings.append(f"Untested OS: {self.os_type}")
            return True, f"{self.os_type} (untested)"
    
    def check_memory(self) -> Tuple[bool, str]:
        """Vérifie la mémoire disponible"""
        try:
            import psutil
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
            import psutil
        
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        total_gb = mem.total / (1024**3)
        
        min_required = 8 if self.args.env == "production" else 4
        
        if available_gb >= min_required:
            return True, f"{available_gb:.1f}GB available of {total_gb:.1f}GB"
        else:
            self.errors.append(f"Insufficient memory: {available_gb:.1f}GB available, {min_required}GB required")
            return False, f"Need {min_required}GB, have {available_gb:.1f}GB"
    
    def check_disk_space(self) -> Tuple[bool, str]:
        """Vérifie l'espace disque disponible"""
        import shutil
        
        stat = shutil.disk_usage(self.project_root)
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        
        min_required = 50 if self.args.env == "production" else 20
        
        if free_gb >= min_required:
            return True, f"{free_gb:.1f}GB free of {total_gb:.1f}GB"
        else:
            self.warnings.append(f"Low disk space: {free_gb:.1f}GB free, {min_required}GB recommended")
            return True, f"Recommend {min_required}GB, have {free_gb:.1f}GB"
    
    def check_network(self) -> Tuple[bool, str]:
        """Vérifie la connectivité réseau"""
        test_urls = [
            "https://api.binance.com",
            "https://pypi.org",
            "https://github.com"
        ]
        
        failed = []
        for url in test_urls:
            try:
                urllib.request.urlopen(url, timeout=5)
            except Exception:
                failed.append(url.split('/')[2])
        
        if not failed:
            return True, "All endpoints reachable"
        elif len(failed) < len(test_urls):
            self.warnings.append(f"Some endpoints unreachable: {', '.join(failed)}")
            return True, f"Partial connectivity ({len(failed)} failed)"
        else:
            self.errors.append("No network connectivity")
            return False, "No connectivity"
    
    def check_tools(self) -> Tuple[bool, str]:
        """Vérifie les outils système requis"""
        required_tools = {
            "git": "Git version control",
            "curl": "HTTP client" if self.os_type != "Windows" else None,
            "docker": "Container runtime" if not self.args.skip_services else None,
        }
        
        missing = []
        for tool, description in required_tools.items():
            if description is None:
                continue
            if shutil.which(tool) is None:
                missing.append(tool)
        
        if not missing:
            return True, "All tools available"
        else:
            self.warnings.append(f"Missing tools: {', '.join(missing)}")
            return True, f"Missing: {', '.join(missing)}"
    
    def check_databases(self) -> Tuple[bool, str]:
        """Vérifie la disponibilité des serveurs de base de données"""
        checks = []
        
        # PostgreSQL/TimescaleDB
        try:
            subprocess.run(["pg_config", "--version"], capture_output=True, check=True)
            checks.append("PostgreSQL ✓")
        except Exception:
            checks.append("PostgreSQL ✗")
        
        # Redis
        if shutil.which("redis-cli"):
            checks.append("Redis ✓")
        else:
            checks.append("Redis ✗")
        
        return True, ", ".join(checks)
    
    def create_directories(self) -> bool:
        """Crée la structure de répertoires du projet"""
        console.print("\n[bold]Creating Directory Structure:[/bold]")
        
        directories = [
            self.config_dir,
            self.data_dir / "raw",
            self.data_dir / "processed", 
            self.data_dir / "features",
            self.logs_dir / "trading",
            self.logs_dir / "system",
            self.logs_dir / "errors",
            self.models_dir / "trained",
            self.models_dir / "production",
            self.checkpoints_dir,
            self.project_root / "backups",
            self.project_root / "reports",
            self.project_root / "cache"
        ]
        
        for directory in directories:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                console.print(f"  ✓ Created: {directory.relative_to(self.project_root)}")
            else:
                console.print(f"  [dim]→ Exists: {directory.relative_to(self.project_root)}[/dim]")
        
        # Créer les fichiers __init__.py
        init_dirs = [
            "core", "strategies", "ml", "data", "risk", 
            "execution", "monitoring", "config", "utils", "tests"
        ]
        
        for dir_name in init_dirs:
            dir_path = self.project_root / dir_name
            if not dir_path.exists():
                dir_path.mkdir(exist_ok=True)
            
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text(f'"""{dir_name.capitalize()} module for AI Trading Robot"""\n')
        
        return True
    
    def setup_virtualenv(self) -> bool:
        """Configure l'environnement virtuel Python"""
        console.print("\n[bold]Setting up Python Virtual Environment:[/bold]")
        
        if self.venv_path.exists():
            console.print("  [yellow]→ Virtual environment already exists[/yellow]")
            return True
        
        # Créer le venv
        console.print("  Creating virtual environment...")
        subprocess.check_call([sys.executable, "-m", "venv", str(self.venv_path)])
        
        # Déterminer le chemin de pip selon l'OS
        if self.os_type == "Windows":
            pip_path = self.venv_path / "Scripts" / "pip.exe"
            python_path = self.venv_path / "Scripts" / "python.exe"
        else:
            pip_path = self.venv_path / "bin" / "pip"
            python_path = self.venv_path / "bin" / "python"
        
        # Mettre à jour pip
        console.print("  Updating pip...")
        subprocess.check_call([str(python_path), "-m", "pip", "install", "--upgrade", "pip"])
        
        # Installer wheel et setuptools
        subprocess.check_call([str(pip_path), "install", "--upgrade", "wheel", "setuptools"])
        
        console.print("  ✓ Virtual environment created")
        return True
    
    def install_dependencies(self) -> bool:
        """Installe les dépendances Python"""
        console.print("\n[bold]Installing Python Dependencies:[/bold]")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            self.errors.append("requirements.txt not found")
            return False
        
        # Déterminer le chemin de pip
        if self.os_type == "Windows":
            pip_path = self.venv_path / "Scripts" / "pip.exe" if self.venv_path.exists() else "pip"
        else:
            pip_path = self.venv_path / "bin" / "pip" if self.venv_path.exists() else "pip3"
        
        # Installer les dépendances par catégorie pour un meilleur contrôle
        categories = {
            "Core": ["pydantic", "structlog", "aiohttp", "asyncio", "websockets"],
            "Trading": ["ccxt", "python-binance", "alpaca-trade-api"],
            "Data": ["pandas", "numpy", "polars", "redis", "psycopg2-binary"],
            "ML/AI": ["torch", "tensorflow", "scikit-learn", "stable-baselines3"],
            "Monitoring": ["prometheus-client", "rich", "psutil"]
        }
        
        for category, packages in categories.items():
            console.print(f"\n  Installing {category} packages...")
            for package in packages:
                try:
                    # Vérifier si le package est dans requirements.txt
                    if any(package in line for line in requirements_file.read_text().splitlines()):
                        subprocess.check_call(
                            [str(pip_path), "install", package],
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                        console.print(f"    ✓ {package}")
                except subprocess.CalledProcessError:
                    self.warnings.append(f"Failed to install {package}")
                    console.print(f"    [yellow]⚠ {package} (failed)[/yellow]")
        
        # Installer toutes les dépendances restantes
        console.print("\n  Installing remaining dependencies...")
        try:
            subprocess.check_call([str(pip_path), "install", "-r", str(requirements_file)])
            console.print("  ✓ All dependencies installed")
            return True
        except subprocess.CalledProcessError as e:
            self.errors.append(f"Failed to install dependencies: {e}")
            return False
    
    def setup_databases(self) -> bool:
        """Configure les bases de données"""
        if self.args.skip_db:
            return True
        
        console.print("\n[bold]Setting up Databases:[/bold]")
        
        # PostgreSQL/TimescaleDB
        if not self.setup_postgresql():
            return False
        
        # Redis est configuré séparément
        
        # InfluxDB pour les métriques
        if self.args.env in ["staging", "production"]:
            self.setup_influxdb()
        
        return True
    
    def setup_postgresql(self) -> bool:
        """Configure PostgreSQL avec TimescaleDB"""
        console.print("\n  [cyan]PostgreSQL/TimescaleDB Setup:[/cyan]")
        
        # Générer les credentials
        db_password = self.generate_password()
        
        # Créer le script SQL
        sql_script = f"""
-- Create user and database for trading bot
CREATE USER trading_bot WITH PASSWORD '{db_password}';
CREATE DATABASE trading_bot_db OWNER trading_bot;
GRANT ALL PRIVILEGES ON DATABASE trading_bot_db TO trading_bot;

-- Connect to the database
\\c trading_bot_db;

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS market_data;
CREATE SCHEMA IF NOT EXISTS trading;
CREATE SCHEMA IF NOT EXISTS analytics;

GRANT ALL ON SCHEMA market_data TO trading_bot;
GRANT ALL ON SCHEMA trading TO trading_bot;
GRANT ALL ON SCHEMA analytics TO trading_bot;

-- Create tables
CREATE TABLE market_data.ohlcv (
    time        TIMESTAMPTZ NOT NULL,
    exchange    VARCHAR(50) NOT NULL,
    symbol      VARCHAR(20) NOT NULL,
    open        DECIMAL(20,8) NOT NULL,
    high        DECIMAL(20,8) NOT NULL,
    low         DECIMAL(20,8) NOT NULL,
    close       DECIMAL(20,8) NOT NULL,
    volume      DECIMAL(20,8) NOT NULL,
    trades      INTEGER
);

-- Convert to hypertable
SELECT create_hypertable('market_data.ohlcv', 'time');

-- Create indexes
CREATE INDEX idx_ohlcv_symbol_time ON market_data.ohlcv (symbol, time DESC);
CREATE INDEX idx_ohlcv_exchange_symbol ON market_data.ohlcv (exchange, symbol, time DESC);

-- Compression policy (after 7 days)
SELECT add_compression_policy('market_data.ohlcv', INTERVAL '7 days');

-- Retention policy (keep 1 year)
SELECT add_retention_policy('market_data.ohlcv', INTERVAL '365 days');
"""
        
        # Sauvegarder le script
        setup_sql = self.project_root / "scripts" / "setup_db.sql"
        setup_sql.parent.mkdir(exist_ok=True)
        setup_sql.write_text(sql_script)
        
        # Sauvegarder les credentials
        db_config = {
            "host": "localhost",
            "port": 5432,
            "database": "trading_bot_db",
            "username": "trading_bot",
            "password": db_password
        }
        
        db_config_file = self.config_dir / "database.json"
        db_config_file.write_text(json.dumps(db_config, indent=2))
        db_config_file.chmod(0o600)  # Lecture seule pour le propriétaire
        
        console.print("    ✓ Database configuration saved")
        console.print(f"    ✓ SQL script saved to: {setup_sql}")
        console.print("    [yellow]→ Run the SQL script manually:[/yellow]")
        console.print(f"      psql -U postgres -f {setup_sql}")
        
        return True
    
    def setup_redis(self) -> bool:
        """Configure Redis pour le cache"""
        if self.args.skip_db:
            return True
        
        console.print("\n[bold]Setting up Redis Cache:[/bold]")
        
        # Configuration Redis
        redis_config = """
# Redis configuration for Trading Bot
bind 127.0.0.1
port 6379
protected-mode yes
requirepass {password}

# Persistence
save 900 1
save 300 10
save 60 10000
dbfilename trading_bot.rdb
dir ./redis_data/

# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Performance
tcp-keepalive 60
timeout 300

# Logging
loglevel notice
logfile ./logs/redis.log
"""
        
        redis_password = self.generate_password()
        redis_config = redis_config.format(password=redis_password)
        
        # Sauvegarder la configuration
        redis_dir = self.project_root / "redis"
        redis_dir.mkdir(exist_ok=True)
        
        redis_conf = redis_dir / "redis.conf"
        redis_conf.write_text(redis_config)
        redis_conf.chmod(0o600)
        
        # Sauvegarder les credentials
        redis_creds = {
            "host": "localhost",
            "port": 6379,
            "password": redis_password
        }
        
        redis_config_file = self.config_dir / "redis.json"
        redis_config_file.write_text(json.dumps(redis_creds, indent=2))
        redis_config_file.chmod(0o600)
        
        console.print("  ✓ Redis configuration saved")
        console.print("  [yellow]→ Start Redis with:[/yellow]")
        console.print(f"    redis-server {redis_conf}")
        
        return True
    
    def setup_influxdb(self) -> bool:
        """Configure InfluxDB pour les métriques"""
        console.print("\n  [cyan]InfluxDB Setup:[/cyan]")
        
        # Configuration InfluxDB
        influx_config = {
            "url": "http://localhost:8086",
            "token": self.generate_token(),
            "org": "trading-bot",
            "bucket": "metrics"
        }
        
        influx_config_file = self.config_dir / "influxdb.json"
        influx_config_file.write_text(json.dumps(influx_config, indent=2))
        influx_config_file.chmod(0o600)
        
        console.print("    ✓ InfluxDB configuration saved")
        return True
    
    def setup_monitoring(self) -> bool:
        """Configure Prometheus et Grafana"""
        console.print("\n[bold]Setting up Monitoring:[/bold]")
        
        # Configuration Prometheus
        prometheus_config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'trading-bot'
    static_configs:
      - targets: ['localhost:9090']
    
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['localhost:9100']
      
  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
"""
        
        monitoring_dir = self.project_root / "monitoring"
        monitoring_dir.mkdir(exist_ok=True)
        
        prometheus_yml = monitoring_dir / "prometheus.yml"
        prometheus_yml.write_text(prometheus_config)
        
        # Configuration Grafana (datasources)
        grafana_datasources = {
            "apiVersion": 1,
            "datasources": [
                {
                    "name": "Prometheus",
                    "type": "prometheus",
                    "access": "proxy",
                    "url": "http://localhost:9090"
                },
                {
                    "name": "InfluxDB",
                    "type": "influxdb",
                    "access": "proxy",
                    "url": "http://localhost:8086"
                }
            ]
        }
        
        grafana_dir = monitoring_dir / "grafana"
        grafana_dir.mkdir(exist_ok=True)
        
        datasources_yml = grafana_dir / "datasources.yml"
        datasources_yml.write_text(json.dumps(grafana_datasources, indent=2))
        
        console.print("  ✓ Prometheus configuration saved")
        console.print("  ✓ Grafana datasources configured")
        
        return True
    
    def generate_configs(self) -> bool:
        """Génère les fichiers de configuration"""
        console.print("\n[bold]Generating Configuration Files:[/bold]")
        
        # Charger les configurations des bases de données si elles existent
        db_config = {}
        redis_config = {}
        
        db_file = self.config_dir / "database.json"
        redis_file = self.config_dir / "redis.json"
        
        if db_file.exists():
            db_config = json.loads(db_file.read_text())
        if redis_file.exists():
            redis_config = json.loads(redis_file.read_text())
        
        # Configuration principale
        main_config = {
            "app_name": "AI Trading Robot",
            "version": "1.0.0",
            "environment": self.args.env,
            "log_level": "DEBUG" if self.args.env == "development" else "INFO",
            
            "database": db_config or {
                "type": "timescaledb",
                "host": "localhost",
                "port": 5432,
                "database": "trading_bot_db",
                "username": "trading_bot",
                "password": "CHANGE_ME"
            },
            
            "redis": redis_config or {
                "host": "localhost",
                "port": 6379,
                "password": "CHANGE_ME"
            },
            
            "exchanges": {},
            
            "risk": {
                "max_drawdown_percent": 20.0,
                "max_daily_loss_percent": 5.0,
                "max_position_size": 0.05,
                "max_leverage": 3.0
            },
            
            "strategies": {
                "statistical_arbitrage": {
                    "name": "Statistical Arbitrage",
                    "type": "statistical_arbitrage",
                    "enabled": True,
                    "capital_allocation": 0.3,
                    "parameters": {
                        "lookback_period": 100,
                        "z_score_threshold": 2.0,
                        "min_correlation": 0.7
                    }
                },
                "market_making": {
                    "name": "Market Making",
                    "type": "market_making", 
                    "enabled": True,
                    "capital_allocation": 0.3,
                    "parameters": {
                        "spread_multiplier": 1.5,
                        "inventory_target": 0.5,
                        "max_position_size": 1000
                    }
                }
            },
            
            "active_strategies": ["statistical_arbitrage", "market_making"],
            
            "ml": {
                "models_dir": str(self.models_dir),
                "data_dir": str(self.data_dir),
                "checkpoints_dir": str(self.checkpoints_dir),
                "batch_size": 32,
                "learning_rate": 0.001
            },
            
            "monitoring": {
                "metrics_enabled": True,
                "alerts_enabled": True,
                "alert_channels": ["console", "file"]
            }
        }
        
        # Ajouter les configurations des exchanges si spécifiés
        if self.args.exchange:
            for exchange in self.args.exchange.split(','):
                main_config["exchanges"][exchange] = {
                    "name": exchange,
                    "exchange_type": exchange,
                    "enabled": True,
                    "api_key": "YOUR_API_KEY",
                    "api_secret": "YOUR_API_SECRET",
                    "sandbox_mode": self.args.env != "production"
                }
        
        # Sauvegarder la configuration
        config_file = self.config_dir / f"config.{self.args.env}.json"
        config_file.write_text(json.dumps(main_config, indent=2))
        
        # Créer le fichier .env
        env_content = f"""
# AI Trading Robot Environment Configuration
# Generated on {datetime.now().isoformat()}

TRADING_ENVIRONMENT={self.args.env}
TRADING_APP_NAME="AI Trading Robot"
TRADING_LOG_LEVEL={main_config['log_level']}

# Database
TRADING_DATABASE__HOST={main_config['database']['host']}
TRADING_DATABASE__PORT={main_config['database']['port']}
TRADING_DATABASE__DATABASE={main_config['database']['database']}
TRADING_DATABASE__USERNAME={main_config['database']['username']}
TRADING_DATABASE__PASSWORD={main_config['database']['password']}

# Redis
TRADING_REDIS__HOST={main_config['redis']['host']}
TRADING_REDIS__PORT={main_config['redis']['port']}
TRADING_REDIS__PASSWORD={main_config['redis']['password']}

# Add your exchange API keys here
# TRADING_EXCHANGES__BINANCE__API_KEY=your_key_here
# TRADING_EXCHANGES__BINANCE__API_SECRET=your_secret_here
"""
        
        env_file = self.project_root / ".env"
        env_file.write_text(env_content.strip())
        env_file.chmod(0o600)
        
        console.print(f"  ✓ Main configuration saved to: {config_file}")
        console.print(f"  ✓ Environment file saved to: {env_file}")
        
        return True
    
    def install_services(self) -> bool:
        """Installe les services système"""
        if self.args.skip_services:
            return True
        
        console.print("\n[bold]Installing System Services:[/bold]")
        
        # Créer les scripts de démarrage
        scripts_dir = self.project_root / "scripts"
        scripts_dir.mkdir(exist_ok=True)
        
        # Script de démarrage principal
        if self.os_type == "Windows":
            start_script = scripts_dir / "start_bot.bat"
            start_content = f"""
@echo off
echo Starting AI Trading Robot...

REM Activate virtual environment
call {self.venv_path}\\Scripts\\activate.bat

REM Start Redis
start "Redis" redis-server {self.project_root}\\redis\\redis.conf

REM Start the trading bot
python {self.project_root}\\main.py

pause
"""
        else:
            start_script = scripts_dir / "start_bot.sh"
            start_content = f"""#!/bin/bash
echo "Starting AI Trading Robot..."

# Activate virtual environment
source {self.venv_path}/bin/activate

# Start Redis in background
redis-server {self.project_root}/redis/redis.conf &

# Start the trading bot
python {self.project_root}/main.py
"""
        
        start_script.write_text(start_content)
        if self.os_type != "Windows":
            start_script.chmod(0o755)
        
        # Service systemd pour Linux
        if self.os_type == "Linux" and self.args.env == "production":
            service_content = f"""
[Unit]
Description=AI Trading Robot
After=network.target postgresql.service redis.service

[Service]
Type=simple
User={os.getenv('USER')}
WorkingDirectory={self.project_root}
Environment="PATH={self.venv_path}/bin:/usr/local/bin:/usr/bin:/bin"
ExecStart={self.venv_path}/bin/python {self.project_root}/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
            
            service_file = scripts_dir / "trading-bot.service"
            service_file.write_text(service_content.strip())
            
            console.print("  ✓ Systemd service file created")
            console.print("  [yellow]→ Install service with:[/yellow]")
            console.print(f"    sudo cp {service_file} /etc/systemd/system/")
            console.print("    sudo systemctl daemon-reload")
            console.print("    sudo systemctl enable trading-bot")
        
        console.print(f"  ✓ Start script created: {start_script}")
        
        return True
    
    def setup_gpu(self) -> bool:
        """Configure le support GPU si demandé"""
        if not self.args.gpu:
            return True
        
        console.print("\n[bold]Setting up GPU Support:[/bold]")
        
        # Vérifier CUDA
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                console.print(f"  ✓ CUDA available: {gpu_count} GPU(s) detected")
                console.print(f"  ✓ Primary GPU: {gpu_name}")
            else:
                console.print("  [yellow]⚠ CUDA not available - CPU mode will be used[/yellow]")
                self.warnings.append("CUDA not available")
        except ImportError:
            console.print("  [yellow]⚠ PyTorch not installed - skipping GPU check[/yellow]")
        
        return True
    
    def run_validation(self) -> bool:
        """Exécute les tests de validation"""
        console.print("\n[bold]Running Validation Tests:[/bold]")
        
        # Test d'import des modules principaux
        console.print("  Testing module imports...")
        test_imports = [
            "pandas",
            "numpy", 
            "ccxt",
            "torch",
            "redis",
            "asyncio",
            "structlog"
        ]
        
        failed_imports = []
        for module in test_imports:
            try:
                __import__(module)
                console.print(f"    ✓ {module}")
            except ImportError:
                failed_imports.append(module)
                console.print(f"    [red]✗ {module}[/red]")
        
        if failed_imports:
            self.warnings.append(f"Failed imports: {', '.join(failed_imports)}")
        
        # Test de connexion Redis si configuré
        if not self.args.skip_db:
            console.print("\n  Testing Redis connection...")
            try:
                import redis
                r = redis.Redis(host='localhost', port=6379, decode_responses=True)
                r.ping()
                console.print("    ✓ Redis connection successful")
            except Exception as e:
                console.print(f"    [yellow]⚠ Redis connection failed: {str(e)}[/yellow]")
                self.warnings.append("Redis connection failed")
        
        return True
    
    def create_scripts(self) -> bool:
        """Crée des scripts utilitaires supplémentaires"""
        console.print("\n[bold]Creating Utility Scripts:[/bold]")
        
        scripts_dir = self.project_root / "scripts"
        
        # Script de backup
        backup_script = scripts_dir / "backup.py"
        backup_content = '''#!/usr/bin/env python3
"""Script de backup pour le robot de trading"""

import os
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

def create_backup():
    backup_dir = Path("backups")
    backup_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"trading_bot_backup_{timestamp}.tar.gz"
    
    with tarfile.open(backup_dir / backup_name, "w:gz") as tar:
        # Backup config
        tar.add("config", arcname="config")
        # Backup models
        tar.add("models", arcname="models")
        # Backup logs
        tar.add("logs", arcname="logs")
    
    print(f"Backup created: {backup_dir / backup_name}")

if __name__ == "__main__":
    create_backup()
'''
        backup_script.write_text(backup_content)
        backup_script.chmod(0o755)
        
        # Script de monitoring
        monitor_script = scripts_dir / "monitor.py"
        monitor_content = '''#!/usr/bin/env python3
"""Script de monitoring simple pour le robot de trading"""

import psutil
import time
from datetime import datetime

def monitor_system():
    while True:
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        disk = psutil.disk_usage('/').percent
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"CPU: {cpu:5.1f}% | "
              f"Memory: {memory:5.1f}% | "
              f"Disk: {disk:5.1f}%")
        
        time.sleep(10)

if __name__ == "__main__":
    try:
        monitor_system()
    except KeyboardInterrupt:
        print("\\nMonitoring stopped.")
'''
        monitor_script.write_text(monitor_content)
        monitor_script.chmod(0o755)
        
        console.print(f"  ✓ Backup script created: {backup_script}")
        console.print(f"  ✓ Monitor script created: {monitor_script}")
        
        return True
    
    def show_summary(self) -> None:
        """Affiche le résumé de l'installation"""
        console.print("\n" + "="*60)
        console.print(Panel.fit(
            "[bold green]Installation Summary[/bold green]",
            title="🎉 Complete"
        ))
        
        # Composants installés
        if self.installed_components:
            console.print("\n[bold]Successfully Installed:[/bold]")
            for component in self.installed_components:
                console.print(f"  ✓ {component}")
        
        # Avertissements
        if self.warnings:
            console.print("\n[bold yellow]Warnings:[/bold yellow]")
            for warning in self.warnings:
                console.print(f"  ⚠ {warning}")
        
        # Erreurs
        if self.errors:
            console.print("\n[bold red]Errors:[/bold red]")
            for error in self.errors:
                console.print(f"  ✗ {error}")
        
        # Prochaines étapes
        console.print("\n[bold]Next Steps:[/bold]")
        console.print("  1. Review and update configuration files in ./config/")
        console.print("  2. Add your exchange API credentials to .env file")
        
        if not self.args.skip_db:
            console.print("  3. Initialize the database:")
            console.print("     psql -U postgres -f scripts/setup_db.sql")
            console.print("  4. Start Redis server:")
            console.print("     redis-server redis/redis.conf")
        
        console.print("  5. Start the trading bot:")
        if self.os_type == "Windows":
            console.print("     scripts\\start_bot.bat")
        else:
            console.print("     ./scripts/start_bot.sh")
        
        console.print("\n[bold]Documentation:[/bold]")
        console.print("  - Configuration Guide: docs/configuration.md")
        console.print("  - API Reference: docs/api.md")
        console.print("  - Strategy Development: docs/strategies.md")
        
        console.print("\n" + "="*60)
    
    def generate_password(self, length: int = 24) -> str:
        """Génère un mot de passe sécurisé"""
        alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def generate_token(self, length: int = 32) -> str:
        """Génère un token sécurisé"""
        return secrets.token_urlsafe(length)


def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(
        description="AI Trading Robot Installation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--env",
        choices=["development", "staging", "production"],
        default="development",
        help="Environment to install for (default: development)"
    )
    
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip database setup"
    )
    
    parser.add_argument(
        "--skip-services",
        action="store_true",
        help="Skip system services installation"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Configure GPU/CUDA support"
    )
    
    parser.add_argument(
        "--exchange",
        help="Comma-separated list of exchanges to configure (e.g., binance,coinbase)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    # Vérifier qu'on est dans le bon répertoire
    if not (Path.cwd() / "requirements.txt").exists():
        console.print("[red]Error: Must run from project root directory[/red]")
        console.print("Please cd to the trading bot project directory first")
        return 1
    
    # Lancer l'installation
    installer = TradingBotInstaller(args)
    success = installer.run()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())