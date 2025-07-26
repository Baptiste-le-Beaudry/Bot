"""
Système de Surveillance Complète pour Robot de Trading Algorithmique IA
======================================================================

Ce module assure la surveillance en temps réel de tous les composants du système
de trading : infrastructure, services, performances, connexions, et métriques métier.
Intégration native avec Prometheus, Grafana, et systèmes d'alertes.

Fonctionnalités:
- Monitoring système (CPU, RAM, Disk, Network)
- Surveillance des services et processus
- Health checks des composants critiques
- Monitoring des connexions exchanges
- Métriques de trading en temps réel
- Détection d'anomalies par ML
- Dashboard temps réel WebSocket
- Intégration Prometheus/Grafana
- Alertes intelligentes multi-canal
- Logs structurés et traçabilité

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import asyncio
import json
import os
import platform
import socket
import sys
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union, Callable
import subprocess
import statistics
import psutil
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Monitoring et métriques
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, Info,
    start_http_server, generate_latest, CONTENT_TYPE_LATEST
)

# Détection d'anomalies
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# WebSocket pour dashboard temps réel
import aiohttp
from aiohttp import web
import socketio

# Visualisation
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Base de données temps réel
import redis
import influxdb_client
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# Logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_structured_logger
from utils.metrics import MetricsCollector
from monitoring.alerts import AlertManager, AlertSeverity
from config import get_config

console = Console()
logger = get_structured_logger(__name__)


class ComponentStatus(Enum):
    """États possibles des composants"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    STARTING = "starting"
    STOPPING = "stopping"


class MetricType(Enum):
    """Types de métriques collectées"""
    SYSTEM = "system"
    SERVICE = "service"
    TRADING = "trading"
    NETWORK = "network"
    DATABASE = "database"
    CUSTOM = "custom"


@dataclass
class HealthCheckResult:
    """Résultat d'un health check"""
    component: str
    status: ComponentStatus
    message: str
    timestamp: datetime
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_healthy(self) -> bool:
        return self.status == ComponentStatus.HEALTHY


@dataclass
class SystemMetrics:
    """Métriques système actuelles"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_usage_percent: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_recv_mb: float
    network_sent_mb: float
    open_connections: int
    process_count: int
    thread_count: int
    swap_percent: float
    load_average: Tuple[float, float, float]
    uptime_seconds: float


@dataclass
class TradingMetrics:
    """Métriques de trading actuelles"""
    timestamp: datetime
    active_positions: int
    total_volume_24h: float
    orders_per_second: float
    avg_order_latency_ms: float
    p99_order_latency_ms: float
    success_rate: float
    error_rate: float
    total_pnl: float
    unrealized_pnl: float
    current_drawdown: float
    active_strategies: List[str]
    connected_exchanges: List[str]


class SystemMonitor:
    """Moniteur système principal"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        self.running = False
        
        # Configuration
        self.collect_interval = self.config.get('monitoring', {}).get('collect_interval', 5)
        self.history_size = self.config.get('monitoring', {}).get('history_size', 1000)
        
        # Stockage des métriques
        self.metrics_history = defaultdict(lambda: deque(maxlen=self.history_size))
        self.health_status = {}
        self.anomaly_scores = defaultdict(list)
        
        # Composants à surveiller
        self.components = self._init_components()
        
        # Métriques Prometheus
        self._init_prometheus_metrics()
        
        # Détection d'anomalies
        self.anomaly_detector = None
        self._init_anomaly_detection()
        
        # WebSocket server
        self.sio = socketio.AsyncServer(async_mode='aiohttp', cors_allowed_origins='*')
        self.app = web.Application()
        self.sio.attach(self.app)
        
        # Connexions externes
        self.redis_client = None
        self.influx_client = None
        self._init_external_connections()
        
        # Alert manager
        self.alert_manager = AlertManager()
        
        # Thread pools
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("System monitor initialized", 
                   components=len(self.components),
                   collect_interval=self.collect_interval)
    
    def _init_components(self) -> Dict[str, Dict[str, Any]]:
        """Initialise la liste des composants à surveiller"""
        return {
            'trading_engine': {
                'name': 'Trading Engine',
                'check_func': self._check_trading_engine,
                'critical': True
            },
            'data_collector': {
                'name': 'Data Collector',
                'check_func': self._check_data_collector,
                'critical': True
            },
            'risk_manager': {
                'name': 'Risk Manager',
                'check_func': self._check_risk_manager,
                'critical': True
            },
            'database': {
                'name': 'Database',
                'check_func': self._check_database,
                'critical': True
            },
            'redis': {
                'name': 'Redis Cache',
                'check_func': self._check_redis,
                'critical': False
            },
            'exchanges': {
                'name': 'Exchange Connections',
                'check_func': self._check_exchanges,
                'critical': True
            },
            'ml_models': {
                'name': 'ML Models',
                'check_func': self._check_ml_models,
                'critical': False
            }
        }
    
    def _init_prometheus_metrics(self):
        """Initialise les métriques Prometheus"""
        # System metrics
        self.prom_cpu_usage = Gauge('system_cpu_usage_percent', 'CPU usage percentage')
        self.prom_memory_usage = Gauge('system_memory_usage_percent', 'Memory usage percentage')
        self.prom_disk_usage = Gauge('system_disk_usage_percent', 'Disk usage percentage')
        self.prom_network_recv = Counter('system_network_received_bytes', 'Network bytes received')
        self.prom_network_sent = Counter('system_network_sent_bytes', 'Network bytes sent')
        
        # Trading metrics
        self.prom_active_positions = Gauge('trading_active_positions', 'Number of active positions')
        self.prom_orders_per_second = Gauge('trading_orders_per_second', 'Orders executed per second')
        self.prom_order_latency = Histogram('trading_order_latency_ms', 'Order execution latency',
                                          buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000])
        self.prom_total_pnl = Gauge('trading_total_pnl', 'Total P&L')
        self.prom_error_rate = Gauge('trading_error_rate', 'Trading error rate')
        
        # Component health
        self.prom_component_health = Gauge('component_health_status', 'Component health status',
                                         ['component'])
        
        # Custom info
        self.prom_system_info = Info('system_info', 'System information')
        self.prom_system_info.info({
            'version': '1.0.0',
            'platform': platform.system(),
            'python_version': platform.python_version(),
            'hostname': socket.gethostname()
        })
    
    def _init_anomaly_detection(self):
        """Initialise le système de détection d'anomalies"""
        try:
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                n_estimators=100,
                random_state=42
            )
            self.scaler = StandardScaler()
            logger.info("Anomaly detection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize anomaly detection: {str(e)}")
            self.anomaly_detector = None
    
    def _init_external_connections(self):
        """Initialise les connexions externes"""
        # Redis
        try:
            redis_config = self.config.get('redis', {})
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                password=redis_config.get('password'),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}")
            self.redis_client = None
        
        # InfluxDB
        try:
            influx_config = self.config.get('influxdb', {})
            if influx_config:
                self.influx_client = InfluxDBClient(
                    url=influx_config.get('url', 'http://localhost:8086'),
                    token=influx_config.get('token'),
                    org=influx_config.get('org')
                )
                self.influx_write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
                self.influx_bucket = influx_config.get('bucket', 'metrics')
                logger.info("InfluxDB connection established")
        except Exception as e:
            logger.warning(f"InfluxDB connection failed: {str(e)}")
            self.influx_client = None
    
    async def start(self):
        """Démarre le monitoring système"""
        self.running = True
        
        # Démarrer le serveur Prometheus
        prometheus_port = self.config.get('monitoring', {}).get('prometheus_port', 9090)
        start_http_server(prometheus_port)
        logger.info(f"Prometheus metrics server started on port {prometheus_port}")
        
        # Démarrer le serveur WebSocket
        asyncio.create_task(self._start_websocket_server())
        
        # Démarrer les tâches de monitoring
        tasks = [
            asyncio.create_task(self._collect_system_metrics()),
            asyncio.create_task(self._collect_trading_metrics()),
            asyncio.create_task(self._perform_health_checks()),
            asyncio.create_task(self._detect_anomalies()),
            asyncio.create_task(self._broadcast_metrics())
        ]
        
        logger.info("System monitoring started")
        
        # Attendre que toutes les tâches se terminent
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Arrête le monitoring système"""
        self.running = False
        
        # Fermer les connexions
        if self.redis_client:
            self.redis_client.close()
        if self.influx_client:
            self.influx_client.close()
        
        # Arrêter l'executor
        self.executor.shutdown(wait=True)
        
        logger.info("System monitoring stopped")
    
    async def _collect_system_metrics(self):
        """Collecte les métriques système"""
        while self.running:
            try:
                metrics = self._get_system_metrics()
                
                # Stocker dans l'historique
                self.metrics_history['system'].append(metrics)
                
                # Mettre à jour Prometheus
                self.prom_cpu_usage.set(metrics.cpu_percent)
                self.prom_memory_usage.set(metrics.memory_percent)
                self.prom_disk_usage.set(metrics.disk_usage_percent)
                
                # Envoyer à InfluxDB
                if self.influx_client:
                    await self._write_to_influx('system_metrics', metrics)
                
                # Envoyer à Redis
                if self.redis_client:
                    self._write_to_redis('system:metrics:latest', metrics)
                
                # Vérifier les seuils
                await self._check_system_thresholds(metrics)
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {str(e)}")
            
            await asyncio.sleep(self.collect_interval)
    
    def _get_system_metrics(self) -> SystemMetrics:
        """Récupère les métriques système actuelles"""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk
        disk = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()
        
        # Network
        net_io = psutil.net_io_counters()
        connections = len(psutil.net_connections())
        
        # Process
        process_count = len(psutil.pids())
        current_process = psutil.Process()
        thread_count = current_process.num_threads()
        
        # Load average (Unix only)
        try:
            load_avg = os.getloadavg()
        except AttributeError:
            load_avg = (0.0, 0.0, 0.0)
        
        # Uptime
        boot_time = psutil.boot_time()
        uptime = time.time() - boot_time
        
        return SystemMetrics(
            timestamp=datetime.now(timezone.utc),
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_usage_percent=disk.percent,
            disk_io_read_mb=disk_io.read_bytes / (1024**2) if disk_io else 0,
            disk_io_write_mb=disk_io.write_bytes / (1024**2) if disk_io else 0,
            network_recv_mb=net_io.bytes_recv / (1024**2),
            network_sent_mb=net_io.bytes_sent / (1024**2),
            open_connections=connections,
            process_count=process_count,
            thread_count=thread_count,
            swap_percent=swap.percent,
            load_average=load_avg,
            uptime_seconds=uptime
        )
    
    async def _collect_trading_metrics(self):
        """Collecte les métriques de trading"""
        while self.running:
            try:
                metrics = await self._get_trading_metrics()
                
                # Stocker dans l'historique
                self.metrics_history['trading'].append(metrics)
                
                # Mettre à jour Prometheus
                self.prom_active_positions.set(metrics.active_positions)
                self.prom_orders_per_second.set(metrics.orders_per_second)
                self.prom_total_pnl.set(metrics.total_pnl)
                self.prom_error_rate.set(metrics.error_rate)
                
                # Envoyer à InfluxDB
                if self.influx_client:
                    await self._write_to_influx('trading_metrics', metrics)
                
                # Envoyer à Redis
                if self.redis_client:
                    self._write_to_redis('trading:metrics:latest', metrics)
                
                # Vérifier les alertes trading
                await self._check_trading_alerts(metrics)
                
            except Exception as e:
                logger.error(f"Error collecting trading metrics: {str(e)}")
            
            await asyncio.sleep(self.collect_interval)
    
    async def _get_trading_metrics(self) -> TradingMetrics:
        """Récupère les métriques de trading actuelles"""
        # Pour la démo, générer des métriques simulées
        # En production, interroger les vrais composants
        
        return TradingMetrics(
            timestamp=datetime.now(timezone.utc),
            active_positions=np.random.randint(5, 20),
            total_volume_24h=np.random.uniform(100000, 1000000),
            orders_per_second=np.random.uniform(10, 100),
            avg_order_latency_ms=np.random.uniform(1, 10),
            p99_order_latency_ms=np.random.uniform(10, 50),
            success_rate=np.random.uniform(0.95, 0.99),
            error_rate=np.random.uniform(0.001, 0.01),
            total_pnl=np.random.uniform(-5000, 10000),
            unrealized_pnl=np.random.uniform(-2000, 5000),
            current_drawdown=np.random.uniform(0, 0.15),
            active_strategies=['Statistical Arbitrage', 'Market Making', 'Scalping'],
            connected_exchanges=['Binance', 'Coinbase', 'Kraken']
        )
    
    async def _perform_health_checks(self):
        """Effectue les health checks des composants"""
        while self.running:
            try:
                tasks = []
                for component_id, component in self.components.items():
                    task = asyncio.create_task(self._check_component(component_id, component))
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Mettre à jour les statuts
                for i, (component_id, component) in enumerate(self.components.items()):
                    if isinstance(results[i], Exception):
                        logger.error(f"Health check failed for {component_id}: {results[i]}")
                        result = HealthCheckResult(
                            component=component['name'],
                            status=ComponentStatus.UNKNOWN,
                            message=str(results[i]),
                            timestamp=datetime.now(timezone.utc),
                            latency_ms=0
                        )
                    else:
                        result = results[i]
                    
                    self.health_status[component_id] = result
                    
                    # Mettre à jour Prometheus
                    status_value = 1 if result.is_healthy else 0
                    self.prom_component_health.labels(component=component_id).set(status_value)
                    
                    # Alerter si critique et non healthy
                    if component['critical'] and not result.is_healthy:
                        await self.alert_manager.send_alert(
                            severity=AlertSeverity.CRITICAL,
                            title=f"Component {component['name']} is {result.status.value}",
                            message=result.message,
                            metadata={'component': component_id, 'status': result.status.value}
                        )
                
            except Exception as e:
                logger.error(f"Error in health checks: {str(e)}")
            
            await asyncio.sleep(30)  # Health checks toutes les 30 secondes
    
    async def _check_component(self, component_id: str, component: Dict[str, Any]) -> HealthCheckResult:
        """Effectue le health check d'un composant"""
        start_time = time.time()
        
        try:
            # Appeler la fonction de check spécifique
            check_func = component['check_func']
            if asyncio.iscoroutinefunction(check_func):
                result = await check_func()
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor, check_func
                )
            
            latency_ms = (time.time() - start_time) * 1000
            
            return HealthCheckResult(
                component=component['name'],
                status=result['status'],
                message=result['message'],
                timestamp=datetime.now(timezone.utc),
                latency_ms=latency_ms,
                metadata=result.get('metadata', {})
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return HealthCheckResult(
                component=component['name'],
                status=ComponentStatus.UNHEALTHY,
                message=f"Check failed: {str(e)}",
                timestamp=datetime.now(timezone.utc),
                latency_ms=latency_ms
            )
    
    def _check_trading_engine(self) -> Dict[str, Any]:
        """Vérifie l'état du moteur de trading"""
        # Simulé pour la démo
        # En production, vérifier le vrai service
        
        if np.random.random() > 0.95:  # 5% de chance d'être unhealthy
            return {
                'status': ComponentStatus.UNHEALTHY,
                'message': 'Trading engine not responding',
                'metadata': {'last_trade': datetime.now().isoformat()}
            }
        
        return {
            'status': ComponentStatus.HEALTHY,
            'message': 'Trading engine operational',
            'metadata': {
                'active_strategies': 3,
                'orders_processed': 1234,
                'uptime_hours': 72.5
            }
        }
    
    def _check_data_collector(self) -> Dict[str, Any]:
        """Vérifie l'état du collecteur de données"""
        # Simulé pour la démo
        return {
            'status': ComponentStatus.HEALTHY,
            'message': 'Data collector running',
            'metadata': {
                'feeds_active': 5,
                'data_points_per_sec': 1000,
                'last_update': datetime.now().isoformat()
            }
        }
    
    def _check_risk_manager(self) -> Dict[str, Any]:
        """Vérifie l'état du gestionnaire de risques"""
        # Simulé pour la démo
        return {
            'status': ComponentStatus.HEALTHY,
            'message': 'Risk manager active',
            'metadata': {
                'risk_checks_per_min': 120,
                'alerts_triggered': 0,
                'max_exposure': 0.75
            }
        }
    
    def _check_database(self) -> Dict[str, Any]:
        """Vérifie l'état de la base de données"""
        try:
            # En production, faire un vrai test de connexion
            # Pour la démo, simuler
            if np.random.random() > 0.98:  # 2% de chance d'être dégradé
                return {
                    'status': ComponentStatus.DEGRADED,
                    'message': 'Database response slow',
                    'metadata': {'response_time_ms': 500}
                }
            
            return {
                'status': ComponentStatus.HEALTHY,
                'message': 'Database connected',
                'metadata': {
                    'response_time_ms': 5,
                    'connections_active': 10,
                    'disk_usage_gb': 45.2
                }
            }
        except Exception as e:
            return {
                'status': ComponentStatus.UNHEALTHY,
                'message': f'Database error: {str(e)}'
            }
    
    def _check_redis(self) -> Dict[str, Any]:
        """Vérifie l'état de Redis"""
        if self.redis_client:
            try:
                self.redis_client.ping()
                info = self.redis_client.info()
                
                return {
                    'status': ComponentStatus.HEALTHY,
                    'message': 'Redis operational',
                    'metadata': {
                        'used_memory_mb': info.get('used_memory', 0) / (1024**2),
                        'connected_clients': info.get('connected_clients', 0),
                        'ops_per_sec': info.get('instantaneous_ops_per_sec', 0)
                    }
                }
            except Exception as e:
                return {
                    'status': ComponentStatus.UNHEALTHY,
                    'message': f'Redis error: {str(e)}'
                }
        
        return {
            'status': ComponentStatus.UNKNOWN,
            'message': 'Redis not configured'
        }
    
    def _check_exchanges(self) -> Dict[str, Any]:
        """Vérifie l'état des connexions aux exchanges"""
        # Simulé pour la démo
        exchanges_status = {
            'Binance': 'connected' if np.random.random() > 0.05 else 'disconnected',
            'Coinbase': 'connected',
            'Kraken': 'connected'
        }
        
        disconnected = [ex for ex, status in exchanges_status.items() if status != 'connected']
        
        if disconnected:
            return {
                'status': ComponentStatus.DEGRADED,
                'message': f'Some exchanges disconnected: {", ".join(disconnected)}',
                'metadata': exchanges_status
            }
        
        return {
            'status': ComponentStatus.HEALTHY,
            'message': 'All exchanges connected',
            'metadata': exchanges_status
        }
    
    def _check_ml_models(self) -> Dict[str, Any]:
        """Vérifie l'état des modèles ML"""
        # Simulé pour la démo
        return {
            'status': ComponentStatus.HEALTHY,
            'message': 'ML models loaded',
            'metadata': {
                'models_active': ['DQN', 'PPO', 'SAC'],
                'last_prediction': datetime.now().isoformat(),
                'inference_time_ms': 2.5
            }
        }
    
    async def _detect_anomalies(self):
        """Détecte les anomalies dans les métriques"""
        while self.running:
            try:
                if self.anomaly_detector and len(self.metrics_history['system']) >= 100:
                    # Préparer les données pour la détection
                    recent_metrics = list(self.metrics_history['system'])[-100:]
                    
                    features = np.array([
                        [m.cpu_percent, m.memory_percent, m.disk_usage_percent,
                         m.network_recv_mb, m.network_sent_mb]
                        for m in recent_metrics
                    ])
                    
                    # Normaliser les features
                    features_scaled = self.scaler.fit_transform(features)
                    
                    # Détecter les anomalies
                    predictions = self.anomaly_detector.fit_predict(features_scaled)
                    anomaly_scores = self.anomaly_detector.score_samples(features_scaled)
                    
                    # Vérifier la dernière valeur
                    if predictions[-1] == -1:  # Anomalie détectée
                        latest_metric = recent_metrics[-1]
                        anomaly_score = -anomaly_scores[-1]  # Plus élevé = plus anormal
                        
                        await self.alert_manager.send_alert(
                            severity=AlertSeverity.WARNING,
                            title="System Anomaly Detected",
                            message=f"Anomaly score: {anomaly_score:.2f}",
                            metadata={
                                'cpu': latest_metric.cpu_percent,
                                'memory': latest_metric.memory_percent,
                                'score': anomaly_score
                            }
                        )
                        
                        logger.warning(f"Anomaly detected with score {anomaly_score:.2f}")
                    
                    # Stocker les scores
                    self.anomaly_scores['system'] = anomaly_scores.tolist()
                
            except Exception as e:
                logger.error(f"Error in anomaly detection: {str(e)}")
            
            await asyncio.sleep(60)  # Détection toutes les minutes
    
    async def _check_system_thresholds(self, metrics: SystemMetrics):
        """Vérifie les seuils système et alerte si nécessaire"""
        thresholds = self.config.get('monitoring', {}).get('thresholds', {})
        
        # CPU
        cpu_threshold = thresholds.get('cpu_percent', 80)
        if metrics.cpu_percent > cpu_threshold:
            await self.alert_manager.send_alert(
                severity=AlertSeverity.WARNING,
                title="High CPU Usage",
                message=f"CPU usage at {metrics.cpu_percent:.1f}% (threshold: {cpu_threshold}%)",
                metadata={'cpu_percent': metrics.cpu_percent}
            )
        
        # Memory
        memory_threshold = thresholds.get('memory_percent', 85)
        if metrics.memory_percent > memory_threshold:
            await self.alert_manager.send_alert(
                severity=AlertSeverity.WARNING,
                title="High Memory Usage",
                message=f"Memory usage at {metrics.memory_percent:.1f}% (threshold: {memory_threshold}%)",
                metadata={'memory_percent': metrics.memory_percent}
            )
        
        # Disk
        disk_threshold = thresholds.get('disk_percent', 90)
        if metrics.disk_usage_percent > disk_threshold:
            await self.alert_manager.send_alert(
                severity=AlertSeverity.ERROR,
                title="High Disk Usage",
                message=f"Disk usage at {metrics.disk_usage_percent:.1f}% (threshold: {disk_threshold}%)",
                metadata={'disk_percent': metrics.disk_usage_percent}
            )
    
    async def _check_trading_alerts(self, metrics: TradingMetrics):
        """Vérifie les alertes de trading"""
        thresholds = self.config.get('monitoring', {}).get('trading_thresholds', {})
        
        # Drawdown
        max_drawdown = thresholds.get('max_drawdown', 0.20)
        if metrics.current_drawdown > max_drawdown:
            await self.alert_manager.send_alert(
                severity=AlertSeverity.CRITICAL,
                title="Maximum Drawdown Exceeded",
                message=f"Current drawdown: {metrics.current_drawdown:.2%} (max: {max_drawdown:.2%})",
                metadata={'drawdown': metrics.current_drawdown}
            )
        
        # Error rate
        max_error_rate = thresholds.get('max_error_rate', 0.05)
        if metrics.error_rate > max_error_rate:
            await self.alert_manager.send_alert(
                severity=AlertSeverity.ERROR,
                title="High Error Rate",
                message=f"Error rate: {metrics.error_rate:.2%} (max: {max_error_rate:.2%})",
                metadata={'error_rate': metrics.error_rate}
            )
        
        # Latency
        max_latency = thresholds.get('max_order_latency_ms', 100)
        if metrics.p99_order_latency_ms > max_latency:
            await self.alert_manager.send_alert(
                severity=AlertSeverity.WARNING,
                title="High Order Latency",
                message=f"P99 latency: {metrics.p99_order_latency_ms:.1f}ms (max: {max_latency}ms)",
                metadata={'latency_ms': metrics.p99_order_latency_ms}
            )
    
    async def _write_to_influx(self, measurement: str, metrics: Union[SystemMetrics, TradingMetrics]):
        """Écrit les métriques dans InfluxDB"""
        if not self.influx_client:
            return
        
        try:
            point = Point(measurement).time(metrics.timestamp)
            
            # Ajouter tous les champs
            for field, value in metrics.__dict__.items():
                if field != 'timestamp' and not isinstance(value, (list, dict, tuple)):
                    point.field(field, value)
            
            self.influx_write_api.write(bucket=self.influx_bucket, record=point)
            
        except Exception as e:
            logger.error(f"Error writing to InfluxDB: {str(e)}")
    
    def _write_to_redis(self, key: str, metrics: Union[SystemMetrics, TradingMetrics]):
        """Écrit les métriques dans Redis"""
        if not self.redis_client:
            return
        
        try:
            # Convertir en dict sérialisable
            data = {
                k: v for k, v in metrics.__dict__.items()
                if not isinstance(v, (list, dict, tuple)) or k in ['active_strategies', 'connected_exchanges']
            }
            
            # Convertir datetime en string
            if 'timestamp' in data:
                data['timestamp'] = data['timestamp'].isoformat()
            
            # Stocker dans Redis avec TTL
            self.redis_client.setex(
                key,
                300,  # TTL de 5 minutes
                json.dumps(data)
            )
            
            # Ajouter à la liste historique
            history_key = f"{key}:history"
            self.redis_client.lpush(history_key, json.dumps(data))
            self.redis_client.ltrim(history_key, 0, self.history_size - 1)
            
        except Exception as e:
            logger.error(f"Error writing to Redis: {str(e)}")
    
    async def _start_websocket_server(self):
        """Démarre le serveur WebSocket pour le dashboard temps réel"""
        # Routes WebSocket
        @self.sio.event
        async def connect(sid, environ):
            logger.info(f"WebSocket client connected: {sid}")
            await self.sio.emit('welcome', {'message': 'Connected to monitoring server'}, to=sid)
        
        @self.sio.event
        async def disconnect(sid):
            logger.info(f"WebSocket client disconnected: {sid}")
        
        @self.sio.event
        async def subscribe_metrics(sid, data):
            room = data.get('room', 'metrics')
            await self.sio.enter_room(sid, room)
            await self.sio.emit('subscribed', {'room': room}, to=sid)
        
        # Route HTTP pour servir le dashboard
        async def dashboard_handler(request):
            return web.Response(text=self._generate_dashboard_html(), content_type='text/html')
        
        self.app.router.add_get('/', dashboard_handler)
        
        # Démarrer le serveur
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        ws_port = self.config.get('monitoring', {}).get('websocket_port', 8080)
        site = web.TCPSite(runner, 'localhost', ws_port)
        await site.start()
        
        logger.info(f"WebSocket server started on port {ws_port}")
    
    async def _broadcast_metrics(self):
        """Diffuse les métriques aux clients WebSocket"""
        while self.running:
            try:
                # Préparer les données à diffuser
                data = {
                    'timestamp': datetime.now().isoformat(),
                    'system': None,
                    'trading': None,
                    'health': self.health_status
                }
                
                # Dernières métriques système
                if self.metrics_history['system']:
                    latest_system = self.metrics_history['system'][-1]
                    data['system'] = {
                        'cpu': latest_system.cpu_percent,
                        'memory': latest_system.memory_percent,
                        'disk': latest_system.disk_usage_percent,
                        'network_recv': latest_system.network_recv_mb,
                        'network_sent': latest_system.network_sent_mb
                    }
                
                # Dernières métriques trading
                if self.metrics_history['trading']:
                    latest_trading = self.metrics_history['trading'][-1]
                    data['trading'] = {
                        'positions': latest_trading.active_positions,
                        'pnl': latest_trading.total_pnl,
                        'orders_per_sec': latest_trading.orders_per_second,
                        'error_rate': latest_trading.error_rate
                    }
                
                # Diffuser aux clients connectés
                await self.sio.emit('metrics_update', data, room='metrics')
                
            except Exception as e:
                logger.error(f"Error broadcasting metrics: {str(e)}")
            
            await asyncio.sleep(1)  # Broadcast toutes les secondes
    
    def _generate_dashboard_html(self) -> str:
        """Génère le HTML du dashboard temps réel"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>AI Trading Robot - System Monitor</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #e0e0e0;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .metric-label {
            color: #888;
            font-size: 0.9rem;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #4CAF50;
        }
        .chart-container {
            background: #2a2a2a;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            height: 400px;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-healthy { background-color: #4CAF50; }
        .status-degraded { background-color: #FF9800; }
        .status-unhealthy { background-color: #F44336; }
        .components-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }
        .component-card {
            background: #333;
            border-radius: 8px;
            padding: 15px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>AI Trading Robot System Monitor</h1>
            <p>Real-time system and trading metrics</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">CPU Usage</div>
                <div class="metric-value" id="cpu-value">-</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Memory Usage</div>
                <div class="metric-value" id="memory-value">-</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Active Positions</div>
                <div class="metric-value" id="positions-value">-</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Total P&L</div>
                <div class="metric-value" id="pnl-value">-</div>
            </div>
        </div>
        
        <div class="chart-container">
            <div id="performance-chart"></div>
        </div>
        
        <h2>Component Health</h2>
        <div class="components-grid" id="components-grid">
            <!-- Components will be inserted here -->
        </div>
    </div>
    
    <script>
        const socket = io();
        
        // Subscribe to metrics updates
        socket.on('connect', () => {
            console.log('Connected to monitoring server');
            socket.emit('subscribe_metrics', {room: 'metrics'});
        });
        
        // Chart data
        const chartData = {
            cpu: [],
            memory: [],
            timestamps: []
        };
        
        // Initialize chart
        const layout = {
            title: 'System Performance',
            plot_bgcolor: '#2a2a2a',
            paper_bgcolor: '#2a2a2a',
            font: { color: '#e0e0e0' },
            xaxis: { title: 'Time' },
            yaxis: { title: 'Usage %' }
        };
        
        Plotly.newPlot('performance-chart', [
            { x: [], y: [], name: 'CPU', type: 'scatter', mode: 'lines' },
            { x: [], y: [], name: 'Memory', type: 'scatter', mode: 'lines' }
        ], layout);
        
        // Handle metrics updates
        socket.on('metrics_update', (data) => {
            // Update metric values
            if (data.system) {
                document.getElementById('cpu-value').textContent = data.system.cpu.toFixed(1) + '%';
                document.getElementById('memory-value').textContent = data.system.memory.toFixed(1) + '%';
                
                // Update chart
                chartData.timestamps.push(new Date());
                chartData.cpu.push(data.system.cpu);
                chartData.memory.push(data.system.memory);
                
                // Keep only last 100 points
                if (chartData.timestamps.length > 100) {
                    chartData.timestamps.shift();
                    chartData.cpu.shift();
                    chartData.memory.shift();
                }
                
                Plotly.update('performance-chart', {
                    x: [chartData.timestamps, chartData.timestamps],
                    y: [chartData.cpu, chartData.memory]
                });
            }
            
            if (data.trading) {
                document.getElementById('positions-value').textContent = data.trading.positions;
                document.getElementById('pnl-value').textContent = '$' + data.trading.pnl.toFixed(2);
            }
            
            // Update component health
            updateComponentHealth(data.health);
        });
        
        function updateComponentHealth(health) {
            const grid = document.getElementById('components-grid');
            grid.innerHTML = '';
            
            for (const [id, status] of Object.entries(health)) {
                const card = document.createElement('div');
                card.className = 'component-card';
                
                const statusClass = 'status-' + status.status;
                card.innerHTML = `
                    <div>
                        <span class="status-indicator ${statusClass}"></span>
                        <span>${status.component}</span>
                    </div>
                    <div style="color: #888; font-size: 0.8rem;">
                        ${status.latency_ms.toFixed(1)}ms
                    </div>
                `;
                
                grid.appendChild(card);
            }
        }
    </script>
</body>
</html>
"""
    
    def get_system_status(self) -> Dict[str, Any]:
        """Retourne l'état actuel du système"""
        return {
            'health_status': self.health_status,
            'latest_system_metrics': self.metrics_history['system'][-1] if self.metrics_history['system'] else None,
            'latest_trading_metrics': self.metrics_history['trading'][-1] if self.metrics_history['trading'] else None,
            'anomaly_scores': dict(self.anomaly_scores),
            'components': {k: v['name'] for k, v in self.components.items()}
        }
    
    async def run_diagnostic(self) -> Dict[str, Any]:
        """Exécute un diagnostic complet du système"""
        logger.info("Running system diagnostic")
        
        diagnostic = {
            'timestamp': datetime.now().isoformat(),
            'system': {},
            'components': {},
            'recommendations': []
        }
        
        # Diagnostic système
        system_metrics = self._get_system_metrics()
        diagnostic['system'] = {
            'cpu': system_metrics.cpu_percent,
            'memory': system_metrics.memory_percent,
            'disk': system_metrics.disk_usage_percent,
            'uptime_hours': system_metrics.uptime_seconds / 3600
        }
        
        # Diagnostic des composants
        for component_id, component in self.components.items():
            result = await self._check_component(component_id, component)
            diagnostic['components'][component_id] = {
                'status': result.status.value,
                'message': result.message,
                'healthy': result.is_healthy
            }
        
        # Recommandations
        if system_metrics.cpu_percent > 80:
            diagnostic['recommendations'].append("Consider scaling up CPU resources")
        
        if system_metrics.memory_percent > 85:
            diagnostic['recommendations'].append("Memory usage high, consider optimization")
        
        if system_metrics.disk_usage_percent > 90:
            diagnostic['recommendations'].append("Disk space critical, cleanup required")
        
        unhealthy_components = [
            c for c, r in diagnostic['components'].items() 
            if not r['healthy']
        ]
        
        if unhealthy_components:
            diagnostic['recommendations'].append(
                f"Components need attention: {', '.join(unhealthy_components)}"
            )
        
        return diagnostic


class MonitoringDashboard:
    """Dashboard de monitoring interactif"""
    
    def __init__(self, monitor: SystemMonitor):
        import dash
        from dash import html, dcc, dash_table  # Import Dash components
        from dash.dependencies import Output, Input  # Import Output and Input for callbacks

        self.Output = Output
        self.Input = Input
        self.html = html
        self.dcc = dcc
        self.dash_table = dash_table
    def _setup_layout(self):
        """Configure le layout du dashboard"""
        html = self.html
        dcc = self.dcc
        self.app.layout = html.Div([
            html.H1("AI Trading Robot Monitoring Dashboard"),
            
            # Métriques en temps réel
            html.Div(id='live-metrics', children=[]),
            
            # Graphiques
            dcc.Graph(id='system-performance'),
            dcc.Graph(id='trading-performance'),
            
            # Tableau de santé des composants
            html.Div(id='component-health'),
            
            # Intervalle de mise à jour
            dcc.Interval(id='interval-component', interval=5000)  # 5 secondes
        ])
    
    def _setup_callbacks(self):
        """Configure les callbacks du dashboard"""
        @self.app.callback(
            [self.Output('live-metrics', 'children'),
             self.Output('system-performance', 'figure'),
             self.Output('trading-performance', 'figure'),
             self.Output('component-health', 'children')],
            [self.Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n_intervals):
            # Obtenir l'état actuel
            status = self.monitor.get_system_status()
            
            # Métriques en direct
            metrics_cards = self._create_metric_cards(status)
            
            # Graphique système
            system_fig = self._create_system_figure(status)
            
            # Graphique trading
            trading_fig = self._create_trading_figure(status)
            
            # Tableau de santé des composants
            health_table = self._create_health_table(status)
            
            return metrics_cards, system_fig, trading_fig, health_table
    
    def _create_metric_cards(self, status: Dict[str, Any]) -> Any:
        """Crée les cartes de métriques"""
        # À implémenter
        return self.html.Div("Metrics cards")
    
    def _create_system_figure(self, status: Dict[str, Any]) -> go.Figure:
        """Crée le graphique de performance système"""
    def _create_health_table(self, status: Dict[str, Any]):
        """Crée le tableau de santé des composants"""
        html = self.html
        # À implémenter
        return html.Div("Health table")
        """Crée le graphique de performance trading"""
        # À implémenter
        return go.Figure()
    
    def _create_health_table(self, status: Dict[str, Any]) -> Any:
        """Crée le tableau de santé des composants"""
        # À implémenter
        return self.html.Div("Health table")
    
    def run(self, debug: bool = False, port: int = 8050):
        """Lance le dashboard"""
        self.app.run_server(debug=debug, port=port)


# Fonction principale pour démarrer le monitoring
async def start_monitoring():
    """Démarre le système de monitoring"""
    monitor = SystemMonitor()
    
    try:
        await monitor.start()
    except KeyboardInterrupt:
        logger.info("Monitoring interrupted by user")
    finally:
        await monitor.stop()


if __name__ == "__main__":
    # Démarrer le monitoring
    asyncio.run(start_monitoring())