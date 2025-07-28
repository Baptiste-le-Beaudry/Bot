"""
Monitoring and Observability Package
====================================

Système complet de surveillance, logging, alertes et reporting
pour le robot de trading algorithmique.

Composants:
    - PerformanceTracker: Suivi des performances de trading
    - SystemMonitor: Surveillance système et infrastructure
    - AlertManager: Gestion des alertes multi-canaux
    - ReportGenerator: Génération de rapports automatiques

Usage:
    from monitoring import PerformanceTracker, SystemMonitor, AlertManager
    
    tracker = PerformanceTracker(config)
    metrics = await tracker.get_performance_metrics()
"""

from monitoring.performance_tracker import (
    PerformanceTracker,
    PerformanceMetrics,
    TradingMetrics,
    StrategyPerformance,
    PortfolioAnalytics,
    RiskMetrics,
    DrawdownAnalysis
)

from monitoring.system_monitor import (
    SystemMonitor,
    SystemMetrics,
    ComponentHealth,
    ResourceUsage,
    LatencyTracker,
    ServiceStatus,
    HealthCheck
)

from monitoring.alerts import (
    AlertManager,
    Alert,
    AlertLevel,
    AlertChannel,
    AlertRule,
    AlertHistory,
    NotificationService
)

from monitoring.reporting import (
    ReportGenerator,
    ReportType,
    ReportSchedule,
    ReportTemplate,
    DataAggregator,
    Visualization,
    ExportFormat
)

# Version
__version__ = "1.0.0"

# Exports publics
__all__ = [
    # Performance Tracker
    "PerformanceTracker",
    "PerformanceMetrics",
    "TradingMetrics",
    "StrategyPerformance",
    "PortfolioAnalytics",
    "RiskMetrics",
    "DrawdownAnalysis",
    
    # System Monitor
    "SystemMonitor",
    "SystemMetrics",
    "ComponentHealth",
    "ResourceUsage",
    "LatencyTracker",
    "ServiceStatus",
    "HealthCheck",
    
    # Alerts
    "AlertManager",
    "Alert",
    "AlertLevel",
    "AlertChannel",
    "AlertRule",
    "AlertHistory",
    "NotificationService",
    
    # Reporting
    "ReportGenerator",
    "ReportType",
    "ReportSchedule",
    "ReportTemplate",
    "DataAggregator",
    "Visualization",
    "ExportFormat"
]

# Configuration par défaut
DEFAULT_MONITORING_CONFIG = {
    "performance": {
        "metrics_interval": 60,  # seconds
        "calculation_window": 3600,  # 1 hour
        "benchmark": "BTCUSDT",
        "risk_free_rate": 0.02,
        "confidence_level": 0.95,
        "metrics_retention": "90 days"
    },
    "system": {
        "health_check_interval": 30,  # seconds
        "resource_check_interval": 10,
        "latency_buckets": [1, 5, 10, 50, 100, 500, 1000],  # ms
        "cpu_threshold": 80,  # percent
        "memory_threshold": 85,
        "disk_threshold": 90
    },
    "alerts": {
        "channels": {
            "slack": {
                "enabled": True,
                "webhook_url": None,
                "channel": "#trading-alerts",
                "rate_limit": 10  # per minute
            },
            "email": {
                "enabled": False,
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "from_address": None,
                "to_addresses": []
            },
            "telegram": {
                "enabled": False,
                "bot_token": None,
                "chat_id": None
            },
            "pagerduty": {
                "enabled": False,
                "integration_key": None
            }
        },
        "rules": {
            "system_critical": {
                "level": "critical",
                "channels": ["slack", "pagerduty"],
                "throttle": 300  # seconds
            },
            "trading_error": {
                "level": "error",
                "channels": ["slack", "email"],
                "throttle": 60
            },
            "performance_warning": {
                "level": "warning",
                "channels": ["slack"],
                "throttle": 1800
            }
        }
    },
    "reporting": {
        "schedules": {
            "daily": {
                "enabled": True,
                "time": "08:00",
                "timezone": "UTC",
                "recipients": [],
                "format": "pdf"
            },
            "weekly": {
                "enabled": True,
                "day": "monday",
                "time": "09:00",
                "timezone": "UTC",
                "recipients": [],
                "format": "html"
            }
        },
        "templates": {
            "performance": "templates/performance_report.html",
            "risk": "templates/risk_report.html",
            "system": "templates/system_report.html"
        }
    }
}

# Métriques standard à tracker
STANDARD_METRICS = {
    "trading": [
        "total_pnl",
        "realized_pnl",
        "unrealized_pnl",
        "win_rate",
        "profit_factor",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "trades_count",
        "avg_trade_duration"
    ],
    "risk": [
        "var_95",
        "var_99",
        "expected_shortfall",
        "beta",
        "correlation",
        "tracking_error",
        "information_ratio",
        "downside_deviation"
    ],
    "system": [
        "uptime",
        "cpu_usage",
        "memory_usage",
        "disk_usage",
        "network_latency",
        "order_latency",
        "error_rate",
        "throughput"
    ]
}

# Types de visualisations
class VisualizationType:
    """Types de visualisations disponibles"""
    LINE_CHART = "line"
    BAR_CHART = "bar"
    CANDLESTICK = "candlestick"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    PIE_CHART = "pie"
    GAUGE = "gauge"
    TABLE = "table"


# Gestionnaire de monitoring unifié
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
import asyncio


class MonitoringManager:
    """
    Gestionnaire unifié pour toutes les opérations de monitoring.
    Coordonne trackers, monitors, alertes et reporting.
    """
    
    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_MONITORING_CONFIG, **(config or {})}
        
        # Initialiser les composants
        self.performance_tracker = PerformanceTracker(self.config['performance'])
        self.system_monitor = SystemMonitor(self.config['system'])
        self.alert_manager = AlertManager(self.config['alerts'])
        self.report_generator = ReportGenerator(self.config['reporting'])
        
        self._running = False
        self._tasks = []
        
    async def start(self):
        """Démarre tous les composants de monitoring"""
        await self.performance_tracker.start()
        await self.system_monitor.start()
        await self.alert_manager.start()
        
        # Démarrer les tâches périodiques
        self._tasks = [
            asyncio.create_task(self._performance_loop()),
            asyncio.create_task(self._system_loop()),
            asyncio.create_task(self._reporting_loop())
        ]
        
        self._running = True
        
    async def stop(self):
        """Arrête tous les composants"""
        self._running = False
        
        # Annuler les tâches
        for task in self._tasks:
            task.cancel()
            
        await self.performance_tracker.stop()
        await self.system_monitor.stop()
        await self.alert_manager.stop()
        
    async def _performance_loop(self):
        """Boucle de collecte des métriques de performance"""
        interval = self.config['performance']['metrics_interval']
        
        while self._running:
            try:
                metrics = await self.performance_tracker.collect_metrics()
                
                # Vérifier les seuils
                await self._check_performance_thresholds(metrics)
                
            except Exception as e:
                await self.alert_manager.send_alert(
                    Alert(
                        level=AlertLevel.ERROR,
                        source="MonitoringManager",
                        message=f"Performance collection error: {str(e)}"
                    )
                )
                
            await asyncio.sleep(interval)
            
    async def _system_loop(self):
        """Boucle de surveillance système"""
        interval = self.config['system']['resource_check_interval']
        
        while self._running:
            try:
                health = await self.system_monitor.check_health()
                
                # Alerter si problèmes
                if not health.is_healthy:
                    await self.alert_manager.send_alert(
                        Alert(
                            level=AlertLevel.WARNING,
                            source="SystemMonitor",
                            message=f"System health degraded: {health.issues}"
                        )
                    )
                    
            except Exception as e:
                # Log mais ne pas alerter pour éviter les boucles
                import logging
                logging.error(f"System monitoring error: {e}")
                
            await asyncio.sleep(interval)
            
    async def _reporting_loop(self):
        """Boucle de génération de rapports"""
        while self._running:
            try:
                # Vérifier les rapports programmés
                now = datetime.now(timezone.utc)
                
                for schedule_name, schedule in self.config['reporting']['schedules'].items():
                    if schedule.get('enabled') and self._should_generate_report(schedule, now):
                        await self._generate_scheduled_report(schedule_name, schedule)
                        
            except Exception as e:
                await self.alert_manager.send_alert(
                    Alert(
                        level=AlertLevel.ERROR,
                        source="ReportGenerator",
                        message=f"Report generation error: {str(e)}"
                    )
                )
                
            # Vérifier toutes les minutes
            await asyncio.sleep(60)
            
    async def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Vérifie les seuils de performance"""
        # Drawdown excessif
        if metrics.max_drawdown > 0.2:  # 20%
            await self.alert_manager.send_alert(
                Alert(
                    level=AlertLevel.CRITICAL,
                    source="PerformanceTracker",
                    message=f"Critical drawdown: {metrics.max_drawdown:.1%}",
                    details={"metrics": metrics.to_dict()}
                )
            )
            
        # Sharpe ratio faible
        elif metrics.sharpe_ratio < 0.5:
            await self.alert_manager.send_alert(
                Alert(
                    level=AlertLevel.WARNING,
                    source="PerformanceTracker",
                    message=f"Low Sharpe ratio: {metrics.sharpe_ratio:.2f}"
                )
            )
            
    def _should_generate_report(self, schedule: dict, now: datetime) -> bool:
        """Détermine si un rapport doit être généré"""
        # Logique simplifiée - à améliorer
        if schedule.get('frequency') == 'daily':
            return now.hour == int(schedule.get('time', '08:00').split(':')[0])
        return False
        
    async def _generate_scheduled_report(self, name: str, schedule: dict):
        """Génère un rapport programmé"""
        report = await self.report_generator.generate(
            report_type=name,
            format=schedule.get('format', 'pdf')
        )
        
        # Envoyer aux destinataires
        for recipient in schedule.get('recipients', []):
            # TODO: Implémenter l'envoi
            pass
            
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Retourne les données pour le dashboard"""
        return {
            "performance": self.performance_tracker.get_current_metrics(),
            "system": self.system_monitor.get_current_status(),
            "alerts": self.alert_manager.get_recent_alerts(100),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


# Instance globale
_monitoring_manager = None


def get_monitoring_manager(config: dict = None) -> MonitoringManager:
    """Obtient l'instance globale du monitoring"""
    global _monitoring_manager
    if _monitoring_manager is None:
        _monitoring_manager = MonitoringManager(config)
    return _monitoring_manager


# Décorateurs pour monitoring automatique
def monitor_performance(metric_name: str = None):
    """Décorateur pour monitorer automatiquement les performances d'une fonction"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            manager = get_monitoring_manager()
            start_time = asyncio.get_event_loop().time()
            
            try:
                result = await func(*args, **kwargs)
                
                # Enregistrer le succès
                duration = asyncio.get_event_loop().time() - start_time
                await manager.performance_tracker.record_metric(
                    name=metric_name or func.__name__,
                    value=duration,
                    tags={"status": "success"}
                )
                
                return result
                
            except Exception as e:
                # Enregistrer l'échec
                await manager.performance_tracker.record_metric(
                    name=metric_name or func.__name__,
                    value=1,
                    tags={"status": "error", "error": str(e)}
                )
                raise
                
        return wrapper
    return decorator


# Métriques Prometheus
try:
    from prometheus_client import Counter, Gauge, Histogram, Summary
    
    # Métriques globales
    trades_total = Counter('trades_total', 'Total number of trades')
    active_positions = Gauge('active_positions', 'Number of active positions')
    portfolio_value = Gauge('portfolio_value', 'Current portfolio value')
    trade_latency = Histogram('trade_latency_seconds', 'Trade execution latency')
    
except ImportError:
    # Prometheus non installé
    pass

# Initialisation
import logging
logger = logging.getLogger(__name__)
logger.info(f"Monitoring package initialized - Version {__version__}")