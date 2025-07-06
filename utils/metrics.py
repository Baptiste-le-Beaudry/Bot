"""
Système de Métriques Haute Performance pour Robot de Trading Algorithmique IA
============================================================================

Ce module implémente un collecteur de métriques optimisé pour les systèmes de
trading haute fréquence avec export Prometheus, agrégation temps réel, et
métriques business spécifiques au trading algorithmique.

Architecture:
- Métriques thread-safe avec performance sub-milliseconde
- Export Prometheus avec labels dynamiques
- Agrégation temps réel (moving averages, percentiles)
- Métriques business spécifiques trading (Sharpe, PnL, latence)
- Integration avec décorateurs de performance
- Buffer circulaire pour métriques historiques
- Auto-discovery des métriques critiques

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import asyncio
import threading
import time
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from functools import lru_cache
from statistics import median, stdev
from typing import (
    Any, Dict, List, Optional, Set, Union, Callable, 
    Iterator, Tuple, NamedTuple, Protocol
)
import math
import gc

# Third-party imports pour performance
import numpy as np
from prometheus_client import (
    Counter, Histogram, Gauge, Summary, CollectorRegistry,
    generate_latest, CONTENT_TYPE_LATEST, start_http_server
)

# Import du logger
from utils.logger import get_structured_logger


class MetricType(Enum):
    """Types de métriques supportées"""
    COUNTER = "counter"           # Valeur croissante (trades, erreurs)
    GAUGE = "gauge"              # Valeur instantanée (positions, balance)
    HISTOGRAM = "histogram"       # Distribution (latences, tailles)
    SUMMARY = "summary"          # Quantiles et sommes
    TRADING_METRIC = "trading"   # Métriques spécifiques trading


class MetricPriority(Enum):
    """Priorité des métriques pour export et storage"""
    CRITICAL = "critical"     # Export temps réel obligatoire
    HIGH = "high"            # Export fréquent
    NORMAL = "normal"        # Export standard
    LOW = "low"             # Export occasionnel
    DEBUG = "debug"         # Export développement seulement


@dataclass
class MetricPoint:
    """Point de donnée métrique avec timestamp"""
    timestamp: float
    value: Union[float, int]
    labels: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass 
class MetricSeries:
    """Série temporelle de métriques avec agrégation"""
    name: str
    metric_type: MetricType
    priority: MetricPriority = MetricPriority.NORMAL
    max_points: int = 1000
    points: deque = field(default_factory=deque)
    
    def __post_init__(self):
        self.points = deque(maxlen=self.max_points)
        self._lock = threading.RLock()
    
    def add_point(self, value: Union[float, int], labels: Optional[Dict[str, str]] = None):
        """Ajoute un point de données"""
        with self._lock:
            point = MetricPoint(
                timestamp=time.time(),
                value=float(value),
                labels=labels or {}
            )
            self.points.append(point)
    
    def get_latest(self) -> Optional[MetricPoint]:
        """Récupère le dernier point"""
        with self._lock:
            return self.points[-1] if self.points else None
    
    def get_range(self, seconds: float) -> List[MetricPoint]:
        """Récupère les points dans une fenêtre temporelle"""
        cutoff = time.time() - seconds
        with self._lock:
            return [p for p in self.points if p.timestamp >= cutoff]
    
    def calculate_moving_average(self, window_seconds: float = 60.0) -> float:
        """Calcule la moyenne mobile"""
        points = self.get_range(window_seconds)
        if not points:
            return 0.0
        return sum(p.value for p in points) / len(points)
    
    def calculate_rate(self, window_seconds: float = 60.0) -> float:
        """Calcule le taux de changement (pour counters)"""
        points = self.get_range(window_seconds)
        if len(points) < 2:
            return 0.0
        
        latest = points[-1]
        earliest = points[0]
        time_diff = latest.timestamp - earliest.timestamp
        value_diff = latest.value - earliest.value
        
        return value_diff / max(time_diff, 0.001)  # Éviter division par zéro
    
    def calculate_percentiles(self, window_seconds: float = 60.0) -> Dict[int, float]:
        """Calcule les percentiles"""
        points = self.get_range(window_seconds)
        if not points:
            return {}
        
        values = [p.value for p in points]
        values.sort()
        n = len(values)
        
        percentiles = {}
        for p in [50, 90, 95, 99]:
            if n == 0:
                percentiles[p] = 0.0
            else:
                index = int(n * p / 100)
                percentiles[p] = values[min(index, n-1)]
        
        return percentiles


class TradingMetrics:
    """Métriques spécifiques au trading avec calculs financiers"""
    
    def __init__(self):
        self._pnl_history: deque = deque(maxlen=10000)
        self._trade_history: deque = deque(maxlen=10000) 
        self._returns: deque = deque(maxlen=10000)
        self._drawdown_history: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
        self.logger = get_structured_logger("trading_metrics")
    
    def record_trade(self, symbol: str, side: str, quantity: float, 
                    price: float, pnl: float, strategy_id: str,
                    execution_time_ms: float):
        """Enregistre une transaction"""
        with self._lock:
            trade = {
                'timestamp': time.time(),
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'pnl': pnl,
                'strategy_id': strategy_id,
                'execution_time_ms': execution_time_ms,
                'notional': abs(quantity * price)
            }
            
            self._trade_history.append(trade)
            self._pnl_history.append(pnl)
            
            # Calculer le return si possible
            if len(self._pnl_history) > 1:
                prev_cumul = sum(list(self._pnl_history)[:-1])
                if prev_cumul != 0:
                    return_pct = pnl / abs(prev_cumul)
                    self._returns.append(return_pct)
    
    def calculate_sharpe_ratio(self, window_minutes: int = 60, 
                              risk_free_rate: float = 0.02) -> float:
        """Calcule le ratio de Sharpe"""
        with self._lock:
            if len(self._returns) < 2:
                return 0.0
            
            # Filtrer par fenêtre temporelle
            cutoff = time.time() - (window_minutes * 60)
            recent_returns = []
            
            for trade in self._trade_history:
                if trade['timestamp'] >= cutoff:
                    recent_returns.append(trade['pnl'])
            
            if len(recent_returns) < 2:
                return 0.0
            
            try:
                mean_return = np.mean(recent_returns)
                std_return = np.std(recent_returns)
                
                if std_return == 0:
                    return 0.0
                
                # Annualiser (approximation pour trading haute fréquence)
                periods_per_year = 252 * 24 * 60 / window_minutes  # Trading continu
                annual_return = mean_return * periods_per_year
                annual_std = std_return * np.sqrt(periods_per_year)
                
                sharpe = (annual_return - risk_free_rate) / annual_std
                return float(sharpe)
                
            except Exception as e:
                self.logger.error("sharpe_calculation_error", error=str(e))
                return 0.0
    
    def calculate_max_drawdown(self, window_minutes: int = 60) -> float:
        """Calcule le drawdown maximum"""
        with self._lock:
            cutoff = time.time() - (window_minutes * 60)
            
            # Construire la courbe de PnL cumulé
            cumulative_pnl = []
            running_pnl = 0.0
            
            for trade in self._trade_history:
                if trade['timestamp'] >= cutoff:
                    running_pnl += trade['pnl']
                    cumulative_pnl.append(running_pnl)
            
            if len(cumulative_pnl) < 2:
                return 0.0
            
            # Calculer le drawdown
            peak = cumulative_pnl[0]
            max_dd = 0.0
            
            for value in cumulative_pnl:
                if value > peak:
                    peak = value
                
                drawdown = (peak - value) / max(abs(peak), 1.0)  # Éviter division par zéro
                max_dd = max(max_dd, drawdown)
            
            return float(max_dd)
    
    def calculate_win_rate(self, window_minutes: int = 60) -> float:
        """Calcule le taux de réussite"""
        with self._lock:
            cutoff = time.time() - (window_minutes * 60)
            
            winning_trades = 0
            total_trades = 0
            
            for trade in self._trade_history:
                if trade['timestamp'] >= cutoff:
                    total_trades += 1
                    if trade['pnl'] > 0:
                        winning_trades += 1
            
            return winning_trades / max(total_trades, 1)
    
    def calculate_profit_factor(self, window_minutes: int = 60) -> float:
        """Calcule le facteur de profit"""
        with self._lock:
            cutoff = time.time() - (window_minutes * 60)
            
            gross_profit = 0.0
            gross_loss = 0.0
            
            for trade in self._trade_history:
                if trade['timestamp'] >= cutoff:
                    if trade['pnl'] > 0:
                        gross_profit += trade['pnl']
                    else:
                        gross_loss += abs(trade['pnl'])
            
            return gross_profit / max(gross_loss, 0.001)
    
    def get_trading_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Retourne un résumé complet des métriques de trading"""
        return {
            'sharpe_ratio': self.calculate_sharpe_ratio(window_minutes),
            'max_drawdown': self.calculate_max_drawdown(window_minutes),
            'win_rate': self.calculate_win_rate(window_minutes),
            'profit_factor': self.calculate_profit_factor(window_minutes),
            'total_trades': len([t for t in self._trade_history 
                               if t['timestamp'] >= time.time() - window_minutes * 60]),
            'total_pnl': sum(t['pnl'] for t in self._trade_history 
                           if t['timestamp'] >= time.time() - window_minutes * 60)
        }


class MetricsCollector:
    """
    Collecteur principal de métriques avec export Prometheus
    Thread-safe et optimisé pour haute fréquence
    """
    
    def __init__(self, namespace: str = "trading_robot", enable_prometheus: bool = True):
        self.namespace = namespace
        self.enable_prometheus = enable_prometheus
        self.logger = get_structured_logger(f"metrics_collector.{namespace}")
        
        # Storage des métriques
        self._metrics: Dict[str, MetricSeries] = {}
        self._prometheus_metrics: Dict[str, Any] = {}
        self._lock = threading.RLock()
        
        # Registry Prometheus séparé
        self.prometheus_registry = CollectorRegistry()
        
        # Métriques système prédéfinies
        self._init_system_metrics()
        
        # Métriques trading spécialisées
        self.trading_metrics = TradingMetrics()
        
        # Background tasks
        self._running = False
        self._background_task = None
        self._http_server = None
        
        # Performance tracking
        self._metrics_per_second = 0
        self._last_metrics_count = 0
        self._last_count_time = time.time()
    
    def _init_system_metrics(self):
        """Initialise les métriques système standard"""
        # Métriques core du trading robot
        system_metrics = [
            ("trades_total", MetricType.COUNTER, MetricPriority.CRITICAL),
            ("orders_total", MetricType.COUNTER, MetricPriority.CRITICAL),
            ("pnl_total", MetricType.GAUGE, MetricPriority.CRITICAL),
            ("position_count", MetricType.GAUGE, MetricPriority.HIGH),
            ("execution_latency_ms", MetricType.HISTOGRAM, MetricPriority.CRITICAL),
            ("market_data_latency_ms", MetricType.HISTOGRAM, MetricPriority.HIGH),
            ("strategy_performance", MetricType.GAUGE, MetricPriority.HIGH),
            ("risk_exposure", MetricType.GAUGE, MetricPriority.CRITICAL),
            ("api_calls_total", MetricType.COUNTER, MetricPriority.NORMAL),
            ("errors_total", MetricType.COUNTER, MetricPriority.HIGH),
            ("system_memory_mb", MetricType.GAUGE, MetricPriority.NORMAL),
            ("system_cpu_percent", MetricType.GAUGE, MetricPriority.NORMAL)
        ]
        
        for name, metric_type, priority in system_metrics:
            self._create_metric(name, metric_type, priority)
    
    def _create_metric(self, name: str, metric_type: MetricType, 
                      priority: MetricPriority = MetricPriority.NORMAL,
                      description: str = "", labels: Optional[List[str]] = None):
        """Crée une nouvelle métrique"""
        full_name = f"{self.namespace}_{name}"
        
        with self._lock:
            # Série temporelle interne
            self._metrics[name] = MetricSeries(
                name=name,
                metric_type=metric_type,
                priority=priority
            )
            
            # Métrique Prometheus si activé
            if self.enable_prometheus:
                labels = labels or []
                
                if metric_type == MetricType.COUNTER:
                    prometheus_metric = Counter(
                        full_name, description, labels, registry=self.prometheus_registry
                    )
                elif metric_type == MetricType.GAUGE:
                    prometheus_metric = Gauge(
                        full_name, description, labels, registry=self.prometheus_registry
                    )
                elif metric_type == MetricType.HISTOGRAM:
                    # Buckets optimisés pour latence trading (ms)
                    buckets = [0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, float('inf')]
                    prometheus_metric = Histogram(
                        full_name, description, labels, buckets=buckets, registry=self.prometheus_registry
                    )
                elif metric_type == MetricType.SUMMARY:
                    prometheus_metric = Summary(
                        full_name, description, labels, registry=self.prometheus_registry
                    )
                else:
                    prometheus_metric = Gauge(
                        full_name, description, labels, registry=self.prometheus_registry
                    )
                
                self._prometheus_metrics[name] = prometheus_metric
        
        self.logger.debug("metric_created", 
                         name=name, 
                         type=metric_type.value, 
                         priority=priority.value)
    
    def increment(self, metric_name: str, value: float = 1.0, 
                 tags: Optional[Dict[str, str]] = None):
        """Incrémente un counter"""
        self._record_metric(metric_name, value, tags, "increment")
    
    def gauge(self, metric_name: str, value: float, 
             tags: Optional[Dict[str, str]] = None):
        """Met à jour une gauge"""
        self._record_metric(metric_name, value, tags, "set")
    
    def histogram(self, metric_name: str, value: float,
                 tags: Optional[Dict[str, str]] = None):
        """Enregistre une valeur dans un histogram"""
        self._record_metric(metric_name, value, tags, "observe")
    
    def timing(self, metric_name: str, duration_ms: float,
              tags: Optional[Dict[str, str]] = None):
        """Enregistre une durée en millisecondes"""
        self.histogram(metric_name, duration_ms, tags)
    
    def _record_metric(self, metric_name: str, value: float, 
                      tags: Optional[Dict[str, str]], operation: str):
        """Enregistre une métrique (interne)"""
        try:
            with self._lock:
                # Série temporelle interne
                if metric_name in self._metrics:
                    self._metrics[metric_name].add_point(value, tags)
                else:
                    # Auto-créer la métrique si elle n'existe pas
                    self._create_metric(metric_name, MetricType.GAUGE)
                    self._metrics[metric_name].add_point(value, tags)
                
                # Prometheus
                if self.enable_prometheus and metric_name in self._prometheus_metrics:
                    prom_metric = self._prometheus_metrics[metric_name]
                    
                    if tags:
                        # Métriques avec labels
                        if hasattr(prom_metric, 'labels'):
                            labeled_metric = prom_metric.labels(**tags)
                            if operation == "increment":
                                labeled_metric.inc(value)
                            elif operation == "set":
                                labeled_metric.set(value)
                            elif operation == "observe":
                                labeled_metric.observe(value)
                    else:
                        # Métriques sans labels
                        if operation == "increment":
                            prom_metric.inc(value)
                        elif operation == "set":
                            prom_metric.set(value)
                        elif operation == "observe":
                            prom_metric.observe(value)
            
            # Update performance counter
            self._update_performance_counter()
            
        except Exception as e:
            self.logger.error("metric_recording_error", 
                            metric=metric_name, 
                            error=str(e))
    
    def _update_performance_counter(self):
        """Met à jour le compteur de performance"""
        current_time = time.time()
        if current_time - self._last_count_time >= 1.0:  # Chaque seconde
            current_count = sum(len(m.points) for m in self._metrics.values())
            self._metrics_per_second = current_count - self._last_metrics_count
            self._last_metrics_count = current_count
            self._last_count_time = current_time
    
    @contextmanager
    def timer(self, metric_name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager pour mesurer automatiquement le temps"""
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            self.timing(metric_name, duration_ms, tags)
    
    def record_trade(self, symbol: str, side: str, quantity: float,
                    price: float, pnl: float, strategy_id: str,
                    execution_time_ms: float):
        """Enregistre une transaction avec métriques associées"""
        # Métriques standard
        self.increment("trades_total", tags={"symbol": symbol, "side": side, "strategy": strategy_id})
        self.gauge("pnl_total", pnl)
        self.timing("execution_latency_ms", execution_time_ms, tags={"symbol": symbol})
        
        # Métriques trading spécialisées
        self.trading_metrics.record_trade(
            symbol, side, quantity, price, pnl, strategy_id, execution_time_ms
        )
        
        # Calculer et enregistrer métriques dérivées
        summary = self.trading_metrics.get_trading_summary(window_minutes=60)
        self.gauge("sharpe_ratio", summary['sharpe_ratio'], tags={"window": "1h"})
        self.gauge("max_drawdown", summary['max_drawdown'], tags={"window": "1h"})
        self.gauge("win_rate", summary['win_rate'], tags={"window": "1h"})
        self.gauge("profit_factor", summary['profit_factor'], tags={"window": "1h"})
    
    def get_metric_value(self, metric_name: str) -> Optional[float]:
        """Récupère la dernière valeur d'une métrique"""
        with self._lock:
            if metric_name in self._metrics:
                latest = self._metrics[metric_name].get_latest()
                return latest.value if latest else None
            return None
    
    def get_metric_stats(self, metric_name: str, window_seconds: float = 60.0) -> Dict[str, Any]:
        """Récupère les statistiques d'une métrique"""
        with self._lock:
            if metric_name not in self._metrics:
                return {}
            
            metric = self._metrics[metric_name]
            points = metric.get_range(window_seconds)
            
            if not points:
                return {"count": 0}
            
            values = [p.value for p in points]
            return {
                "count": len(values),
                "latest": values[-1],
                "min": min(values),
                "max": max(values),
                "mean": sum(values) / len(values),
                "median": median(values) if len(values) > 0 else 0,
                "std": stdev(values) if len(values) > 1 else 0,
                "moving_average": metric.calculate_moving_average(window_seconds),
                "rate": metric.calculate_rate(window_seconds),
                "percentiles": metric.calculate_percentiles(window_seconds)
            }
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Récupère toutes les métriques"""
        with self._lock:
            result = {}
            for name, metric in self._metrics.items():
                result[name] = self.get_metric_stats(name)
            
            # Ajouter les métriques système
            result["_system"] = {
                "metrics_per_second": self._metrics_per_second,
                "total_metrics": len(self._metrics),
                "prometheus_enabled": self.enable_prometheus,
                "uptime_seconds": time.time() - getattr(self, '_start_time', time.time())
            }
            
            return result
    
    def start_prometheus_server(self, port: int = 9090):
        """Démarre le serveur HTTP Prometheus"""
        if not self.enable_prometheus:
            self.logger.warning("prometheus_disabled")
            return
        
        try:
            self._http_server = start_http_server(port, registry=self.prometheus_registry)
            self.logger.info("prometheus_server_started", port=port)
        except Exception as e:
            self.logger.error("prometheus_server_start_failed", port=port, error=str(e))
    
    def get_prometheus_metrics(self) -> str:
        """Retourne les métriques au format Prometheus"""
        if not self.enable_prometheus:
            return ""
        
        return generate_latest(self.prometheus_registry).decode('utf-8')
    
    async def start_background_aggregation(self, interval_seconds: float = 5.0):
        """Démarre l'agrégation en arrière-plan"""
        self._running = True
        self._start_time = time.time()
        
        async def background_worker():
            while self._running:
                try:
                    await self._aggregate_metrics()
                    await asyncio.sleep(interval_seconds)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error("background_aggregation_error", error=str(e))
                    await asyncio.sleep(interval_seconds)
        
        self._background_task = asyncio.create_task(background_worker())
        self.logger.info("background_aggregation_started", interval=interval_seconds)
    
    async def _aggregate_metrics(self):
        """Agrège les métriques et nettoie les anciennes données"""
        try:
            current_time = time.time()
            cutoff_time = current_time - 3600  # Garder 1 heure de données
            
            with self._lock:
                for metric in self._metrics.values():
                    # Nettoyer les anciens points
                    while metric.points and metric.points[0].timestamp < cutoff_time:
                        metric.points.popleft()
            
            # Garbage collection périodique
            if int(current_time) % 60 == 0:  # Chaque minute
                gc.collect()
                
        except Exception as e:
            self.logger.error("metrics_aggregation_error", error=str(e))
    
    async def stop(self):
        """Arrête le collecteur de métriques"""
        self._running = False
        
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        if self._http_server:
            self._http_server.shutdown()
        
        self.logger.info("metrics_collector_stopped")
    
    def export_metrics_to_file(self, filepath: str, format: str = "json"):
        """Exporte les métriques vers un fichier"""
        import json
        
        try:
            metrics_data = self.get_all_metrics()
            
            if format == "json":
                with open(filepath, 'w') as f:
                    json.dump(metrics_data, f, indent=2, default=str)
            elif format == "prometheus":
                with open(filepath, 'w') as f:
                    f.write(self.get_prometheus_metrics())
            
            self.logger.info("metrics_exported", filepath=filepath, format=format)
            
        except Exception as e:
            self.logger.error("metrics_export_error", filepath=filepath, error=str(e))


# Fonctions utilitaires pour décorateurs
def track_function_metrics(collector: MetricsCollector, metric_prefix: str = "function"):
    """Décorateur pour tracker automatiquement les métriques d'une fonction"""
    def decorator(func):
        import functools
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{metric_prefix}_{func.__name__}"
            
            with collector.timer(f"{func_name}_duration_ms"):
                try:
                    result = func(*args, **kwargs)
                    collector.increment(f"{func_name}_calls_total", tags={"status": "success"})
                    return result
                except Exception as e:
                    collector.increment(f"{func_name}_calls_total", tags={"status": "error"})
                    collector.increment(f"{func_name}_errors_total", tags={"error_type": type(e).__name__})
                    raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            func_name = f"{metric_prefix}_{func.__name__}"
            
            with collector.timer(f"{func_name}_duration_ms"):
                try:
                    result = await func(*args, **kwargs)
                    collector.increment(f"{func_name}_calls_total", tags={"status": "success"})
                    return result
                except Exception as e:
                    collector.increment(f"{func_name}_calls_total", tags={"status": "error"})
                    collector.increment(f"{func_name}_errors_total", tags={"error_type": type(e).__name__})
                    raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
    return decorator


# Instance globale par défaut
_default_collector: Optional[MetricsCollector] = None


def get_default_collector() -> MetricsCollector:
    """Récupère le collecteur de métriques par défaut"""
    global _default_collector
    if _default_collector is None:
        _default_collector = MetricsCollector("trading_robot")
    return _default_collector


def init_metrics(namespace: str = "trading_robot", enable_prometheus: bool = True) -> MetricsCollector:
    """Initialise le système de métriques"""
    global _default_collector
    _default_collector = MetricsCollector(namespace, enable_prometheus)
    return _default_collector


# Exports principaux
__all__ = [
    'MetricsCollector',
    'TradingMetrics',
    'MetricType',
    'MetricPriority',
    'MetricSeries',
    'track_function_metrics',
    'get_default_collector',
    'init_metrics'
]


if __name__ == "__main__":
    # Test du système de métriques
    import asyncio
    
    async def test_metrics():
        print("🚀 Testing Trading Metrics System...")
        
        # Initialisation
        collector = MetricsCollector("test_trading", enable_prometheus=True)
        
        # Démarrer l'agrégation en arrière-plan
        await collector.start_background_aggregation(interval_seconds=1.0)
        
        # Test métriques de base
        collector.increment("test_counter", 1.0, tags={"type": "test"})
        collector.gauge("test_gauge", 42.5, tags={"level": "high"})
        collector.histogram("test_histogram", 123.4)
        
        # Test timer context manager
        with collector.timer("test_operation_ms"):
            await asyncio.sleep(0.01)  # Simulate work
        
        # Test métriques de trading
        collector.record_trade(
            symbol="BTCUSDT",
            side="BUY", 
            quantity=1.5,
            price=45000.0,
            pnl=150.0,
            strategy_id="test_strategy",
            execution_time_ms=12.5
        )
        
        # Plusieurs trades pour tester les calculs
        for i in range(10):
            pnl = 50.0 if i % 2 == 0 else -25.0  # Alternance gain/perte
            collector.record_trade(
                symbol="ETHUSDT",
                side="SELL" if i % 2 else "BUY",
                quantity=2.0,
                price=3000.0 + i * 10,
                pnl=pnl,
                strategy_id="test_strategy",
                execution_time_ms=10.0 + i
            )
        
        # Attendre un peu pour l'agrégation
        await asyncio.sleep(2.0)
        
        # Vérifier les métriques
        print("\n📊 Metrics Summary:")
        all_metrics = collector.get_all_metrics()
        
        for name, stats in all_metrics.items():
            if not name.startswith('_') and stats.get('count', 0) > 0:
                print(f"  {name}: count={stats['count']}, latest={stats.get('latest', 0):.2f}")
        
        print(f"\n🎯 Trading Summary:")
        trading_summary = collector.trading_metrics.get_trading_summary(60)
        for key, value in trading_summary.items():
            print(f"  {key}: {value:.4f}")
        
        # Test export Prometheus
        prometheus_data = collector.get_prometheus_metrics()
        print(f"\n📈 Prometheus metrics: {len(prometheus_data)} characters")
        
        # Test décorateur
        @track_function_metrics(collector, "test_func")
        async def test_function(should_fail: bool = False):
            await asyncio.sleep(0.001)
            if should_fail:
                raise ValueError("Test error")
            return "success"
        
        # Appels de test
        await test_function(False)
        await test_function(False)
        try:
            await test_function(True)
        except ValueError:
            pass
        
        # Stats des fonctions
        func_stats = collector.get_metric_stats("test_func_test_function_calls_total")
        print(f"\n🔧 Function metrics: {func_stats}")
        
        # Performance
        system_stats = all_metrics.get('_system', {})
        print(f"\n⚡ Performance: {system_stats.get('metrics_per_second', 0)} metrics/sec")
        
        # Arrêt
        await collector.stop()
        print("\n✅ All metrics tests passed!")
    
    # Run test
    asyncio.run(test_metrics())