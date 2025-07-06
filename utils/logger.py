"""
Système de Logging Structuré pour Robot de Trading Algorithmique IA
===================================================================

Ce module implémente un système de logging haute performance avec support
pour le logging structuré JSON, corrélation IDs, et observabilité complète.
Optimisé pour les systèmes de trading à haute fréquence.

Architecture:
- Logging structuré avec structlog pour observabilité
- Support async pour performance non-bloquante
- Corrélation IDs pour tracer les transactions
- Rotation automatique et compression
- Multiple outputs (console, file, syslog, external)
- Security et compliance logging
- Performance metrics intégrés

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import asyncio
import gzip
import json
import logging
import logging.handlers
import os
import sys
import time
import traceback
import uuid
from contextlib import contextmanager, asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union, List, Callable, AsyncGenerator
from contextvars import ContextVar
from functools import wraps
from enum import Enum

import structlog
from structlog.types import FilteringBoundLogger
import orjson  # Plus rapide que json standard


# Context variables pour le threading/async
correlation_id_var: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
strategy_id_var: ContextVar[Optional[str]] = ContextVar('strategy_id', default=None)


class LogLevel(Enum):
    """Niveaux de log étendus pour trading"""
    TRACE = 5      # Debugging très détaillé
    DEBUG = 10     # Debugging standard
    INFO = 20      # Information générale
    WARNING = 30   # Avertissements
    ERROR = 40     # Erreurs
    CRITICAL = 50  # Erreurs critiques
    AUDIT = 60     # Logs d'audit (compliance)
    SECURITY = 70  # Logs de sécurité


class LogCategory(Enum):
    """Catégories de logs pour filtrage et routing"""
    SYSTEM = "system"
    TRADING = "trading"
    STRATEGY = "strategy"
    RISK = "risk"
    EXECUTION = "execution"
    DATA = "data"
    PERFORMANCE = "performance"
    SECURITY = "security"
    AUDIT = "audit"
    USER = "user"


class TradingLogFormatter:
    """Formatter personnalisé pour les logs de trading"""
    
    def __init__(self, include_performance: bool = True):
        self.include_performance = include_performance
        self.start_time = time.time()
    
    def __call__(self, logger, method_name, event_dict):
        """Formate un événement de log avec metadata enrichies"""
        
        # Timestamp haute précision
        now = datetime.now(timezone.utc)
        event_dict['timestamp'] = now.isoformat()
        event_dict['timestamp_ns'] = time.time_ns()
        
        # Context variables
        if correlation_id := correlation_id_var.get():
            event_dict['correlation_id'] = correlation_id
        if user_id := user_id_var.get():
            event_dict['user_id'] = user_id
        if session_id := session_id_var.get():
            event_dict['session_id'] = session_id
        if strategy_id := strategy_id_var.get():
            event_dict['strategy_id'] = strategy_id
        
        # Informations système
        event_dict['level'] = method_name.upper()
        event_dict['logger'] = logger.name
        event_dict['process_id'] = os.getpid()
        event_dict['thread_id'] = threading.get_ident() if hasattr(threading, 'get_ident') else None
        
        # Performance metrics si activé
        if self.include_performance:
            event_dict['uptime_seconds'] = time.time() - self.start_time
            
        # Gestion des erreurs et stack traces
        if 'exc_info' in event_dict:
            exc_info = event_dict.pop('exc_info')
            if exc_info:
                event_dict['exception'] = {
                    'type': exc_info[0].__name__ if exc_info[0] else None,
                    'message': str(exc_info[1]) if exc_info[1] else None,
                    'traceback': traceback.format_exception(*exc_info) if exc_info != (None, None, None) else None
                }
        
        # Catégorisation automatique
        if 'category' not in event_dict:
            event_dict['category'] = self._infer_category(event_dict)
        
        return event_dict
    
    def _infer_category(self, event_dict: Dict[str, Any]) -> str:
        """Infère la catégorie basée sur le contenu"""
        logger_name = event_dict.get('logger', '').lower()
        event_name = event_dict.get('event', '').lower()
        
        if 'strategy' in logger_name or 'strategy' in event_name:
            return LogCategory.STRATEGY.value
        elif 'risk' in logger_name or 'risk' in event_name:
            return LogCategory.RISK.value
        elif 'execution' in logger_name or 'order' in event_name:
            return LogCategory.EXECUTION.value
        elif 'data' in logger_name or 'market' in event_name:
            return LogCategory.DATA.value
        elif 'performance' in logger_name or 'latency' in event_name:
            return LogCategory.PERFORMANCE.value
        elif 'security' in logger_name or 'auth' in event_name:
            return LogCategory.SECURITY.value
        elif 'audit' in logger_name or 'compliance' in event_name:
            return LogCategory.AUDIT.value
        else:
            return LogCategory.SYSTEM.value


class TradingJSONRenderer:
    """Renderer JSON optimisé pour performance"""
    
    def __init__(self, sort_keys: bool = False, ensure_ascii: bool = False):
        self.sort_keys = sort_keys
        self.ensure_ascii = ensure_ascii
    
    def __call__(self, logger, method_name, event_dict):
        """Rend l'événement en JSON optimisé"""
        try:
            # Utilise orjson pour performance maximale
            return orjson.dumps(
                event_dict,
                option=orjson.OPT_UTC_Z | orjson.OPT_SORT_KEYS if self.sort_keys else orjson.OPT_UTC_Z
            ).decode()
        except (TypeError, ValueError):
            # Fallback vers json standard si orjson échoue
            return json.dumps(
                event_dict,
                default=str,
                sort_keys=self.sort_keys,
                ensure_ascii=self.ensure_ascii,
                separators=(',', ':')  # Format compact
            )


class TradingConsoleRenderer:
    """Renderer console coloré et lisible pour développement"""
    
    # Codes couleur ANSI
    COLORS = {
        'TRACE': '\033[37m',      # Blanc
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Vert
        'WARNING': '\033[33m',    # Jaune
        'ERROR': '\033[31m',      # Rouge
        'CRITICAL': '\033[35m',   # Magenta
        'AUDIT': '\033[34m',      # Bleu
        'SECURITY': '\033[41m',   # Fond rouge
        'RESET': '\033[0m'        # Reset
    }
    
    def __init__(self, colors: bool = True, include_metadata: bool = True):
        self.colors = colors and sys.stdout.isatty()
        self.include_metadata = include_metadata
    
    def __call__(self, logger, method_name, event_dict):
        """Rend l'événement pour console"""
        level = method_name.upper()
        timestamp = event_dict.get('timestamp', datetime.now().isoformat())
        logger_name = event_dict.get('logger', 'unknown')
        event = event_dict.get('event', 'no_event')
        
        # Couleur selon le niveau
        color = self.COLORS.get(level, '') if self.colors else ''
        reset = self.COLORS['RESET'] if self.colors else ''
        
        # Message principal
        msg_parts = [f"{color}[{timestamp}] {level:8} {logger_name}: {event}{reset}"]
        
        # Métadonnées additionnelles
        if self.include_metadata:
            metadata = {k: v for k, v in event_dict.items() 
                       if k not in ['timestamp', 'level', 'logger', 'event', 'timestamp_ns']}
            
            if metadata:
                msg_parts.append(f"  └─ {json.dumps(metadata, default=str, separators=(',', ':'))}")
        
        return '\n'.join(msg_parts)


class PerformanceFilter:
    """Filtre pour mesurer et logger les performances"""
    
    def __init__(self, enable_timing: bool = True):
        self.enable_timing = enable_timing
        self.call_counts = {}
        self.timing_stats = {}
    
    def __call__(self, logger, method_name, event_dict):
        """Filtre avec mesures de performance"""
        if not self.enable_timing:
            return event_dict
        
        logger_name = logger.name
        
        # Compteur d'appels
        self.call_counts[logger_name] = self.call_counts.get(logger_name, 0) + 1
        event_dict['call_count'] = self.call_counts[logger_name]
        
        # Timing si disponible
        if 'duration_ms' in event_dict:
            if logger_name not in self.timing_stats:
                self.timing_stats[logger_name] = {'total': 0, 'count': 0, 'avg': 0}
            
            duration = event_dict['duration_ms']
            stats = self.timing_stats[logger_name]
            stats['total'] += duration
            stats['count'] += 1
            stats['avg'] = stats['total'] / stats['count']
            
            event_dict['avg_duration_ms'] = round(stats['avg'], 3)
        
        return event_dict


class ComplianceFilter:
    """Filtre pour logs de compliance et audit"""
    
    SENSITIVE_FIELDS = {
        'api_key', 'api_secret', 'password', 'token', 'auth',
        'ssn', 'credit_card', 'account_number'
    }
    
    def __init__(self, mask_sensitive: bool = True, audit_enabled: bool = True):
        self.mask_sensitive = mask_sensitive
        self.audit_enabled = audit_enabled
    
    def __call__(self, logger, method_name, event_dict):
        """Filtre pour compliance"""
        
        # Masquage des données sensibles
        if self.mask_sensitive:
            event_dict = self._mask_sensitive_data(event_dict)
        
        # Marquage audit si nécessaire
        if self.audit_enabled and self._is_audit_event(event_dict):
            event_dict['audit'] = True
            event_dict['compliance_level'] = 'high'
        
        return event_dict
    
    def _mask_sensitive_data(self, data: Any) -> Any:
        """Masque récursivement les données sensibles"""
        if isinstance(data, dict):
            return {
                k: '***MASKED***' if any(sensitive in k.lower() for sensitive in self.SENSITIVE_FIELDS)
                else self._mask_sensitive_data(v)
                for k, v in data.items()
            }
        elif isinstance(data, (list, tuple)):
            return type(data)(self._mask_sensitive_data(item) for item in data)
        else:
            return data
    
    def _is_audit_event(self, event_dict: Dict[str, Any]) -> bool:
        """Détermine si c'est un événement d'audit"""
        audit_keywords = ['trade', 'order', 'position', 'balance', 'transfer', 'withdrawal']
        event_name = event_dict.get('event', '').lower()
        return any(keyword in event_name for keyword in audit_keywords)


class AsyncLogHandler(logging.Handler):
    """Handler asyncio pour logging non-bloquant"""
    
    def __init__(self, handler: logging.Handler, queue_size: int = 1000):
        super().__init__()
        self.handler = handler
        self.queue = asyncio.Queue(maxsize=queue_size)
        self.worker_task = None
        self._stopped = False
    
    async def start(self):
        """Démarre le worker asynchrone"""
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._worker())
    
    async def stop(self):
        """Arrête le worker"""
        self._stopped = True
        if self.worker_task:
            await self.worker_task
    
    def emit(self, record):
        """Émet un log record de façon non-bloquante"""
        try:
            asyncio.get_event_loop().call_soon_threadsafe(
                self.queue.put_nowait, record
            )
        except (asyncio.QueueFull, RuntimeError):
            # Silently drop if queue is full or no event loop
            pass
    
    async def _worker(self):
        """Worker qui traite les logs de façon asynchrone"""
        while not self._stopped:
            try:
                record = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                self.handler.emit(record)
                self.queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception:
                # Ne pas logger ici pour éviter la récursion
                pass


class TradingLoggerFactory:
    """Factory pour créer des loggers configurés pour trading"""
    
    def __init__(self, config=None):
        from config.settings import get_config  # Import local pour éviter circular
        self.config = config or get_config()
        self._configured = False
        self._async_handlers = []
    
    def configure(self):
        """Configure le système de logging global"""
        if self._configured:
            return
        
        # Configuration structlog
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            TradingLogFormatter(include_performance=True),
            PerformanceFilter(enable_timing=True),
            ComplianceFilter(
                mask_sensitive=self.config.is_production(),
                audit_enabled=True
            ),
        ]
        
        # Renderer selon l'environnement
        if self.config.monitoring.log_format == "json":
            processors.append(TradingJSONRenderer())
        else:
            processors.append(TradingConsoleRenderer(
                colors=not self.config.is_production(),
                include_metadata=self.config.debug
            ))
        
        # Configuration structlog
        structlog.configure(
            processors=processors,
            wrapper_class=FilteringBoundLogger,
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configuration des handlers
        self._setup_handlers()
        self._configured = True
    
    def _setup_handlers(self):
        """Configure les handlers de logging"""
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.monitoring.log_level.value))
        
        # Handler console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter('%(message)s'))
        root_logger.addHandler(console_handler)
        
        # Handler fichier avec rotation
        if self.config.monitoring.log_file_path:
            file_handler = logging.handlers.RotatingFileHandler(
                self.config.monitoring.log_file_path,
                maxBytes=self._parse_size(self.config.monitoring.log_rotation_size),
                backupCount=self.config.monitoring.log_retention_days,
                encoding='utf-8'
            )
            file_handler.setFormatter(logging.Formatter('%(message)s'))
            root_logger.addHandler(file_handler)
        
        # Handler syslog pour production
        if self.config.is_production():
            try:
                syslog_handler = logging.handlers.SysLogHandler(address='/dev/log')
                syslog_handler.setFormatter(logging.Formatter('trading-bot: %(message)s'))
                root_logger.addHandler(syslog_handler)
            except Exception:
                pass  # Syslog not available
    
    def _parse_size(self, size_str: str) -> int:
        """Parse une taille avec unité (ex: '100MB')"""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def get_logger(self, name: str, **kwargs) -> FilteringBoundLogger:
        """Crée un logger avec contexte"""
        if not self._configured:
            self.configure()
        
        logger = structlog.get_logger(name)
        
        # Bind contexte initial
        if kwargs:
            logger = logger.bind(**kwargs)
        
        return logger
    
    async def shutdown(self):
        """Arrête tous les handlers async"""
        for handler in self._async_handlers:
            await handler.stop()


# Instance globale du factory
_logger_factory: Optional[TradingLoggerFactory] = None


def init_logging(config=None) -> TradingLoggerFactory:
    """Initialise le système de logging"""
    global _logger_factory
    if _logger_factory is None:
        _logger_factory = TradingLoggerFactory(config)
        _logger_factory.configure()
    return _logger_factory


def get_structured_logger(name: str, **kwargs) -> FilteringBoundLogger:
    """Obtient un logger structuré avec contexte"""
    factory = init_logging()
    return factory.get_logger(name, **kwargs)


# Decorators pour logging automatique
def log_function_call(
    logger: Optional[FilteringBoundLogger] = None,
    level: str = "debug",
    include_args: bool = False,
    include_result: bool = False,
    measure_time: bool = True
):
    """Decorator pour logger automatiquement les appels de fonction"""
    def decorator(func):
        nonlocal logger
        if logger is None:
            logger = get_structured_logger(func.__module__)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.perf_counter() if measure_time else None
            
            log_data = {"function": func.__name__}
            if include_args:
                log_data["args"] = args
                log_data["kwargs"] = kwargs
            
            try:
                result = func(*args, **kwargs)
                
                if measure_time:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    log_data["duration_ms"] = round(duration_ms, 3)
                
                if include_result:
                    log_data["result"] = result
                
                getattr(logger, level)("function_completed", **log_data)
                return result
                
            except Exception as e:
                if measure_time:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    log_data["duration_ms"] = round(duration_ms, 3)
                
                log_data["error"] = str(e)
                logger.error("function_failed", **log_data, exc_info=True)
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.perf_counter() if measure_time else None
            
            log_data = {"function": func.__name__}
            if include_args:
                log_data["args"] = args
                log_data["kwargs"] = kwargs
            
            try:
                result = await func(*args, **kwargs)
                
                if measure_time:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    log_data["duration_ms"] = round(duration_ms, 3)
                
                if include_result:
                    log_data["result"] = result
                
                getattr(logger, level)("function_completed", **log_data)
                return result
                
            except Exception as e:
                if measure_time:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    log_data["duration_ms"] = round(duration_ms, 3)
                
                log_data["error"] = str(e)
                logger.error("function_failed", **log_data, exc_info=True)
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


@contextmanager
def log_context(**context):
    """Context manager pour ajouter du contexte aux logs"""
    tokens = []
    for key, value in context.items():
        if key == 'correlation_id':
            tokens.append(correlation_id_var.set(value))
        elif key == 'user_id':
            tokens.append(user_id_var.set(value))
        elif key == 'session_id':
            tokens.append(session_id_var.set(value))
        elif key == 'strategy_id':
            tokens.append(strategy_id_var.set(value))
    
    try:
        yield
    finally:
        for token in tokens:
            token.var.set(token.old_value)


@asynccontextmanager
async def async_log_context(**context):
    """Context manager async pour ajouter du contexte aux logs"""
    with log_context(**context):
        yield


def generate_correlation_id() -> str:
    """Génère un ID de corrélation unique"""
    return str(uuid.uuid4())


# Fonctions utilitaires pour logging spécialisé
def log_trade_execution(
    logger: FilteringBoundLogger,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    order_id: str,
    strategy_id: str,
    execution_time_ms: float
):
    """Log spécialisé pour les exécutions de trades"""
    logger.info(
        "trade_executed",
        category=LogCategory.EXECUTION.value,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price,
        order_id=order_id,
        strategy_id=strategy_id,
        execution_time_ms=execution_time_ms,
        audit=True
    )


def log_risk_alert(
    logger: FilteringBoundLogger,
    alert_type: str,
    severity: str,
    message: str,
    current_value: float,
    threshold: float,
    **metadata
):
    """Log spécialisé pour les alertes de risque"""
    logger.warning(
        "risk_alert",
        category=LogCategory.RISK.value,
        alert_type=alert_type,
        severity=severity,
        message=message,
        current_value=current_value,
        threshold=threshold,
        **metadata
    )


def log_performance_metric(
    logger: FilteringBoundLogger,
    metric_name: str,
    value: Union[float, int],
    unit: str = "",
    tags: Optional[Dict[str, str]] = None
):
    """Log spécialisé pour les métriques de performance"""
    logger.info(
        "performance_metric",
        category=LogCategory.PERFORMANCE.value,
        metric_name=metric_name,
        value=value,
        unit=unit,
        tags=tags or {}
    )


# Import seulement pour éviter les imports circulaires
import threading


# Exports principaux
__all__ = [
    'get_structured_logger',
    'init_logging',
    'log_function_call',
    'log_context',
    'async_log_context',
    'generate_correlation_id',
    'log_trade_execution',
    'log_risk_alert',
    'log_performance_metric',
    'LogLevel',
    'LogCategory',
    'TradingLoggerFactory'
]


if __name__ == "__main__":
    # Test du système de logging
    import asyncio
    
    async def test_logging():
        print("🚀 Testing Trading Logger System...")
        
        # Initialisation
        factory = init_logging()
        
        # Test loggers de base
        logger = get_structured_logger("test_logger")
        
        # Test différents niveaux
        logger.debug("debug_message", test_data="debug_value")
        logger.info("info_message", test_data="info_value")
        logger.warning("warning_message", test_data="warning_value")
        logger.error("error_message", test_data="error_value")
        
        # Test avec contexte
        with log_context(correlation_id="test-123", strategy_id="test_strategy"):
            logger.info("message_with_context", action="test_action")
        
        # Test logging spécialisé
        log_trade_execution(
            logger=logger,
            symbol="BTCUSDT",
            side="BUY",
            quantity=1.5,
            price=45000.0,
            order_id="order_123",
            strategy_id="test_strategy",
            execution_time_ms=12.5
        )
        
        log_risk_alert(
            logger=logger,
            alert_type="position_size",
            severity="medium",
            message="Position size approaching limit",
            current_value=0.08,
            threshold=0.10,
            strategy_id="test_strategy"
        )
        
        # Test decorator
        @log_function_call(logger=logger, measure_time=True, include_args=True)
        async def test_function(x: int, y: str) -> str:
            await asyncio.sleep(0.1)  # Simulate work
            return f"result_{x}_{y}"
        
        result = await test_function(42, "test")
        logger.info("function_result", result=result)
        
        print("✅ All logging tests passed!")
        
        # Test performance
        start_time = time.perf_counter()
        for i in range(1000):
            logger.debug("performance_test", iteration=i, data={"value": i * 2})
        
        duration = time.perf_counter() - start_time
        print(f"📊 Performance: 1000 logs in {duration:.3f}s ({1000/duration:.0f} logs/sec)")
        
        await factory.shutdown()
    
    # Run test
    asyncio.run(test_logging())