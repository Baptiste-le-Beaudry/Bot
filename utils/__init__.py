"""
Utilities Package
================

Collection d'utilitaires et helpers pour le robot de trading.
Fournit logging structuré, calcul de métriques, décorateurs
et fonctions d'aide générales.

Composants:
    - Logger: Système de logging structuré avec contexte
    - Metrics: Calculs de métriques financières
    - Decorators: Décorateurs pour retry, cache, performance
    - Helpers: Fonctions utilitaires générales

Usage:
    from utils import get_logger, calculate_sharpe, retry_async
    
    logger = get_logger(__name__)
    sharpe = calculate_sharpe(returns)
"""

from utils.logger import (
    get_structured_logger,
    setup_logging,
    LogContext,
    TradeLogger,
    PerformanceLogger,
    ErrorLogger,
    AuditLogger
)

from utils.metrics import (
    MetricsCalculator,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_var,
    calculate_beta,
    calculate_alpha,
    calculate_information_ratio,
    calculate_calmar_ratio,
    calculate_omega_ratio
)

from utils.decorators import (
    retry_async,
    retry_sync,
    circuit_breaker,
    rate_limit,
    cache_result,
    measure_time,
    validate_input,
    require_auth,
    log_execution,
    handle_errors
)

from utils.helpers import (
    # Time helpers
    get_utc_now,
    timestamp_to_datetime,
    datetime_to_timestamp,
    get_market_hours,
    is_market_open,
    
    # Data helpers
    round_to_tick_size,
    normalize_symbol,
    parse_timeframe,
    convert_timeframe,
    
    # Math helpers
    safe_divide,
    calculate_percentage_change,
    exponential_smoothing,
    moving_average,
    
    # System helpers
    get_memory_usage,
    get_cpu_usage,
    create_pid_file,
    remove_pid_file,
    
    # Crypto helpers
    generate_nonce,
    create_signature,
    encrypt_credentials,
    decrypt_credentials
)

# Version
__version__ = "1.0.0"

# Exports publics
__all__ = [
    # Logger
    "get_structured_logger",
    "setup_logging",
    "LogContext",
    "TradeLogger",
    "PerformanceLogger",
    "ErrorLogger",
    "AuditLogger",
    
    # Metrics
    "MetricsCalculator",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_max_drawdown",
    "calculate_var",
    "calculate_beta",
    "calculate_alpha",
    "calculate_information_ratio",
    "calculate_calmar_ratio",
    "calculate_omega_ratio",
    
    # Decorators
    "retry_async",
    "retry_sync",
    "circuit_breaker",
    "rate_limit",
    "cache_result",
    "measure_time",
    "validate_input",
    "require_auth",
    "log_execution",
    "handle_errors",
    
    # Helpers
    "get_utc_now",
    "timestamp_to_datetime",
    "datetime_to_timestamp",
    "get_market_hours",
    "is_market_open",
    "round_to_tick_size",
    "normalize_symbol",
    "parse_timeframe",
    "convert_timeframe",
    "safe_divide",
    "calculate_percentage_change",
    "exponential_smoothing",
    "moving_average",
    "get_memory_usage",
    "get_cpu_usage",
    "create_pid_file",
    "remove_pid_file",
    "generate_nonce",
    "create_signature",
    "encrypt_credentials",
    "decrypt_credentials"
]

# Configuration par défaut
DEFAULT_UTILS_CONFIG = {
    "logging": {
        "level": "INFO",
        "format": "json",
        "file": "logs/trading.log",
        "rotation": "100MB",
        "retention": "30 days",
        "structured": True
    },
    "metrics": {
        "precision": 4,
        "annualization_factor": 252,  # Trading days
        "risk_free_rate": 0.02,
        "confidence_level": 0.95
    },
    "decorators": {
        "retry_attempts": 3,
        "retry_delay": 1.0,
        "circuit_breaker_threshold": 5,
        "circuit_breaker_timeout": 60,
        "rate_limit_calls": 100,
        "rate_limit_period": 60,
        "cache_ttl": 300
    }
}

# Constantes utiles
TRADING_DAYS_PER_YEAR = 252
SECONDS_PER_DAY = 86400
NANOSECONDS_PER_SECOND = 1_000_000_000

# Mapping des exchanges et leurs spécificités
EXCHANGE_SPECS = {
    "binance": {
        "tick_size": 0.00000001,
        "lot_size": 0.00000001,
        "max_orders": 200,
        "rate_limit": 1200,
        "base_url": "https://api.binance.com"
    },
    "coinbase": {
        "tick_size": 0.01,
        "lot_size": 0.00000001,
        "max_orders": 100,
        "rate_limit": 600,
        "base_url": "https://api.exchange.coinbase.com"
    }
}

# Types et structures communes
from enum import Enum
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime
from typing import Optional, Union, List, Dict, Any


class TimeFrame(Enum):
    """Timeframes standards"""
    TICK = "tick"
    SECOND_1 = "1s"
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"


@dataclass
class PriceLevel:
    """Niveau de prix avec volume"""
    price: Decimal
    volume: Decimal
    count: Optional[int] = None
    
    @property
    def value(self) -> Decimal:
        return self.price * self.volume


@dataclass
class TimeRange:
    """Plage temporelle"""
    start: datetime
    end: datetime
    
    @property
    def duration(self) -> timedelta:
        return self.end - self.start
        
    def contains(self, timestamp: datetime) -> bool:
        return self.start <= timestamp <= self.end
        
    def overlaps(self, other: 'TimeRange') -> bool:
        return not (self.end < other.start or self.start > other.end)


# Fonctions d'initialisation
def setup_utils(config: dict = None):
    """Configure le package utils"""
    cfg = {**DEFAULT_UTILS_CONFIG, **(config or {})}
    
    # Setup logging
    setup_logging(cfg['logging'])
    
    # Configurer les décorateurs
    from utils.decorators import configure_decorators
    configure_decorators(cfg['decorators'])
    
    return cfg


# Classes utilitaires
class CircularBuffer:
    """Buffer circulaire pour données en streaming"""
    
    def __init__(self, size: int):
        self.size = size
        self.buffer = []
        self.index = 0
        
    def append(self, item):
        if len(self.buffer) < self.size:
            self.buffer.append(item)
        else:
            self.buffer[self.index] = item
            self.index = (self.index + 1) % self.size
            
    def get_all(self) -> list:
        if len(self.buffer) < self.size:
            return self.buffer.copy()
        return self.buffer[self.index:] + self.buffer[:self.index]
        
    def get_latest(self, n: int = 1):
        all_items = self.get_all()
        return all_items[-n:] if n <= len(all_items) else all_items


class RateLimiter:
    """Limiteur de débit simple"""
    
    def __init__(self, calls: int, period: float):
        self.calls = calls
        self.period = period
        self.timestamps = []
        
    async def acquire(self):
        now = time.time()
        # Nettoyer les anciens timestamps
        self.timestamps = [t for t in self.timestamps if now - t < self.period]
        
        if len(self.timestamps) >= self.calls:
            # Attendre
            sleep_time = self.period - (now - self.timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                
        self.timestamps.append(time.time())


# Gestionnaire d'erreurs global
class ErrorHandler:
    """Gestionnaire d'erreurs centralisé"""
    
    def __init__(self):
        self.handlers = {}
        self.logger = get_structured_logger(__name__)
        
    def register(self, error_type: type, handler: callable):
        """Enregistre un handler pour un type d'erreur"""
        self.handlers[error_type] = handler
        
    async def handle(self, error: Exception, context: dict = None):
        """Gère une erreur"""
        error_type = type(error)
        
        # Chercher un handler spécifique
        for exc_type, handler in self.handlers.items():
            if issubclass(error_type, exc_type):
                return await handler(error, context)
                
        # Handler par défaut
        self.logger.error(
            "Unhandled error",
            error=str(error),
            error_type=error_type.__name__,
            context=context,
            exc_info=True
        )


# Instance globale
_error_handler = ErrorHandler()


def get_error_handler() -> ErrorHandler:
    """Obtient le gestionnaire d'erreurs global"""
    return _error_handler


# Validation helpers
def validate_price(price: Union[str, float, Decimal]) -> Decimal:
    """Valide et convertit un prix"""
    try:
        price_decimal = Decimal(str(price))
        if price_decimal <= 0:
            raise ValueError("Price must be positive")
        return price_decimal
    except Exception as e:
        raise ValueError(f"Invalid price: {price}") from e


def validate_quantity(quantity: Union[str, float, Decimal]) -> Decimal:
    """Valide et convertit une quantité"""
    try:
        qty_decimal = Decimal(str(quantity))
        if qty_decimal <= 0:
            raise ValueError("Quantity must be positive")
        return qty_decimal
    except Exception as e:
        raise ValueError(f"Invalid quantity: {quantity}") from e


# Imports requis
import time
import asyncio
from datetime import timedelta

# Auto-configuration au chargement
try:
    _utils_config = setup_utils()
except Exception as e:
    import warnings
    warnings.warn(f"Failed to setup utils: {e}")

# Initialisation
import logging
logger = logging.getLogger(__name__)
logger.info(f"Utils package initialized - Version {__version__}")