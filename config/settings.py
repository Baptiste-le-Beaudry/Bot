"""
Configuration Complète du Robot de Trading Algorithmique IA
==========================================================

Ce module définit toute la configuration du système de trading avec validation
robuste via Pydantic. Support pour configuration par environnement et secrets
management pour la sécurité en production.

Architecture:
- Configuration hiérarchique avec validation Pydantic
- Support multi-environnement (dev/staging/prod)
- Gestion sécurisée des credentials
- Paramètres optimisés pour performance sub-milliseconde
- Configuration hot-reload pour paramètres non-critiques

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import os
import time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set
from urllib.parse import urlparse

import yaml
from pydantic import (
    BaseModel, BaseSettings, Field, validator, root_validator,
    SecretStr, AnyUrl, DirectoryPath, FilePath
)
from pydantic.env_settings import SettingsSourceCallable


class Environment(str, Enum):
    """Environnements de déploiement supportés"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Niveaux de log supportés"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ExchangeType(str, Enum):
    """Types d'exchanges supportés"""
    BINANCE = "binance"
    BINANCE_US = "binance_us"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    FTX = "ftx"
    INTERACTIVE_BROKERS = "interactive_brokers"
    ALPACA = "alpaca"
    PAPER_TRADING = "paper_trading"


class DatabaseType(str, Enum):
    """Types de bases de données supportées"""
    TIMESCALEDB = "timescaledb"
    CLICKHOUSE = "clickhouse"
    INFLUXDB = "influxdb"
    POSTGRESQL = "postgresql"


class StrategyType(str, Enum):
    """Types de stratégies disponibles"""
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    MARKET_MAKING = "market_making"
    SCALPING = "scalping"
    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    DRL_AGENT = "drl_agent"
    ENSEMBLE = "ensemble"


# Configuration des Exchanges et APIs
class ExchangeConfig(BaseModel):
    """Configuration pour un exchange spécifique"""
    name: str
    exchange_type: ExchangeType
    enabled: bool = True
    
    # Credentials (seront chiffrés en production)
    api_key: SecretStr
    api_secret: SecretStr
    passphrase: Optional[SecretStr] = None
    
    # Configuration réseau
    base_url: Optional[AnyUrl] = None
    sandbox_mode: bool = False
    rate_limit_requests_per_second: int = 10
    max_connections: int = 10
    timeout_seconds: int = 30
    
    # Configuration trading
    supported_pairs: List[str] = Field(default_factory=list)
    default_leverage: Decimal = Decimal("1.0")
    min_order_size: Dict[str, Decimal] = Field(default_factory=dict)
    fee_rate: Decimal = Decimal("0.001")  # 0.1% par défaut
    
    @validator('api_key', 'api_secret')
    def validate_credentials(cls, v):
        if not v or len(v.get_secret_value()) < 10:
            raise ValueError("API credentials must be at least 10 characters")
        return v


# Configuration Risk Management
class RiskConfig(BaseModel):
    """Configuration complète du risk management"""
    
    # Limites globales
    max_total_exposure: Decimal = Decimal("1.0")  # 100% du capital
    max_daily_loss: Decimal = Decimal("0.05")     # 5% par jour
    max_drawdown: Decimal = Decimal("0.20")       # 20% max
    max_position_size: Decimal = Decimal("0.10")  # 10% par position
    max_correlation: Decimal = Decimal("0.80")    # 80% corrélation max
    
    # VaR et Expected Shortfall
    var_confidence_level: Decimal = Decimal("0.99")  # 99% VaR
    var_holding_period_days: int = 1
    expected_shortfall_confidence: Decimal = Decimal("0.975")  # 97.5% ES
    
    # Position sizing
    use_kelly_criterion: bool = True
    kelly_fraction: Decimal = Decimal("0.25")  # 25% du Kelly optimal
    volatility_target: Decimal = Decimal("0.12")  # 12% volatilité annuelle
    
    # Stop-loss et hedging
    use_dynamic_stops: bool = True
    default_stop_loss: Decimal = Decimal("0.02")  # 2% stop-loss
    trailing_stop_distance: Decimal = Decimal("0.015")  # 1.5% trailing
    auto_hedge_threshold: Decimal = Decimal("0.15")  # Hedge automatique à 15%
    
    # Circuit breakers
    enable_circuit_breakers: bool = True
    circuit_breaker_loss_threshold: Decimal = Decimal("0.03")  # 3% perte trigger
    circuit_breaker_cooldown_minutes: int = 30
    
    @validator('max_daily_loss', 'max_drawdown', 'max_position_size')
    def validate_risk_limits(cls, v):
        if v <= 0 or v > 1:
            raise ValueError("Risk limits must be between 0 and 1")
        return v


# Configuration Performance et Latence
class PerformanceConfig(BaseModel):
    """Configuration pour optimisation des performances"""
    
    # Event processing
    event_queue_size: int = 10000
    event_store_size: int = 50000
    max_events_per_second: int = 10000
    
    # Latence cibles (en millisecondes)
    target_signal_to_order_latency_ms: int = 5
    target_order_to_market_latency_ms: int = 10
    target_data_processing_latency_ms: int = 1
    
    # Optimisations mémoire
    enable_memory_pooling: bool = True
    gc_collection_threshold: int = 1000
    max_memory_usage_gb: int = 8
    
    # Optimisations CPU
    enable_numba_jit: bool = True
    max_worker_threads: int = 4
    cpu_affinity: Optional[List[int]] = None
    
    # Cache et buffer
    data_buffer_size: int = 100000
    indicator_cache_size: int = 10000
    enable_redis_cache: bool = True


# Configuration Stratégies
class StrategyConfig(BaseModel):
    """Configuration d'une stratégie de trading"""
    strategy_id: str
    strategy_type: StrategyType
    enabled: bool = True
    
    # Allocation de capital
    capital_allocation: Decimal = Decimal("0.1")  # 10% par défaut
    max_positions: int = 10
    
    # Paramètres spécifiques (sera étendu par stratégie)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    # Risk management spécifique
    risk_override: Optional[RiskConfig] = None
    
    # Performance
    min_sharpe_ratio: Decimal = Decimal("1.0")
    max_monthly_trades: int = 1000
    
    @validator('capital_allocation')
    def validate_allocation(cls, v):
        if v <= 0 or v > 1:
            raise ValueError("Capital allocation must be between 0 and 1")
        return v


# Configuration Base de Données
class DatabaseConfig(BaseModel):
    """Configuration des bases de données"""
    
    # Base principale (TimescaleDB)
    primary_db_type: DatabaseType = DatabaseType.TIMESCALEDB
    primary_db_url: SecretStr
    primary_db_pool_size: int = 20
    primary_db_timeout: int = 30
    
    # Cache (Redis)
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")
    redis_pool_size: int = 10
    redis_timeout: int = 5
    
    # Analytics (ClickHouse optionnel)
    analytics_db_url: Optional[SecretStr] = None
    analytics_enabled: bool = False
    
    # Configuration data retention
    tick_data_retention_days: int = 30
    ohlc_data_retention_days: int = 365 * 2  # 2 ans
    trades_retention_days: int = 365 * 7    # 7 ans (compliance)
    
    # Compression et archivage
    enable_compression: bool = True
    compression_ratio_target: Decimal = Decimal("0.1")  # 90% compression
    auto_archive_enabled: bool = True


# Configuration Monitoring
class MonitoringConfig(BaseModel):
    """Configuration monitoring et observabilité"""
    
    # Logging
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    log_file_path: Optional[Path] = None
    log_rotation_size: str = "100MB"
    log_retention_days: int = 30
    
    # Métriques
    metrics_enabled: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    
    # Health checks
    health_check_interval: int = 30  # secondes
    health_check_timeout: int = 10
    health_check_port: int = 8080
    
    # Alerting
    enable_alerts: bool = True
    slack_webhook_url: Optional[SecretStr] = None
    email_alerts_enabled: bool = False
    sms_alerts_enabled: bool = False
    pagerduty_enabled: bool = False
    
    # Observability tools
    enable_prometheus: bool = True
    enable_grafana: bool = True
    enable_jaeger_tracing: bool = False
    
    # Performance monitoring
    track_latencies: bool = True
    track_memory_usage: bool = True
    track_cpu_usage: bool = True


# Configuration Machine Learning
class MLConfig(BaseModel):
    """Configuration pour les modèles ML et DRL"""
    
    # Training
    enable_training: bool = True
    training_data_days: int = 365
    validation_split: Decimal = Decimal("0.2")
    test_split: Decimal = Decimal("0.1")
    
    # Model storage
    model_storage_path: DirectoryPath = Path("./models")
    checkpoint_interval_hours: int = 6
    max_model_versions: int = 10
    
    # DRL Configuration
    drl_framework: str = "stable_baselines3"  # ou "rllib"
    drl_algorithm: str = "PPO"  # PPO, SAC, DQN
    training_episodes: int = 10000
    learning_rate: Decimal = Decimal("0.0003")
    
    # Feature engineering
    feature_lookback_periods: List[int] = [5, 15, 30, 60, 240]  # en minutes
    technical_indicators: List[str] = [
        "SMA", "EMA", "RSI", "MACD", "BollingerBands", "ATR"
    ]
    
    # Hardware
    use_gpu: bool = True
    gpu_memory_limit: Optional[int] = None  # MB
    parallel_environments: int = 4


# Configuration principale
class TradingConfig(BaseSettings):
    """Configuration principale du système de trading"""
    
    # Méta-configuration
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    config_version: str = "1.0.0"
    start_time: float = Field(default_factory=time.time)
    
    # Paths et fichiers
    base_dir: DirectoryPath = Path.cwd()
    data_dir: DirectoryPath = Path("./data")
    logs_dir: DirectoryPath = Path("./logs")
    temp_dir: DirectoryPath = Path("./temp")
    
    # Configuration des sous-systèmes
    exchanges: Dict[str, ExchangeConfig] = Field(default_factory=dict)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    strategies: Dict[str, StrategyConfig] = Field(default_factory=dict)
    database: DatabaseConfig
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    
    # Trading session
    trading_enabled: bool = True
    paper_trading_mode: bool = True  # Sécurité par défaut
    session_timeout_hours: int = 24
    
    # Symbols et marchés
    primary_quote_currency: str = "USDT"
    supported_base_currencies: Set[str] = {"BTC", "ETH", "USDT", "USD"}
    excluded_symbols: Set[str] = Field(default_factory=set)
    
    # API et intégrations
    external_apis: Dict[str, str] = Field(default_factory=dict)
    webhook_endpoints: Dict[str, AnyUrl] = Field(default_factory=dict)
    
    # Sécurité
    encryption_key: Optional[SecretStr] = None
    jwt_secret: SecretStr = SecretStr("your-secret-key-change-in-production")
    api_rate_limit: int = 1000  # requests per minute
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        env_nested_delimiter = "__"
        case_sensitive = False
        validate_assignment = True
        
        # Sources de configuration par ordre de priorité
        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            return (
                init_settings,      # Arguments explicites
                env_settings,       # Variables d'environnement
                yaml_config_source, # Fichier YAML
                file_secret_settings, # Secrets files
            )
    
    @root_validator
    def validate_configuration(cls, values):
        """Validation globale de la configuration"""
        env = values.get('environment')
        
        # Validation production
        if env == Environment.PRODUCTION:
            if values.get('debug'):
                raise ValueError("Debug mode cannot be enabled in production")
            if values.get('paper_trading_mode'):
                print("WARNING: Paper trading mode enabled in production")
        
        # Validation exchanges
        exchanges = values.get('exchanges', {})
        if not exchanges:
            raise ValueError("At least one exchange must be configured")
        
        # Validation allocation de capital des stratégies
        strategies = values.get('strategies', {})
        total_allocation = sum(
            strategy.capital_allocation 
            for strategy in strategies.values()
        )
        if total_allocation > Decimal("1.0"):
            raise ValueError(f"Total strategy allocation ({total_allocation}) exceeds 100%")
        
        return values
    
    @validator('base_dir', 'data_dir', 'logs_dir', 'temp_dir')
    def create_directories(cls, v):
        """Crée les répertoires s'ils n'existent pas"""
        if not v.exists():
            v.mkdir(parents=True, exist_ok=True)
        return v
    
    def get_exchange_config(self, exchange_name: str) -> Optional[ExchangeConfig]:
        """Récupère la configuration d'un exchange"""
        return self.exchanges.get(exchange_name)
    
    def get_strategy_config(self, strategy_id: str) -> Optional[StrategyConfig]:
        """Récupère la configuration d'une stratégie"""
        return self.strategies.get(strategy_id)
    
    def is_production(self) -> bool:
        """Vérifie si on est en environnement de production"""
        return self.environment == Environment.PRODUCTION
    
    def get_db_url(self, mask_credentials: bool = True) -> str:
        """Récupère l'URL de la DB avec option de masquer les credentials"""
        url = self.database.primary_db_url.get_secret_value()
        if mask_credentials:
            parsed = urlparse(url)
            return f"{parsed.scheme}://***:***@{parsed.hostname}:{parsed.port}{parsed.path}"
        return url


def yaml_config_source(settings: BaseSettings) -> Dict[str, Any]:
    """Source de configuration YAML personnalisée"""
    config_file = Path("config.yaml")
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    return {}


# Factory functions pour créer des configurations prédéfinies
def create_development_config() -> TradingConfig:
    """Crée une configuration de développement"""
    exchanges = {
        "binance_testnet": ExchangeConfig(
            name="binance_testnet",
            exchange_type=ExchangeType.BINANCE,
            api_key=SecretStr("test_api_key"),
            api_secret=SecretStr("test_api_secret"),
            sandbox_mode=True,
            supported_pairs=["BTCUSDT", "ETHUSDT", "ADAUSDT"],
            rate_limit_requests_per_second=5
        )
    }
    
    strategies = {
        "simple_ma": StrategyConfig(
            strategy_id="simple_ma",
            strategy_type=StrategyType.TREND_FOLLOWING,
            capital_allocation=Decimal("0.1"),
            parameters={
                "fast_ma_period": 10,
                "slow_ma_period": 30,
                "max_positions": 3
            }
        )
    }
    
    database = DatabaseConfig(
        primary_db_url=SecretStr("postgresql://user:pass@localhost:5432/trading_dev"),
        redis_url=SecretStr("redis://localhost:6379/1")
    )
    
    return TradingConfig(
        environment=Environment.DEVELOPMENT,
        debug=True,
        paper_trading_mode=True,
        exchanges=exchanges,
        strategies=strategies,
        database=database
    )


def create_production_config() -> TradingConfig:
    """Crée une configuration de production (à personnaliser)"""
    # Cette fonction devrait charger depuis des variables d'environnement
    # ou des services de secrets management en production
    
    database = DatabaseConfig(
        primary_db_url=SecretStr(os.getenv("DATABASE_URL", "postgresql://localhost:5432/trading")),
        redis_url=SecretStr(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    )
    
    return TradingConfig(
        environment=Environment.PRODUCTION,
        debug=False,
        paper_trading_mode=False,  # Attention: real trading!
        database=database,
        monitoring=MonitoringConfig(
            log_level=LogLevel.INFO,
            enable_alerts=True,
            slack_webhook_url=SecretStr(os.getenv("SLACK_WEBHOOK_URL", ""))
        )
    )


# Instance globale (sera initialisée selon l'environnement)
config: Optional[TradingConfig] = None


def init_config(env: Optional[Environment] = None) -> TradingConfig:
    """Initialise la configuration globale"""
    global config
    
    if env is None:
        env = Environment(os.getenv("TRADING_ENV", Environment.DEVELOPMENT))
    
    if env == Environment.DEVELOPMENT:
        config = create_development_config()
    elif env == Environment.PRODUCTION:
        config = create_production_config()
    else:
        # Configuration par défaut depuis environnement/fichiers
        config = TradingConfig()
    
    return config


def get_config() -> TradingConfig:
    """Récupère la configuration globale"""
    global config
    if config is None:
        config = init_config()
    return config


# Export des principales classes pour usage externe
__all__ = [
    "TradingConfig",
    "ExchangeConfig", 
    "RiskConfig",
    "StrategyConfig",
    "DatabaseConfig",
    "MonitoringConfig",
    "MLConfig",
    "PerformanceConfig",
    "Environment",
    "ExchangeType",
    "StrategyType",
    "init_config",
    "get_config",
    "create_development_config",
    "create_production_config"
]


if __name__ == "__main__":
    # Test de la configuration
    print("🚀 Testing Trading Configuration...")
    
    # Test configuration de développement
    dev_config = create_development_config()
    print(f"✅ Development config created: {dev_config.environment}")
    print(f"📊 Exchanges configured: {list(dev_config.exchanges.keys())}")
    print(f"🔄 Strategies configured: {list(dev_config.strategies.keys())}")
    print(f"💾 Database: {dev_config.get_db_url()}")
    
    # Test validation
    try:
        # Cette config devrait échouer (allocation > 100%)
        invalid_strategies = {
            "strat1": StrategyConfig(
                strategy_id="strat1", 
                strategy_type=StrategyType.SCALPING,
                capital_allocation=Decimal("0.7")
            ),
            "strat2": StrategyConfig(
                strategy_id="strat2", 
                strategy_type=StrategyType.MARKET_MAKING,
                capital_allocation=Decimal("0.6")
            )
        }
        
        TradingConfig(
            environment=Environment.DEVELOPMENT,
            exchanges=dev_config.exchanges,
            strategies=invalid_strategies,
            database=dev_config.database
        )
        print("❌ Validation should have failed!")
        
    except ValueError as e:
        print(f"✅ Validation working: {e}")
    
    print("\n📋 Configuration summary:")
    print(f"- Environment: {dev_config.environment}")
    print(f"- Paper trading: {dev_config.paper_trading_mode}")
    print(f"- Max daily loss: {dev_config.risk.max_daily_loss*100}%")
    print(f"- Event queue size: {dev_config.performance.event_queue_size}")
    print(f"- Health check interval: {dev_config.monitoring.health_check_interval}s")
    
    print("\n🎯 Configuration system ready!")