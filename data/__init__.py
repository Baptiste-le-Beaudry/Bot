"""
Data Management Package
======================

Système complet de gestion des données pour le trading algorithmique.
Inclut la collecte multi-exchanges, le traitement, la validation,
le stockage optimisé et le feature store.

Composants:
    - Collectors: Collecte de données depuis différentes sources
    - Processors: Normalisation et validation des données
    - Storage: Stockage TimescaleDB et cache Redis
    - DataManager: Orchestrateur principal des données

Usage:
    from data import DataManager, BinanceCollector, DataNormalizer
    
    data_manager = DataManager(config)
    await data_manager.start_collection(['BTCUSDT', 'ETHUSDT'])
"""

# Collectors
from data.collectors import (
    BinanceCollector,
    InteractiveBrokersCollector,
    MultiExchangeCollector,
    BaseCollector,
    CollectorState,
    DataSource
)

# Processors
from data.processors import (
    DataNormalizer,
    DataValidator,
    FeatureStore,
    DataQuality,
    ValidationResult,
    NormalizationMethod
)

# Storage
from data.storage import (
    TimeSeriesDB,
    RedisCache,
    DataManager,
    StorageConfig,
    QueryBuilder,
    DataRetention
)

# Version
__version__ = "1.0.0"

# Exports publics
__all__ = [
    # Collectors
    "BinanceCollector",
    "InteractiveBrokersCollector",
    "MultiExchangeCollector",
    "BaseCollector",
    "CollectorState",
    "DataSource",
    
    # Processors
    "DataNormalizer",
    "DataValidator",
    "FeatureStore",
    "DataQuality",
    "ValidationResult",
    "NormalizationMethod",
    
    # Storage
    "TimeSeriesDB",
    "RedisCache",
    "DataManager",
    "StorageConfig",
    "QueryBuilder",
    "DataRetention"
]

# Configuration par défaut
DEFAULT_DATA_CONFIG = {
    "collectors": {
        "binance": {
            "rate_limit": 1200,  # requests per minute
            "websocket_streams": ["trade", "depth", "kline"],
            "rest_interval": 1,  # seconds
            "reconnect_interval": 5,
            "max_reconnect_attempts": 10
        },
        "ib": {
            "host": "127.0.0.1",
            "port": 7497,  # TWS paper trading
            "client_id": 1,
            "account": "DU1234567",
            "market_data_type": 3  # Delayed
        }
    },
    "storage": {
        "timescale": {
            "host": "localhost",
            "port": 5432,
            "database": "trading_data",
            "user": "trader",
            "chunk_time_interval": "1 day",
            "compression_after": "7 days",
            "retention_policy": "365 days"
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "decode_responses": True,
            "max_connections": 50,
            "socket_keepalive": True,
            "socket_keepalive_options": {}
        }
    },
    "processing": {
        "batch_size": 1000,
        "validation_threshold": 0.95,
        "outlier_std_threshold": 4,
        "missing_data_threshold": 0.05,
        "normalization_method": "z-score",
        "feature_window": 100
    },
    "retention": {
        "raw_data": "30 days",
        "aggregated_1m": "90 days",
        "aggregated_5m": "180 days",
        "aggregated_1h": "365 days",
        "aggregated_1d": "unlimited"
    }
}

# Types de données supportés
class DataType:
    """Types de données de marché"""
    TRADE = "trade"
    ORDERBOOK = "orderbook"
    KLINE = "kline"
    TICKER = "ticker"
    FUNDING = "funding"
    LIQUIDATION = "liquidation"
    OPEN_INTEREST = "open_interest"
    
    @classmethod
    def all(cls):
        return [
            cls.TRADE, cls.ORDERBOOK, cls.KLINE,
            cls.TICKER, cls.FUNDING, cls.LIQUIDATION,
            cls.OPEN_INTEREST
        ]


# Timeframes supportés
class TimeFrame:
    """Timeframes pour les données OHLCV"""
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
    
    @classmethod
    def get_seconds(cls, timeframe: str) -> int:
        """Convertit un timeframe en secondes"""
        mapping = {
            cls.TICK: 0,
            cls.SECOND_1: 1,
            cls.MINUTE_1: 60,
            cls.MINUTE_5: 300,
            cls.MINUTE_15: 900,
            cls.MINUTE_30: 1800,
            cls.HOUR_1: 3600,
            cls.HOUR_4: 14400,
            cls.DAY_1: 86400,
            cls.WEEK_1: 604800,
            cls.MONTH_1: 2592000
        }
        return mapping.get(timeframe, 60)


# Schema de données standard
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any


@dataclass
class MarketData:
    """Structure standard pour les données de marché"""
    timestamp: datetime
    symbol: str
    exchange: str
    price: Decimal
    volume: Decimal
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    bid_size: Optional[Decimal] = None
    ask_size: Optional[Decimal] = None
    open_interest: Optional[Decimal] = None
    funding_rate: Optional[Decimal] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> dict:
        """Convertit en dictionnaire"""
        data = {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'exchange': self.exchange,
            'price': float(self.price),
            'volume': float(self.volume)
        }
        
        # Ajouter les champs optionnels
        for field in ['bid', 'ask', 'bid_size', 'ask_size', 
                     'open_interest', 'funding_rate']:
            value = getattr(self, field)
            if value is not None:
                data[field] = float(value)
                
        if self.metadata:
            data['metadata'] = self.metadata
            
        return data


@dataclass
class OrderBookSnapshot:
    """Snapshot du carnet d'ordres"""
    timestamp: datetime
    symbol: str
    exchange: str
    bids: List[tuple[Decimal, Decimal]]  # [(price, size), ...]
    asks: List[tuple[Decimal, Decimal]]
    sequence_id: Optional[int] = None
    
    @property
    def best_bid(self) -> Optional[tuple[Decimal, Decimal]]:
        return self.bids[0] if self.bids else None
        
    @property
    def best_ask(self) -> Optional[tuple[Decimal, Decimal]]:
        return self.asks[0] if self.asks else None
        
    @property
    def spread(self) -> Optional[Decimal]:
        if self.best_bid and self.best_ask:
            return self.best_ask[0] - self.best_bid[0]
        return None


# Gestionnaire principal unifié
class UnifiedDataManager:
    """
    Gestionnaire unifié pour toutes les opérations de données.
    Orchestre collectors, processors et storage.
    """
    
    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_DATA_CONFIG, **(config or {})}
        
        # Initialiser les composants
        self.collectors = {}
        self.normalizer = DataNormalizer(self.config['processing'])
        self.validator = DataValidator(self.config['processing'])
        self.feature_store = FeatureStore(self.config)
        self.timeseries_db = TimeSeriesDB(self.config['storage']['timescale'])
        self.redis_cache = RedisCache(self.config['storage']['redis'])
        
        self._running = False
        
    async def start(self):
        """Démarre tous les composants"""
        await self.timeseries_db.connect()
        await self.redis_cache.connect()
        await self.feature_store.initialize()
        self._running = True
        
    async def stop(self):
        """Arrête tous les composants"""
        self._running = False
        for collector in self.collectors.values():
            await collector.stop()
        await self.timeseries_db.disconnect()
        await self.redis_cache.disconnect()
        
    def add_collector(self, name: str, collector: BaseCollector):
        """Ajoute un collector"""
        self.collectors[name] = collector
        
    async def start_collection(self, symbols: List[str], data_types: List[str] = None):
        """Démarre la collecte pour les symboles spécifiés"""
        if data_types is None:
            data_types = [DataType.TRADE, DataType.ORDERBOOK]
            
        for name, collector in self.collectors.items():
            await collector.subscribe(symbols, data_types)
            await collector.start()


# Instance globale optionnelle
_data_manager = None


def get_data_manager(config: dict = None) -> UnifiedDataManager:
    """Obtient l'instance globale du gestionnaire de données"""
    global _data_manager
    if _data_manager is None:
        _data_manager = UnifiedDataManager(config)
    return _data_manager


# Helpers pour requêtes courantes
async def get_latest_price(symbol: str, exchange: str = None) -> Optional[Decimal]:
    """Récupère le dernier prix d'un symbole"""
    dm = get_data_manager()
    cache_key = f"price:{exchange}:{symbol}" if exchange else f"price:*:{symbol}"
    
    # Essayer le cache d'abord
    cached = await dm.redis_cache.get(cache_key)
    if cached:
        return Decimal(cached)
        
    # Sinon requête DB
    query = f"""
        SELECT price FROM market_data 
        WHERE symbol = '{symbol}'
        {f"AND exchange = '{exchange}'" if exchange else ""}
        ORDER BY timestamp DESC
        LIMIT 1
    """
    result = await dm.timeseries_db.query(query)
    
    if result:
        price = Decimal(str(result[0]['price']))
        await dm.redis_cache.set(cache_key, str(price), expire=5)
        return price
        
    return None


async def get_ohlcv(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    exchange: str = None
) -> List[Dict[str, Any]]:
    """Récupère les données OHLCV"""
    dm = get_data_manager()
    
    # Construire la requête
    interval = TimeFrame.get_seconds(timeframe)
    query = f"""
        SELECT 
            time_bucket('{interval} seconds', timestamp) as time,
            first(price, timestamp) as open,
            max(price) as high,
            min(price) as low,
            last(price, timestamp) as close,
            sum(volume) as volume
        FROM market_data
        WHERE symbol = '{symbol}'
        AND timestamp >= '{start.isoformat()}'
        AND timestamp <= '{end.isoformat()}'
        {f"AND exchange = '{exchange}'" if exchange else ""}
        GROUP BY time
        ORDER BY time
    """
    
    return await dm.timeseries_db.query(query)