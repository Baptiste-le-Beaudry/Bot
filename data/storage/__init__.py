"""
Data Storage Sub-package
=======================

Gestion du stockage des données avec TimescaleDB et Redis.
"""

from data.storage.timeseries_db import (
    TimeSeriesDB,
    TimeScaleDBClient,
    QueryBuilder,
    DataPoint,
    TimeBucket
)

from data.storage.redis_cache import (
    RedisCache,
    CacheStrategy,
    CacheEntry,
    TTLPolicy,
    EvictionPolicy
)

from data.storage.data_manager import (
    DataManager,
    StorageBackend,
    DataRetention,
    CompressionPolicy,
    PartitionStrategy
)

# Types communs
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union


class StorageBackend(Enum):
    """Backends de stockage disponibles"""
    TIMESCALEDB = "timescaledb"
    REDIS = "redis"
    INFLUXDB = "influxdb"
    CLICKHOUSE = "clickhouse"
    S3 = "s3"


class DataRetention(Enum):
    """Politiques de rétention"""
    RAW_1_DAY = "1d"
    RAW_7_DAYS = "7d"
    RAW_30_DAYS = "30d"
    AGGREGATED_90_DAYS = "90d"
    AGGREGATED_1_YEAR = "365d"
    UNLIMITED = "unlimited"


@dataclass
class StorageConfig:
    """Configuration du stockage"""
    backend: StorageBackend
    connection_string: str
    pool_size: int = 10
    timeout: int = 30
    retry_attempts: int = 3
    compression: bool = True
    partitioning: bool = True


__all__ = [
    # TimeSeries DB
    "TimeSeriesDB",
    "TimeScaleDBClient",
    "QueryBuilder",
    "DataPoint",
    "TimeBucket",
    
    # Redis Cache
    "RedisCache",
    "CacheStrategy",
    "CacheEntry",
    "TTLPolicy",
    "EvictionPolicy",
    
    # Data Manager
    "DataManager",
    "StorageBackend",
    "DataRetention",
    "CompressionPolicy",
    "PartitionStrategy",
    
    # Config
    "StorageConfig"
]