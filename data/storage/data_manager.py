"""
Gestionnaire Principal des Données pour Robot de Trading Algorithmique IA
========================================================================

Ce module centralise et orchestre toute la gestion des données du système de trading.
Il fournit une interface unifiée pour accéder aux différentes sources de données,
gère le cache, l'archivage, la qualité des données et l'optimisation des performances.

Fonctionnalités:
- Interface unifiée pour TimescaleDB, Redis, InfluxDB
- Gestion intelligente du cache multi-niveaux
- Partitionnement et archivage automatique
- Compression et optimisation du stockage
- Validation et nettoyage des données
- Agrégation et resampling temps réel
- Backup et récupération automatique
- Monitoring de la qualité des données
- API d'accès aux données haute performance
- Support des requêtes distribuées

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import asyncio
import json
import gzip
import pickle
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union, AsyncIterator, Callable
import hashlib
import threading
import time
import numpy as np
import pandas as pd
from functools import lru_cache
import pyarrow as pa
import pyarrow.parquet as pq
import asyncpg
import redis
from redis import Redis
import influxdb_client
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS, ASYNCHRONOUS
import sqlalchemy
from sqlalchemy import create_engine, MetaData, Table, Column, select, and_, or_
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool, QueuePool
import aiofiles
import aioboto3
from motor.motor_asyncio import AsyncIOMotorClient

# Monitoring et logging
from prometheus_client import Counter, Gauge, Histogram, Summary
from rich.console import Console
from rich.progress import track

# Local imports
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.logger import get_structured_logger
from utils.decorators import retry_async, circuit_breaker, rate_limit
from utils.metrics import MetricsCollector
from monitoring.alerts import AlertManager, AlertSeverity
from config import get_config

console = Console()
logger = get_structured_logger(__name__)


class DataType(Enum):
    """Types de données gérées"""
    OHLCV = "ohlcv"
    TICK = "tick"
    ORDER_BOOK = "order_book"
    TRADES = "trades"
    INDICATORS = "indicators"
    FEATURES = "features"
    PREDICTIONS = "predictions"
    POSITIONS = "positions"
    ORDERS = "orders"
    PERFORMANCE = "performance"


class StorageBackend(Enum):
    """Backends de stockage disponibles"""
    TIMESCALEDB = "timescaledb"
    REDIS = "redis"
    INFLUXDB = "influxdb"
    PARQUET = "parquet"
    S3 = "s3"
    MONGODB = "mongodb"
    MEMORY = "memory"


class QueryPriority(Enum):
    """Priorité des requêtes"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BATCH = 5


@dataclass
class DataQuery:
    """Représentation d'une requête de données"""
    data_type: DataType
    symbol: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    aggregation: Optional[str] = None
    limit: Optional[int] = None
    offset: int = 0
    priority: QueryPriority = QueryPriority.NORMAL
    cache_ttl: Optional[int] = None
    
    def cache_key(self) -> str:
        """Génère une clé de cache pour la requête"""
        key_parts = [
            self.data_type.value,
            self.symbol or "all",
            str(self.start_time) if self.start_time else "none",
            str(self.end_time) if self.end_time else "none",
            json.dumps(self.filters, sort_keys=True),
            self.aggregation or "none",
            str(self.limit) if self.limit else "all"
        ]
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()


@dataclass
class DataPoint:
    """Point de données générique"""
    timestamp: datetime
    symbol: str
    data_type: DataType
    value: Union[float, Dict[str, Any], List[Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'data_type': self.data_type.value,
            'value': self.value,
            'metadata': self.metadata
        }


class DataManager:
    """Gestionnaire principal des données"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        
        # Configuration
        self.cache_size = self.config.get('data', {}).get('cache_size', 10000)
        self.batch_size = self.config.get('data', {}).get('batch_size', 1000)
        self.compression_enabled = self.config.get('data', {}).get('compression', True)
        self.archive_days = self.config.get('data', {}).get('archive_after_days', 30)
        
        # Backends de stockage
        self.backends: Dict[StorageBackend, Any] = {}
        self._init_backends()
        
        # Cache multi-niveaux
        self.memory_cache = OrderedDict()
        self.cache_stats = defaultdict(lambda: {'hits': 0, 'misses': 0})
        self._cache_lock = threading.RLock()
        
        # Buffer d'écriture
        self.write_buffer: Dict[str, List[DataPoint]] = defaultdict(list)
        self.buffer_lock = threading.Lock()
        
        # Executors pour opérations parallèles
        self.thread_executor = ThreadPoolExecutor(max_workers=4)
        self.process_executor = ProcessPoolExecutor(max_workers=2)
        
        # Métriques
        self._init_metrics()
        
        # État et monitoring
        self.is_running = False
        self.health_status = {}
        self.data_quality_scores = {}
        
        # Alert manager
        self.alert_manager = AlertManager()
        
        logger.info("Data manager initialized", 
                   backends=list(self.backends.keys()),
                   cache_size=self.cache_size)
    
    def _init_backends(self):
        """Initialise les backends de stockage"""
        # TimescaleDB (principal)
        try:
            self._init_timescaledb()
        except Exception as e:
            logger.error(f"Failed to initialize TimescaleDB: {str(e)}")
        
        # Redis (cache)
        try:
            self._init_redis()
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {str(e)}")
        
        # InfluxDB (métriques)
        try:
            self._init_influxdb()
        except Exception as e:
            logger.error(f"Failed to initialize InfluxDB: {str(e)}")
        
        # Stockage fichier local (backup)
        self._init_file_storage()
        
        # S3 (archivage)
        if self.config.get('data', {}).get('s3_enabled', False):
            try:
                self._init_s3()
            except Exception as e:
                logger.error(f"Failed to initialize S3: {str(e)}")
    
    def _init_timescaledb(self):
        """Initialise la connexion TimescaleDB"""
        db_config = self.config.get('database', {})
        
        # Connexion synchrone pour certaines opérations
        self.timescale_engine = create_engine(
            f"postgresql://{db_config['username']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}",
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        # Connexion asynchrone pour les opérations principales
        self.async_engine = create_async_engine(
            f"postgresql+asyncpg://{db_config['username']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}",
            pool_size=20,
            max_overflow=10
        )
        
        # Session factory
        self.async_session = sessionmaker(
            self.async_engine, 
            class_=AsyncSession, 
            expire_on_commit=False
        )
        
        # Métadonnées
        self.metadata = MetaData()
        
        # Créer les tables si nécessaire
        self._create_tables()
        
        self.backends[StorageBackend.TIMESCALEDB] = self.timescale_engine
        logger.info("TimescaleDB initialized successfully")
    
    def _init_redis(self):
        """Initialise la connexion Redis"""
        redis_config = self.config.get('redis', {})
        
        self.redis_client = redis.Redis(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            password=redis_config.get('password'),
            decode_responses=False,  # Pour stocker des données binaires
            max_connections=50
        )
        
        # Pool asynchrone
        self.redis_pool = redis.ConnectionPool(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            password=redis_config.get('password'),
            max_connections=50
        )
        
        # Test de connexion
        self.redis_client.ping()
        
        self.backends[StorageBackend.REDIS] = self.redis_client
        logger.info("Redis initialized successfully")
    
    def _init_influxdb(self):
        """Initialise la connexion InfluxDB"""
        influx_config = self.config.get('influxdb', {})
        
        if not influx_config:
            return
        
        self.influx_client = InfluxDBClient(
            url=influx_config.get('url', 'http://localhost:8086'),
            token=influx_config.get('token'),
            org=influx_config.get('org'),
            timeout=30_000
        )
        
        self.influx_write_api = self.influx_client.write_api(write_options=ASYNCHRONOUS)
        self.influx_query_api = self.influx_client.query_api()
        self.influx_bucket = influx_config.get('bucket', 'trading')
        
        self.backends[StorageBackend.INFLUXDB] = self.influx_client
        logger.info("InfluxDB initialized successfully")
    
    def _init_file_storage(self):
        """Initialise le stockage fichier local"""
        self.data_dir = Path(self.config.get('data', {}).get('data_dir', './data'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Répertoires par type de données
        for data_type in DataType:
            (self.data_dir / data_type.value).mkdir(exist_ok=True)
        
        self.backends[StorageBackend.PARQUET] = self.data_dir
        logger.info(f"File storage initialized at {self.data_dir}")
    
    def _init_s3(self):
        """Initialise la connexion S3 pour l'archivage"""
        s3_config = self.config.get('data', {}).get('s3', {})
        
        self.s3_bucket = s3_config.get('bucket')
        self.s3_session = aioboto3.Session(
            aws_access_key_id=s3_config.get('access_key'),
            aws_secret_access_key=s3_config.get('secret_key'),
            region_name=s3_config.get('region', 'us-east-1')
        )
        
        self.backends[StorageBackend.S3] = self.s3_session
        logger.info(f"S3 initialized with bucket: {self.s3_bucket}")
    
    def _init_metrics(self):
        """Initialise les métriques Prometheus"""
        self.metrics = {
            'queries_total': Counter('data_queries_total', 'Total data queries', ['data_type', 'backend']),
            'query_duration': Histogram('data_query_duration_seconds', 'Query duration', ['data_type']),
            'cache_hits': Counter('data_cache_hits_total', 'Cache hits'),
            'cache_misses': Counter('data_cache_misses_total', 'Cache misses'),
            'writes_total': Counter('data_writes_total', 'Total data writes', ['data_type', 'backend']),
            'write_errors': Counter('data_write_errors_total', 'Write errors', ['backend']),
            'data_points_stored': Gauge('data_points_stored', 'Total data points stored', ['data_type']),
            'storage_size_bytes': Gauge('data_storage_size_bytes', 'Storage size in bytes', ['backend']),
            'buffer_size': Gauge('data_buffer_size', 'Current buffer size'),
            'compression_ratio': Gauge('data_compression_ratio', 'Compression ratio achieved')
        }
    
    def _create_tables(self):
        """Crée les tables dans TimescaleDB"""
        with self.timescale_engine.connect() as conn:
            # Table OHLCV
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    open DECIMAL(20,8) NOT NULL,
                    high DECIMAL(20,8) NOT NULL,
                    low DECIMAL(20,8) NOT NULL,
                    close DECIMAL(20,8) NOT NULL,
                    volume DECIMAL(20,8) NOT NULL,
                    trades INTEGER,
                    vwap DECIMAL(20,8),
                    UNIQUE(time, symbol)
                );
            """)
            
            # Convertir en hypertable
            try:
                conn.execute("""
                    SELECT create_hypertable('ohlcv', 'time', 
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE
                    );
                """)
            except Exception:
                pass  # Table déjà hypertable
            
            # Table Trades
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    trade_id VARCHAR(100) UNIQUE,
                    price DECIMAL(20,8) NOT NULL,
                    quantity DECIMAL(20,8) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    maker BOOLEAN,
                    fee DECIMAL(20,8)
                );
            """)
            
            # Table Order Book
            conn.execute("""
                CREATE TABLE IF NOT EXISTS order_book (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    side VARCHAR(10) NOT NULL,
                    price DECIMAL(20,8) NOT NULL,
                    quantity DECIMAL(20,8) NOT NULL,
                    level INTEGER NOT NULL
                );
            """)
            
            # Table Features
            conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    time TIMESTAMPTZ NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    feature_name VARCHAR(100) NOT NULL,
                    value DOUBLE PRECISION NOT NULL,
                    UNIQUE(time, symbol, feature_name)
                );
            """)
            
            # Créer les index
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_time ON ohlcv (symbol, time DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades (symbol, time DESC);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_features_symbol_feature ON features (symbol, feature_name, time DESC);")
            
            # Politiques de compression
            conn.execute("""
                SELECT add_compression_policy('ohlcv', 
                    compress_after => INTERVAL '7 days',
                    if_not_exists => TRUE
                );
            """)
            
            logger.info("Database tables created/verified")
    
    async def start(self):
        """Démarre le gestionnaire de données"""
        self.is_running = True
        
        # Démarrer les tâches de maintenance
        tasks = [
            asyncio.create_task(self._flush_buffer_task()),
            asyncio.create_task(self._cache_cleanup_task()),
            asyncio.create_task(self._archive_task()),
            asyncio.create_task(self._health_check_task()),
            asyncio.create_task(self._data_quality_task())
        ]
        
        logger.info("Data manager started")
        
        # Attendre que toutes les tâches se terminent
        await asyncio.gather(*tasks)
    
    async def stop(self):
        """Arrête le gestionnaire de données"""
        self.is_running = False
        
        # Flush final du buffer
        await self._flush_all_buffers()
        
        # Fermer les connexions
        if StorageBackend.INFLUXDB in self.backends:
            self.influx_client.close()
        
        if hasattr(self, 'async_engine'):
            await self.async_engine.dispose()
        
        # Arrêter les executors
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        logger.info("Data manager stopped")
    
    async def store(self, data_points: Union[DataPoint, List[DataPoint]], 
                   priority: QueryPriority = QueryPriority.NORMAL) -> bool:
        """
        Stocke des points de données
        
        Args:
            data_points: Point(s) de données à stocker
            priority: Priorité du stockage
            
        Returns:
            Success status
        """
        if not isinstance(data_points, list):
            data_points = [data_points]
        
        if not data_points:
            return True
        
        try:
            # Validation des données
            valid_points = await self._validate_data_points(data_points)
            
            if not valid_points:
                logger.warning("No valid data points to store")
                return False
            
            # Ajouter au buffer selon la priorité
            if priority in [QueryPriority.CRITICAL, QueryPriority.HIGH]:
                # Écriture immédiate pour haute priorité
                await self._write_to_backends(valid_points)
            else:
                # Ajouter au buffer pour écriture batch
                with self.buffer_lock:
                    for point in valid_points:
                        key = f"{point.data_type.value}:{point.symbol}"
                        self.write_buffer[key].append(point)
                        
                        # Flush si buffer plein
                        if len(self.write_buffer[key]) >= self.batch_size:
                            asyncio.create_task(self._flush_buffer(key))
                
                # Mettre à jour les métriques
                self.metrics['buffer_size'].set(sum(len(v) for v in self.write_buffer.values()))
            
            # Mettre à jour le cache
            for point in valid_points:
                self._update_cache(point)
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing data: {str(e)}")
            self.metrics['write_errors'].labels(backend='all').inc()
            return False
    
    async def query(self, query: DataQuery) -> pd.DataFrame:
        """
        Exécute une requête de données
        
        Args:
            query: Requête à exécuter
            
        Returns:
            DataFrame avec les résultats
        """
        start_time = time.time()
        
        try:
            # Vérifier le cache
            cache_key = query.cache_key()
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result is not None:
                self.metrics['cache_hits'].inc()
                return cached_result
            
            self.metrics['cache_misses'].inc()
            
            # Déterminer le backend optimal
            backend = self._select_backend(query)
            
            # Exécuter la requête
            if backend == StorageBackend.TIMESCALEDB:
                result = await self._query_timescaledb(query)
            elif backend == StorageBackend.REDIS:
                result = await self._query_redis(query)
            elif backend == StorageBackend.INFLUXDB:
                result = await self._query_influxdb(query)
            elif backend == StorageBackend.PARQUET:
                result = await self._query_parquet(query)
            else:
                raise ValueError(f"Unsupported backend: {backend}")
            
            # Appliquer les filtres et transformations
            if result is not None and not result.empty:
                result = self._apply_query_filters(result, query)
                
                # Mettre en cache si demandé
                if query.cache_ttl:
                    self._put_in_cache(cache_key, result, query.cache_ttl)
            
            # Métriques
            duration = time.time() - start_time
            self.metrics['query_duration'].labels(data_type=query.data_type.value).observe(duration)
            self.metrics['queries_total'].labels(
                data_type=query.data_type.value, 
                backend=backend.value
            ).inc()
            
            return result if result is not None else pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Query error: {str(e)}", query=query)
            raise
    
    async def _query_timescaledb(self, query: DataQuery) -> pd.DataFrame:
        """Exécute une requête sur TimescaleDB"""
        async with self.async_session() as session:
            # Construire la requête SQL
            sql = self._build_sql_query(query)
            
            # Exécuter
            result = await session.execute(sql)
            rows = result.fetchall()
            
            if not rows:
                return pd.DataFrame()
            
            # Convertir en DataFrame
            df = pd.DataFrame(rows, columns=result.keys())
            
            # Parser les timestamps
            if 'time' in df.columns:
                df['time'] = pd.to_datetime(df['time'])
                df.set_index('time', inplace=True)
            
            return df
    
    def _build_sql_query(self, query: DataQuery) -> str:
        """Construit une requête SQL à partir d'un DataQuery"""
        # Table selon le type de données
        table_map = {
            DataType.OHLCV: "ohlcv",
            DataType.TRADES: "trades",
            DataType.ORDER_BOOK: "order_book",
            DataType.FEATURES: "features"
        }
        
        table = table_map.get(query.data_type, "ohlcv")
        
        # Colonnes à sélectionner
        if query.data_type == DataType.OHLCV:
            columns = "time, symbol, open, high, low, close, volume"
        elif query.data_type == DataType.TRADES:
            columns = "time, symbol, trade_id, price, quantity, side"
        else:
            columns = "*"
        
        # Construction de la requête
        sql_parts = [f"SELECT {columns} FROM {table}"]
        
        # Conditions WHERE
        conditions = []
        
        if query.symbol:
            conditions.append(f"symbol = '{query.symbol}'")
        
        if query.start_time:
            conditions.append(f"time >= '{query.start_time.isoformat()}'")
        
        if query.end_time:
            conditions.append(f"time <= '{query.end_time.isoformat()}'")
        
        # Filtres additionnels
        for key, value in query.filters.items():
            if isinstance(value, str):
                conditions.append(f"{key} = '{value}'")
            else:
                conditions.append(f"{key} = {value}")
        
        if conditions:
            sql_parts.append(f"WHERE {' AND '.join(conditions)}")
        
        # Agrégation
        if query.aggregation:
            if query.aggregation == "1h":
                sql_parts.append("GROUP BY time_bucket('1 hour', time), symbol")
            elif query.aggregation == "1d":
                sql_parts.append("GROUP BY time_bucket('1 day', time), symbol")
        
        # Ordre et limite
        sql_parts.append("ORDER BY time DESC")
        
        if query.limit:
            sql_parts.append(f"LIMIT {query.limit}")
            
        if query.offset:
            sql_parts.append(f"OFFSET {query.offset}")
        
        return " ".join(sql_parts)
    
    async def _query_redis(self, query: DataQuery) -> pd.DataFrame:
        """Exécute une requête sur Redis"""
        try:
            # Clé Redis selon le type et symbole
            key_pattern = f"{query.data_type.value}:{query.symbol or '*'}"
            
            # Scanner les clés
            keys = []
            for key in self.redis_client.scan_iter(match=key_pattern, count=1000):
                keys.append(key)
            
            if not keys:
                return pd.DataFrame()
            
            # Pipeline pour récupérer les valeurs
            pipe = self.redis_client.pipeline()
            for key in keys[:query.limit or 1000]:  # Limiter
                pipe.get(key)
            
            values = pipe.execute()
            
            # Désérialiser et convertir en DataFrame
            data = []
            for key, value in zip(keys, values):
                if value:
                    try:
                        if self.compression_enabled:
                            value = gzip.decompress(value)
                        obj = pickle.loads(value)
                        data.append(obj)
                    except Exception as e:
                        logger.warning(f"Failed to deserialize Redis value: {e}")
            
            if not data:
                return pd.DataFrame()
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Redis query error: {str(e)}")
            return pd.DataFrame()
    
    async def _query_influxdb(self, query: DataQuery) -> pd.DataFrame:
        """Exécute une requête sur InfluxDB"""
        if StorageBackend.INFLUXDB not in self.backends:
            return pd.DataFrame()
        
        try:
            # Construire la requête Flux
            flux_query = self._build_flux_query(query)
            
            # Exécuter
            result = self.influx_query_api.query_data_frame(flux_query)
            
            if isinstance(result, list):
                result = pd.concat(result, ignore_index=True)
            
            return result
            
        except Exception as e:
            logger.error(f"InfluxDB query error: {str(e)}")
            return pd.DataFrame()
    
    def _build_flux_query(self, query: DataQuery) -> str:
        """Construit une requête Flux pour InfluxDB"""
        parts = [
            f'from(bucket: "{self.influx_bucket}")',
            f'|> range(start: {query.start_time.isoformat() if query.start_time else "-30d"})'
        ]
        
        if query.end_time:
            parts.append(f'|> range(stop: {query.end_time.isoformat()})')
        
        if query.data_type:
            parts.append(f'|> filter(fn: (r) => r["_measurement"] == "{query.data_type.value}")')
        
        if query.symbol:
            parts.append(f'|> filter(fn: (r) => r["symbol"] == "{query.symbol}")')
        
        if query.aggregation:
            window = query.aggregation
            parts.append(f'|> aggregateWindow(every: {window}, fn: mean)')
        
        if query.limit:
            parts.append(f'|> limit(n: {query.limit})')
        
        return '\n'.join(parts)
    
    async def _query_parquet(self, query: DataQuery) -> pd.DataFrame:
        """Exécute une requête sur les fichiers Parquet"""
        try:
            # Chemin des fichiers
            data_path = self.data_dir / query.data_type.value
            
            if query.symbol:
                file_pattern = f"{query.symbol}_*.parquet"
            else:
                file_pattern = "*.parquet"
            
            # Lister les fichiers
            files = list(data_path.glob(file_pattern))
            
            if not files:
                return pd.DataFrame()
            
            # Filtrer par date si nécessaire
            if query.start_time or query.end_time:
                filtered_files = []
                for file in files:
                    # Extraire la date du nom de fichier
                    # Format: SYMBOL_YYYYMMDD.parquet
                    try:
                        date_str = file.stem.split('_')[-1]
                        file_date = datetime.strptime(date_str, '%Y%m%d')
                        
                        if query.start_time and file_date.date() < query.start_time.date():
                            continue
                        if query.end_time and file_date.date() > query.end_time.date():
                            continue
                            
                        filtered_files.append(file)
                    except Exception:
                        pass
                
                files = filtered_files
            
            if not files:
                return pd.DataFrame()
            
            # Lire les fichiers
            dfs = []
            for file in files[:10]:  # Limiter pour la performance
                df = pq.read_table(file).to_pandas()
                dfs.append(df)
            
            # Combiner
            result = pd.concat(dfs, ignore_index=True)
            
            # Appliquer les filtres temporels plus fins
            if 'timestamp' in result.columns:
                result['timestamp'] = pd.to_datetime(result['timestamp'])
                
                if query.start_time:
                    result = result[result['timestamp'] >= query.start_time]
                if query.end_time:
                    result = result[result['timestamp'] <= query.end_time]
            
            return result
            
        except Exception as e:
            logger.error(f"Parquet query error: {str(e)}")
            return pd.DataFrame()
    
    def _select_backend(self, query: DataQuery) -> StorageBackend:
        """Sélectionne le backend optimal pour une requête"""
        # Logique de sélection basée sur le type et la période
        
        # Données très récentes (< 1 heure) -> Redis
        if query.start_time and (datetime.now(timezone.utc) - query.start_time).total_seconds() < 3600:
            if StorageBackend.REDIS in self.backends:
                return StorageBackend.REDIS
        
        # Métriques et performances -> InfluxDB
        if query.data_type in [DataType.PERFORMANCE, DataType.INDICATORS]:
            if StorageBackend.INFLUXDB in self.backends:
                return StorageBackend.INFLUXDB
        
        # Données archivées (> 30 jours) -> Parquet/S3
        if query.start_time and (datetime.now(timezone.utc) - query.start_time).days > self.archive_days:
            if StorageBackend.PARQUET in self.backends:
                return StorageBackend.PARQUET
        
        # Par défaut -> TimescaleDB
        return StorageBackend.TIMESCALEDB
    
    def _apply_query_filters(self, df: pd.DataFrame, query: DataQuery) -> pd.DataFrame:
        """Applique les filtres et transformations sur le DataFrame"""
        if df.empty:
            return df
        
        # Filtres additionnels
        for key, value in query.filters.items():
            if key in df.columns:
                df = df[df[key] == value]
        
        # Limite et offset
        if query.offset:
            df = df.iloc[query.offset:]
        
        if query.limit:
            df = df.iloc[:query.limit]
        
        return df
    
    def _get_from_cache(self, key: str) -> Optional[pd.DataFrame]:
        """Récupère depuis le cache mémoire"""
        with self._cache_lock:
            if key in self.memory_cache:
                # Déplacer en fin (LRU)
                self.memory_cache.move_to_end(key)
                self.cache_stats[key]['hits'] += 1
                
                # Vérifier l'expiration
                data, expiry = self.memory_cache[key]
                if expiry and datetime.now() > expiry:
                    del self.memory_cache[key]
                    return None
                
                return data.copy()
            
            self.cache_stats[key]['misses'] += 1
            return None
    
    def _put_in_cache(self, key: str, data: pd.DataFrame, ttl_seconds: int):
        """Met en cache avec TTL"""
        with self._cache_lock:
            # Limiter la taille du cache
            if len(self.memory_cache) >= self.cache_size:
                # Supprimer le plus ancien (LRU)
                self.memory_cache.popitem(last=False)
            
            expiry = datetime.now() + timedelta(seconds=ttl_seconds) if ttl_seconds else None
            self.memory_cache[key] = (data.copy(), expiry)
    
    def _update_cache(self, data_point: DataPoint):
        """Met à jour le cache avec un nouveau point"""
        # Mise à jour partielle du cache pour les données temps réel
        cache_key = f"realtime:{data_point.data_type.value}:{data_point.symbol}"
        
        with self._cache_lock:
            if cache_key not in self.memory_cache:
                # Créer un nouveau DataFrame
                df = pd.DataFrame([data_point.to_dict()])
                self.memory_cache[cache_key] = (df, None)
            else:
                # Ajouter au DataFrame existant
                df, expiry = self.memory_cache[cache_key]
                new_row = pd.DataFrame([data_point.to_dict()])
                df = pd.concat([df, new_row], ignore_index=True)
                
                # Limiter la taille
                if len(df) > 1000:
                    df = df.iloc[-1000:]
                
                self.memory_cache[cache_key] = (df, expiry)
    
    async def _validate_data_points(self, data_points: List[DataPoint]) -> List[DataPoint]:
        """Valide et nettoie les points de données"""
        valid_points = []
        
        for point in data_points:
            try:
                # Validation basique
                if not point.timestamp or not point.symbol:
                    logger.warning("Invalid data point: missing timestamp or symbol")
                    continue
                
                # Validation spécifique au type
                if point.data_type == DataType.OHLCV:
                    if not self._validate_ohlcv(point.value):
                        continue
                elif point.data_type == DataType.TRADES:
                    if not self._validate_trade(point.value):
                        continue
                
                # Normalisation
                point = self._normalize_data_point(point)
                
                valid_points.append(point)
                
            except Exception as e:
                logger.warning(f"Data validation error: {str(e)}")
        
        return valid_points
    
    def _validate_ohlcv(self, data: Dict[str, Any]) -> bool:
        """Valide les données OHLCV"""
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        
        for field in required_fields:
            if field not in data or data[field] is None:
                return False
            
            # Vérifier que ce sont des nombres
            try:
                float(data[field])
            except (TypeError, ValueError):
                return False
        
        # Validation logique
        if data['high'] < data['low']:
            return False
        if data['high'] < data['open'] or data['high'] < data['close']:
            return False
        if data['low'] > data['open'] or data['low'] > data['close']:
            return False
        if data['volume'] < 0:
            return False
        
        return True
    
    def _validate_trade(self, data: Dict[str, Any]) -> bool:
        """Valide les données de trade"""
        required_fields = ['price', 'quantity', 'side']
        
        for field in required_fields:
            if field not in data:
                return False
        
        # Validations
        if data['price'] <= 0 or data['quantity'] <= 0:
            return False
        if data['side'] not in ['BUY', 'SELL', 'buy', 'sell']:
            return False
        
        return True
    
    def _normalize_data_point(self, point: DataPoint) -> DataPoint:
        """Normalise un point de données"""
        # Normaliser les timestamps en UTC
        if point.timestamp.tzinfo is None:
            point.timestamp = point.timestamp.replace(tzinfo=timezone.utc)
        else:
            point.timestamp = point.timestamp.astimezone(timezone.utc)
        
        # Normaliser les symboles
        point.symbol = point.symbol.upper()
        
        # Normaliser les valeurs selon le type
        if point.data_type == DataType.TRADES and isinstance(point.value, dict):
            if 'side' in point.value:
                point.value['side'] = point.value['side'].upper()
        
        return point
    
    async def _write_to_backends(self, data_points: List[DataPoint]):
        """Écrit les données vers les backends appropriés"""
        # Grouper par type
        grouped = defaultdict(list)
        for point in data_points:
            grouped[point.data_type].append(point)
        
        # Écrire en parallèle vers différents backends
        tasks = []
        
        for data_type, points in grouped.items():
            # TimescaleDB pour données historiques
            if StorageBackend.TIMESCALEDB in self.backends:
                tasks.append(self._write_to_timescaledb(points))
            
            # Redis pour cache temps réel
            if StorageBackend.REDIS in self.backends:
                tasks.append(self._write_to_redis(points))
            
            # InfluxDB pour métriques
            if StorageBackend.INFLUXDB in self.backends and data_type in [DataType.PERFORMANCE, DataType.INDICATORS]:
                tasks.append(self._write_to_influxdb(points))
        
        # Attendre toutes les écritures
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Vérifier les erreurs
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Write error in task {i}: {str(result)}")
                self.metrics['write_errors'].labels(backend='unknown').inc()
    
    async def _write_to_timescaledb(self, data_points: List[DataPoint]):
        """Écrit dans TimescaleDB"""
        if not data_points:
            return
        
        try:
            # Grouper par type de données
            by_type = defaultdict(list)
            for point in data_points:
                by_type[point.data_type].append(point)
            
            async with self.async_session() as session:
                for data_type, points in by_type.items():
                    if data_type == DataType.OHLCV:
                        await self._insert_ohlcv(session, points)
                    elif data_type == DataType.TRADES:
                        await self._insert_trades(session, points)
                    elif data_type == DataType.FEATURES:
                        await self._insert_features(session, points)
                
                await session.commit()
                
            self.metrics['writes_total'].labels(
                data_type=data_points[0].data_type.value,
                backend='timescaledb'
            ).inc(len(data_points))
            
        except Exception as e:
            logger.error(f"TimescaleDB write error: {str(e)}")
            self.metrics['write_errors'].labels(backend='timescaledb').inc()
            raise
    
    async def _insert_ohlcv(self, session, points: List[DataPoint]):
        """Insère des données OHLCV"""
        values = []
        for point in points:
            data = point.value
            values.append({
                'time': point.timestamp,
                'symbol': point.symbol,
                'open': float(data['open']),
                'high': float(data['high']),
                'low': float(data['low']),
                'close': float(data['close']),
                'volume': float(data['volume']),
                'trades': data.get('trades'),
                'vwap': data.get('vwap')
            })
        
        # Insert avec gestion des conflits
        stmt = """
            INSERT INTO ohlcv (time, symbol, open, high, low, close, volume, trades, vwap)
            VALUES (:time, :symbol, :open, :high, :low, :close, :volume, :trades, :vwap)
            ON CONFLICT (time, symbol) DO UPDATE SET
                open = EXCLUDED.open,
                high = EXCLUDED.high,
                low = EXCLUDED.low,
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                trades = EXCLUDED.trades,
                vwap = EXCLUDED.vwap
        """
        
        await session.execute(stmt, values)
    
    async def _write_to_redis(self, data_points: List[DataPoint]):
        """Écrit dans Redis pour cache temps réel"""
        if not data_points:
            return
        
        try:
            pipe = self.redis_client.pipeline()
            
            for point in data_points:
                # Clé unique
                key = f"{point.data_type.value}:{point.symbol}:{point.timestamp.timestamp()}"
                
                # Sérialiser
                value = pickle.dumps(point.to_dict())
                
                # Compresser si activé
                if self.compression_enabled:
                    value = gzip.compress(value)
                
                # Stocker avec expiration
                ttl = 3600 * 24  # 24 heures par défaut
                pipe.setex(key, ttl, value)
                
                # Ajouter aux sorted sets pour requêtes par temps
                score = point.timestamp.timestamp()
                pipe.zadd(f"{point.data_type.value}:{point.symbol}:timeline", {key: score})
                
                # Maintenir la taille du sorted set
                pipe.zremrangebyrank(f"{point.data_type.value}:{point.symbol}:timeline", 0, -10000)
            
            pipe.execute()
            
            self.metrics['writes_total'].labels(
                data_type=data_points[0].data_type.value,
                backend='redis'
            ).inc(len(data_points))
            
        except Exception as e:
            logger.error(f"Redis write error: {str(e)}")
            self.metrics['write_errors'].labels(backend='redis').inc()
    
    async def _write_to_influxdb(self, data_points: List[DataPoint]):
        """Écrit dans InfluxDB"""
        if not data_points or StorageBackend.INFLUXDB not in self.backends:
            return
        
        try:
            points = []
            
            for data_point in data_points:
                point = Point(data_point.data_type.value) \
                    .tag("symbol", data_point.symbol) \
                    .time(data_point.timestamp, WritePrecision.NS)
                
                # Ajouter les fields selon le type
                if isinstance(data_point.value, dict):
                    for field, value in data_point.value.items():
                        if isinstance(value, (int, float)):
                            point.field(field, value)
                else:
                    point.field("value", data_point.value)
                
                points.append(point)
            
            # Écrire en batch
            self.influx_write_api.write(
                bucket=self.influx_bucket,
                record=points
            )
            
            self.metrics['writes_total'].labels(
                data_type=data_points[0].data_type.value,
                backend='influxdb'
            ).inc(len(data_points))
            
        except Exception as e:
            logger.error(f"InfluxDB write error: {str(e)}")
            self.metrics['write_errors'].labels(backend='influxdb').inc()
    
    async def _flush_buffer_task(self):
        """Tâche de flush périodique du buffer"""
        while self.is_running:
            try:
                await asyncio.sleep(5)  # Flush toutes les 5 secondes
                
                # Copier les clés pour éviter les modifications pendant l'itération
                with self.buffer_lock:
                    keys_to_flush = list(self.write_buffer.keys())
                
                for key in keys_to_flush:
                    if self.write_buffer[key]:
                        await self._flush_buffer(key)
                        
            except Exception as e:
                logger.error(f"Buffer flush error: {str(e)}")
    
    async def _flush_buffer(self, key: str):
        """Flush un buffer spécifique"""
        with self.buffer_lock:
            if key not in self.write_buffer or not self.write_buffer[key]:
                return
            
            points = self.write_buffer[key]
            self.write_buffer[key] = []
        
        # Écrire les points
        await self._write_to_backends(points)
        
        logger.debug(f"Flushed {len(points)} points for {key}")
    
    async def _flush_all_buffers(self):
        """Flush tous les buffers"""
        with self.buffer_lock:
            all_keys = list(self.write_buffer.keys())
        
        for key in all_keys:
            await self._flush_buffer(key)
    
    async def _cache_cleanup_task(self):
        """Nettoie périodiquement le cache"""
        while self.is_running:
            try:
                await asyncio.sleep(300)  # Toutes les 5 minutes
                
                with self._cache_lock:
                    # Supprimer les entrées expirées
                    expired_keys = []
                    now = datetime.now()
                    
                    for key, (data, expiry) in self.memory_cache.items():
                        if expiry and now > expiry:
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        del self.memory_cache[key]
                    
                    if expired_keys:
                        logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
                    
                    # Log des statistiques de cache
                    total_hits = sum(stats['hits'] for stats in self.cache_stats.values())
                    total_misses = sum(stats['misses'] for stats in self.cache_stats.values())
                    
                    if total_hits + total_misses > 0:
                        hit_rate = total_hits / (total_hits + total_misses)
                        logger.info(f"Cache hit rate: {hit_rate:.2%}")
                        
            except Exception as e:
                logger.error(f"Cache cleanup error: {str(e)}")
    
    async def _archive_task(self):
        """Archive les anciennes données"""
        while self.is_running:
            try:
                # Archivage quotidien
                await asyncio.sleep(86400)  # 24 heures
                
                await self._archive_old_data()
                
            except Exception as e:
                logger.error(f"Archive task error: {str(e)}")
    
    async def _archive_old_data(self):
        """Archive les données plus anciennes que archive_days"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.archive_days)
        
        logger.info(f"Starting data archival for data older than {cutoff_date}")
        
        try:
            # Pour chaque type de données
            for data_type in [DataType.OHLCV, DataType.TRADES]:
                # Requête pour obtenir les anciennes données
                query = DataQuery(
                    data_type=data_type,
                    end_time=cutoff_date,
                    limit=100000  # Batch size
                )
                
                while True:
                    # Récupérer un batch
                    df = await self.query(query)
                    
                    if df.empty:
                        break
                    
                    # Sauvegarder en Parquet
                    await self._save_to_parquet(df, data_type)
                    
                    # Si S3 est configuré, uploader
                    if StorageBackend.S3 in self.backends:
                        await self._upload_to_s3(df, data_type)
                    
                    # Supprimer de TimescaleDB
                    await self._delete_from_timescaledb(data_type, cutoff_date)
                    
                    # Si moins d'un batch complet, on a fini
                    if len(df) < query.limit:
                        break
                    
                    # Mettre à jour l'offset pour le prochain batch
                    query.offset += query.limit
            
            logger.info("Data archival completed")
            
        except Exception as e:
            logger.error(f"Archive error: {str(e)}")
            await self.alert_manager.send_alert(
                severity=AlertSeverity.ERROR,
                title="Data Archive Failed",
                message=str(e)
            )
    
    async def _save_to_parquet(self, df: pd.DataFrame, data_type: DataType):
        """Sauvegarde un DataFrame en Parquet"""
        if df.empty:
            return
        
        # Déterminer le nom de fichier
        date_str = df.index.min().strftime('%Y%m%d')
        symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'all'
        
        filename = f"{symbol}_{date_str}.parquet"
        filepath = self.data_dir / data_type.value / filename
        
        # Sauvegarder avec compression
        table = pa.Table.from_pandas(df)
        pq.write_table(
            table, 
            filepath,
            compression='snappy',
            use_dictionary=True,
            compression_level=None
        )
        
        logger.info(f"Saved {len(df)} rows to {filepath}")
        
        # Calculer le ratio de compression
        original_size = df.memory_usage(deep=True).sum()
        compressed_size = filepath.stat().st_size
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        self.metrics['compression_ratio'].set(compression_ratio)
    
    async def _upload_to_s3(self, df: pd.DataFrame, data_type: DataType):
        """Upload vers S3"""
        if StorageBackend.S3 not in self.backends:
            return
        
        try:
            # Convertir en Parquet en mémoire
            table = pa.Table.from_pandas(df)
            buf = pa.BufferOutputStream()
            pq.write_table(table, buf, compression='snappy')
            
            # Upload vers S3
            date_str = df.index.min().strftime('%Y/%m/%d')
            symbol = df['symbol'].iloc[0] if 'symbol' in df.columns else 'all'
            key = f"data/{data_type.value}/{date_str}/{symbol}.parquet"
            
            async with self.s3_session.client('s3') as s3:
                await s3.put_object(
                    Bucket=self.s3_bucket,
                    Key=key,
                    Body=buf.getvalue().to_pybytes()
                )
            
            logger.info(f"Uploaded to S3: {key}")
            
        except Exception as e:
            logger.error(f"S3 upload error: {str(e)}")
    
    async def _delete_from_timescaledb(self, data_type: DataType, cutoff_date: datetime):
        """Supprime les anciennes données de TimescaleDB"""
        table_map = {
            DataType.OHLCV: "ohlcv",
            DataType.TRADES: "trades"
        }
        
        table = table_map.get(data_type)
        if not table:
            return
        
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    f"DELETE FROM {table} WHERE time < :cutoff",
                    {"cutoff": cutoff_date}
                )
                
                deleted = result.rowcount
                await session.commit()
                
                if deleted > 0:
                    logger.info(f"Deleted {deleted} old records from {table}")
                    
        except Exception as e:
            logger.error(f"Delete error: {str(e)}")
    
    async def _health_check_task(self):
        """Vérifie périodiquement la santé des backends"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Toutes les minutes
                
                health_status = {}
                
                # TimescaleDB
                if StorageBackend.TIMESCALEDB in self.backends:
                    health_status['timescaledb'] = await self._check_timescaledb_health()
                
                # Redis
                if StorageBackend.REDIS in self.backends:
                    health_status['redis'] = await self._check_redis_health()
                
                # InfluxDB
                if StorageBackend.INFLUXDB in self.backends:
                    health_status['influxdb'] = await self._check_influxdb_health()
                
                self.health_status = health_status
                
                # Alerter si problème
                for backend, status in health_status.items():
                    if not status['healthy']:
                        await self.alert_manager.send_alert(
                            severity=AlertSeverity.ERROR,
                            title=f"Data Backend Unhealthy: {backend}",
                            message=status.get('error', 'Unknown error')
                        )
                        
            except Exception as e:
                logger.error(f"Health check error: {str(e)}")
    
    async def _check_timescaledb_health(self) -> Dict[str, Any]:
        """Vérifie la santé de TimescaleDB"""
        try:
            async with self.async_session() as session:
                result = await session.execute("SELECT 1")
                result.scalar()
                
                # Vérifier l'espace disque
                disk_result = await session.execute("""
                    SELECT 
                        pg_database_size(current_database()) as db_size,
                        pg_size_pretty(pg_database_size(current_database())) as db_size_pretty
                """)
                
                row = disk_result.first()
                
                return {
                    'healthy': True,
                    'db_size': row.db_size,
                    'db_size_pretty': row.db_size_pretty
                }
                
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    async def _check_redis_health(self) -> Dict[str, Any]:
        """Vérifie la santé de Redis"""
        try:
            info = self.redis_client.info()
            
            return {
                'healthy': True,
                'used_memory_mb': info['used_memory'] / (1024 * 1024),
                'connected_clients': info['connected_clients'],
                'total_commands_processed': info['total_commands_processed']
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    async def _check_influxdb_health(self) -> Dict[str, Any]:
        """Vérifie la santé d'InfluxDB"""
        try:
            # Simple ping
            self.influx_client.ping()
            
            return {
                'healthy': True
            }
            
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    async def _data_quality_task(self):
        """Vérifie périodiquement la qualité des données"""
        while self.is_running:
            try:
                await asyncio.sleep(3600)  # Toutes les heures
                
                quality_scores = {}
                
                # Vérifier chaque type de données
                for data_type in [DataType.OHLCV, DataType.TRADES]:
                    score = await self._check_data_quality(data_type)
                    quality_scores[data_type.value] = score
                    
                    if score < 0.95:  # Seuil de qualité
                        await self.alert_manager.send_alert(
                            severity=AlertSeverity.WARNING,
                            title=f"Data Quality Issue: {data_type.value}",
                            message=f"Quality score: {score:.2%}"
                        )
                
                self.data_quality_scores = quality_scores
                
            except Exception as e:
                logger.error(f"Data quality check error: {str(e)}")
    
    async def _check_data_quality(self, data_type: DataType) -> float:
        """Vérifie la qualité des données pour un type"""
        try:
            # Requête d'échantillon
            query = DataQuery(
                data_type=data_type,
                start_time=datetime.now(timezone.utc) - timedelta(hours=1),
                limit=1000
            )
            
            df = await self.query(query)
            
            if df.empty:
                return 0.0
            
            # Métriques de qualité
            quality_metrics = []
            
            # 1. Complétude (pas de valeurs manquantes)
            completeness = 1.0 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
            quality_metrics.append(completeness)
            
            # 2. Cohérence temporelle (pas de gaps)
            if len(df) > 1:
                time_diffs = df.index.to_series().diff()
                expected_interval = time_diffs.mode()[0]
                consistency = (time_diffs == expected_interval).sum() / len(time_diffs)
                quality_metrics.append(consistency)
            
            # 3. Validité des valeurs
            if data_type == DataType.OHLCV:
                # Vérifier OHLC consistency
                valid_ohlc = (
                    (df['high'] >= df['low']) & 
                    (df['high'] >= df['open']) & 
                    (df['high'] >= df['close']) &
                    (df['low'] <= df['open']) & 
                    (df['low'] <= df['close'])
                ).sum() / len(df)
                quality_metrics.append(valid_ohlc)
            
            # Score global
            return np.mean(quality_metrics) if quality_metrics else 1.0
            
        except Exception as e:
            logger.error(f"Quality check error for {data_type}: {str(e)}")
            return 0.0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques du gestionnaire de données"""
        with self._cache_lock:
            cache_size = len(self.memory_cache)
            total_hits = sum(stats['hits'] for stats in self.cache_stats.values())
            total_misses = sum(stats['misses'] for stats in self.cache_stats.values())
            hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
        
        buffer_size = sum(len(v) for v in self.write_buffer.values())
        
        return {
            'cache': {
                'size': cache_size,
                'hit_rate': hit_rate,
                'total_hits': total_hits,
                'total_misses': total_misses
            },
            'buffer': {
                'size': buffer_size,
                'queues': len(self.write_buffer)
            },
            'health': self.health_status,
            'data_quality': self.data_quality_scores,
            'backends': list(self.backends.keys())
        }
    
    async def optimize_storage(self):
        """Optimise le stockage (compression, vacuum, etc.)"""
        logger.info("Starting storage optimization")
        
        try:
            # TimescaleDB optimizations
            if StorageBackend.TIMESCALEDB in self.backends:
                async with self.async_session() as session:
                    # Recompression des anciennes chunks
                    await session.execute("""
                        SELECT compress_chunk(chunk) 
                        FROM show_chunks('ohlcv', older_than => INTERVAL '7 days')
                        WHERE NOT is_compressed(chunk);
                    """)
                    
                    # VACUUM ANALYZE
                    await session.execute("VACUUM ANALYZE ohlcv;")
                    await session.execute("VACUUM ANALYZE trades;")
                    
                    await session.commit()
            
            # Redis optimization
            if StorageBackend.REDIS in self.backends:
                # Nettoyer les clés expirées
                self.redis_client.execute_command('MEMORY', 'PURGE')
            
            logger.info("Storage optimization completed")
            
        except Exception as e:
            logger.error(f"Storage optimization error: {str(e)}")


# Exemple d'utilisation
async def example_usage():
    """Exemple d'utilisation du DataManager"""
    # Initialiser
    manager = DataManager()
    
    # Démarrer
    await manager.start()
    
    # Stocker des données OHLCV
    ohlcv_point = DataPoint(
        timestamp=datetime.now(timezone.utc),
        symbol="BTC-USDT",
        data_type=DataType.OHLCV,
        value={
            'open': 50000,
            'high': 51000,
            'low': 49500,
            'close': 50500,
            'volume': 1000
        }
    )
    
    await manager.store(ohlcv_point)
    
    # Requête de données
    query = DataQuery(
        data_type=DataType.OHLCV,
        symbol="BTC-USDT",
        start_time=datetime.now(timezone.utc) - timedelta(hours=24),
        limit=100
    )
    
    df = await manager.query(query)
    console.print(f"Retrieved {len(df)} rows")
    
    # Statistiques
    stats = manager.get_statistics()
    console.print(f"Cache hit rate: {stats['cache']['hit_rate']:.2%}")
    
    # Arrêter
    await manager.stop()


if __name__ == "__main__":
    asyncio.run(example_usage())