"""
Redis Cache - Cache Haute Performance pour Trading Algorithmique
===============================================================

Ce module implémente un cache Redis optimisé pour les données de trading
haute fréquence avec support pour structures de données avancées, pub/sub
temps réel et sérialisation ultra-rapide.

Fonctionnalités:
- Cache L1/L2 avec TTL intelligent
- Structures optimisées (Sorted Sets, Streams, HyperLogLog)
- Pub/Sub pour broadcast temps réel
- Pipeline et transactions pour performance
- Compression automatique des grandes données
- Sharding et clustering support
- Monitoring et métriques intégrés

Architecture:
- Connection pooling avec redis-py asyncio
- Sérialisation MessagePack/Pickle optimisée
- Éviction LRU/LFU configurable
- Backup automatique vers storage permanent
- Circuit breaker pour résilience

Latence cible: < 1ms pour reads, < 2ms pour writes

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import asyncio
import redis.asyncio as redis
import msgpack
import pickle
import zlib
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Set, AsyncIterator
import hashlib
import time
from collections import defaultdict
import numpy as np
import pandas as pd

# Imports internes
from core.portfolio_manager import Symbol, Price, Quantity
from utils.logger import get_structured_logger
from utils.metrics import MetricsCollector
from utils.decorators import retry_async, circuit_breaker


class CacheNamespace(Enum):
    """Namespaces pour organisation des clés"""
    MARKET_DATA = "market"
    ORDER_BOOK = "orderbook"
    TICKS = "ticks"
    OHLCV = "ohlcv"
    POSITIONS = "positions"
    SIGNALS = "signals"
    FEATURES = "features"
    METRICS = "metrics"
    TEMP = "temp"


class SerializationMethod(Enum):
    """Méthodes de sérialisation disponibles"""
    JSON = "json"
    MSGPACK = "msgpack"
    PICKLE = "pickle"
    NUMPY = "numpy"
    COMPRESSED = "compressed"


@dataclass
class CacheConfig:
    """Configuration du cache Redis"""
    # Connection
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    ssl: bool = False
    
    # Pool
    max_connections: int = 100
    min_connections: int = 10
    connection_timeout: float = 20.0
    socket_timeout: float = 5.0
    socket_keepalive: bool = True
    
    # Performance
    decode_responses: bool = False  # Garder bytes pour performance
    health_check_interval: int = 30
    retry_on_timeout: bool = True
    
    # TTL par défaut (secondes)
    default_ttl: int = 300  # 5 minutes
    ttl_by_namespace: Dict[str, int] = field(default_factory=lambda: {
        CacheNamespace.MARKET_DATA.value: 60,      # 1 minute
        CacheNamespace.ORDER_BOOK.value: 30,       # 30 secondes
        CacheNamespace.TICKS.value: 300,           # 5 minutes
        CacheNamespace.OHLCV.value: 3600,          # 1 heure
        CacheNamespace.POSITIONS.value: 0,         # Pas d'expiration
        CacheNamespace.SIGNALS.value: 180,         # 3 minutes
        CacheNamespace.FEATURES.value: 600,        # 10 minutes
        CacheNamespace.METRICS.value: 1800,        # 30 minutes
        CacheNamespace.TEMP.value: 60              # 1 minute
    })
    
    # Sérialisation
    default_serializer: SerializationMethod = SerializationMethod.MSGPACK
    compression_threshold: int = 1024  # Compresser si > 1KB
    compression_level: int = 6
    
    # Éviction
    max_memory: str = "4gb"
    eviction_policy: str = "allkeys-lru"
    
    # Monitoring
    enable_monitoring: bool = True
    stats_interval: int = 60  # Secondes


@dataclass
class CacheStats:
    """Statistiques du cache"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    expires: int = 0
    errors: int = 0
    
    # Par namespace
    namespace_hits: Dict[str, int] = field(default_factory=dict)
    namespace_misses: Dict[str, int] = field(default_factory=dict)
    
    # Performance
    avg_get_time_ms: float = 0.0
    avg_set_time_ms: float = 0.0
    
    # Taille
    keys_count: int = 0
    memory_usage_mb: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class RedisCache:
    """
    Cache Redis haute performance pour données de trading
    avec support complet pour structures de données avancées
    """
    
    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        metrics_collector: Optional[MetricsCollector] = None
    ):
        self.config = config or CacheConfig()
        self.metrics = metrics_collector
        
        # Logger
        self.logger = get_structured_logger(
            "redis_cache",
            module="data.storage"
        )
        
        # Connection pools
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub_client: Optional[redis.Redis] = None
        
        # État
        self._initialized = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Statistiques
        self.stats = CacheStats()
        self._timing_buffer = defaultdict(list)
        
        # Sérialiseurs
        self._serializers = {
            SerializationMethod.JSON: self._json_serializer,
            SerializationMethod.MSGPACK: self._msgpack_serializer,
            SerializationMethod.PICKLE: self._pickle_serializer,
            SerializationMethod.NUMPY: self._numpy_serializer,
            SerializationMethod.COMPRESSED: self._compressed_serializer
        }
        
        self.logger.info("redis_cache_created", config=config)
    
    async def initialize(self) -> None:
        """Initialise les connexions Redis"""
        if self._initialized:
            return
        
        try:
            # Créer le pool principal
            self.redis_client = await redis.from_url(
                f"redis://{self.config.host}:{self.config.port}/{self.config.db}",
                password=self.config.password,
                max_connections=self.config.max_connections,
                decode_responses=self.config.decode_responses,
                health_check_interval=self.config.health_check_interval,
                socket_timeout=self.config.socket_timeout,
                socket_keepalive=self.config.socket_keepalive,
                retry_on_timeout=self.config.retry_on_timeout
            )
            
            # Créer un client séparé pour pub/sub
            self.pubsub_client = await redis.from_url(
                f"redis://{self.config.host}:{self.config.port}/{self.config.db}",
                password=self.config.password,
                decode_responses=False
            )
            
            # Tester la connexion
            await self.redis_client.ping()
            
            # Configurer Redis si nécessaire
            await self._configure_redis()
            
            # Démarrer le monitoring
            if self.config.enable_monitoring:
                self._monitor_task = asyncio.create_task(self._monitoring_loop())
            
            self._initialized = True
            self.logger.info("redis_cache_initialized")
            
        except Exception as e:
            self.logger.error(
                "redis_initialization_error",
                error=str(e),
                exc_info=True
            )
            raise
    
    async def close(self) -> None:
        """Ferme les connexions Redis"""
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        if self.redis_client:
            await self.redis_client.close()
        
        if self.pubsub_client:
            await self.pubsub_client.close()
        
        self._initialized = False
        self.logger.info("redis_cache_closed")
    
    async def _configure_redis(self) -> None:
        """Configure Redis avec les paramètres optimaux"""
        try:
            # Configurer la mémoire max et la politique d'éviction
            await self.redis_client.config_set("maxmemory", self.config.max_memory)
            await self.redis_client.config_set("maxmemory-policy", self.config.eviction_policy)
            
            # Optimisations pour les données de trading
            await self.redis_client.config_set("save", "")  # Désactiver RDB save
            await self.redis_client.config_set("stop-writes-on-bgsave-error", "no")
            
            self.logger.debug("redis_configured", 
                            max_memory=self.config.max_memory,
                            eviction_policy=self.config.eviction_policy)
            
        except Exception as e:
            self.logger.warning("redis_configuration_warning", error=str(e))
    
    # Méthodes de base avec métriques
    
    @retry_async(max_attempts=3, backoff_factor=2.0)
    async def get(
        self,
        key: str,
        namespace: Optional[CacheNamespace] = None,
        deserializer: Optional[SerializationMethod] = None
    ) -> Optional[Any]:
        """Récupère une valeur du cache"""
        start_time = time.perf_counter()
        full_key = self._make_key(key, namespace)
        
        try:
            value = await self.redis_client.get(full_key)
            
            # Métriques
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._timing_buffer['get'].append(elapsed_ms)
            
            if value is None:
                self.stats.misses += 1
                if namespace:
                    self.stats.namespace_misses[namespace.value] = \
                        self.stats.namespace_misses.get(namespace.value, 0) + 1
                return None
            
            self.stats.hits += 1
            if namespace:
                self.stats.namespace_hits[namespace.value] = \
                    self.stats.namespace_hits.get(namespace.value, 0) + 1
            
            # Désérialiser
            deserializer = deserializer or self.config.default_serializer
            return self._deserialize(value, deserializer)
            
        except Exception as e:
            self.stats.errors += 1
            self.logger.error("cache_get_error", key=full_key, error=str(e))
            return None
    
    @retry_async(max_attempts=3, backoff_factor=2.0)
    async def set(
        self,
        key: str,
        value: Any,
        namespace: Optional[CacheNamespace] = None,
        ttl: Optional[int] = None,
        serializer: Optional[SerializationMethod] = None
    ) -> bool:
        """Définit une valeur dans le cache"""
        start_time = time.perf_counter()
        full_key = self._make_key(key, namespace)
        
        try:
            # Sérialiser
            serializer = serializer or self.config.default_serializer
            serialized = self._serialize(value, serializer)
            
            # TTL par namespace ou défaut
            if ttl is None:
                if namespace:
                    ttl = self.config.ttl_by_namespace.get(
                        namespace.value, 
                        self.config.default_ttl
                    )
                else:
                    ttl = self.config.default_ttl
            
            # Stocker
            if ttl > 0:
                result = await self.redis_client.setex(full_key, ttl, serialized)
            else:
                result = await self.redis_client.set(full_key, serialized)
            
            # Métriques
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._timing_buffer['set'].append(elapsed_ms)
            self.stats.sets += 1
            
            return bool(result)
            
        except Exception as e:
            self.stats.errors += 1
            self.logger.error("cache_set_error", key=full_key, error=str(e))
            return False
    
    async def delete(
        self,
        key: str,
        namespace: Optional[CacheNamespace] = None
    ) -> bool:
        """Supprime une clé du cache"""
        full_key = self._make_key(key, namespace)
        
        try:
            result = await self.redis_client.delete(full_key)
            self.stats.deletes += 1
            return bool(result)
        except Exception as e:
            self.stats.errors += 1
            self.logger.error("cache_delete_error", key=full_key, error=str(e))
            return False
    
    async def exists(
        self,
        key: str,
        namespace: Optional[CacheNamespace] = None
    ) -> bool:
        """Vérifie si une clé existe"""
        full_key = self._make_key(key, namespace)
        
        try:
            return bool(await self.redis_client.exists(full_key))
        except Exception as e:
            self.logger.error("cache_exists_error", key=full_key, error=str(e))
            return False
    
    # Méthodes pour données de marché
    
    async def cache_tick(
        self,
        symbol: Symbol,
        tick_data: Dict[str, Any],
        ttl: int = 60
    ) -> bool:
        """Cache un tick de marché"""
        # Ajouter timestamp si absent
        if 'timestamp' not in tick_data:
            tick_data['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Stocker dans une liste limitée
        key = f"tick:{symbol}"
        pipe = self.redis_client.pipeline()
        
        # Ajouter à la liste (FIFO)
        pipe.lpush(key, self._serialize(tick_data, SerializationMethod.MSGPACK))
        pipe.ltrim(key, 0, 999)  # Garder les 1000 derniers
        pipe.expire(key, ttl)
        
        try:
            await pipe.execute()
            return True
        except Exception as e:
            self.logger.error("cache_tick_error", symbol=symbol, error=str(e))
            return False
    
    async def get_recent_ticks(
        self,
        symbol: Symbol,
        count: int = 100
    ) -> List[Dict[str, Any]]:
        """Récupère les ticks récents"""
        key = f"tick:{symbol}"
        
        try:
            # Récupérer les N derniers ticks
            raw_ticks = await self.redis_client.lrange(key, 0, count - 1)
            
            ticks = []
            for raw_tick in raw_ticks:
                tick = self._deserialize(raw_tick, SerializationMethod.MSGPACK)
                if tick:
                    ticks.append(tick)
            
            return ticks
            
        except Exception as e:
            self.logger.error("get_ticks_error", symbol=symbol, error=str(e))
            return []
    
    async def cache_order_book(
        self,
        symbol: Symbol,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        ttl: int = 30
    ) -> bool:
        """Cache un snapshot du carnet d'ordres"""
        bid_key = f"orderbook:{symbol}:bids"
        ask_key = f"orderbook:{symbol}:asks"
        meta_key = f"orderbook:{symbol}:meta"
        
        pipe = self.redis_client.pipeline()
        
        try:
            # Effacer les anciennes données
            pipe.delete(bid_key, ask_key)
            
            # Stocker les bids (prix négatif pour tri décroissant)
            for price, quantity in bids[:50]:  # Top 50
                pipe.zadd(bid_key, {f"{price}:{quantity}": -float(price)})
            
            # Stocker les asks
            for price, quantity in asks[:50]:  # Top 50
                pipe.zadd(ask_key, {f"{price}:{quantity}": float(price)})
            
            # Métadonnées
            meta = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'best_bid': bids[0][0] if bids else 0,
                'best_ask': asks[0][0] if asks else 0,
                'spread': asks[0][0] - bids[0][0] if bids and asks else 0
            }
            pipe.hset(meta_key, mapping={
                k: str(v) for k, v in meta.items()
            })
            
            # TTL
            pipe.expire(bid_key, ttl)
            pipe.expire(ask_key, ttl)
            pipe.expire(meta_key, ttl)
            
            await pipe.execute()
            return True
            
        except Exception as e:
            self.logger.error("cache_orderbook_error", symbol=symbol, error=str(e))
            return False
    
    async def get_order_book(
        self,
        symbol: Symbol,
        depth: int = 20
    ) -> Optional[Dict[str, Any]]:
        """Récupère le carnet d'ordres du cache"""
        bid_key = f"orderbook:{symbol}:bids"
        ask_key = f"orderbook:{symbol}:asks"
        meta_key = f"orderbook:{symbol}:meta"
        
        try:
            pipe = self.redis_client.pipeline()
            pipe.zrange(bid_key, 0, depth - 1)
            pipe.zrange(ask_key, 0, depth - 1)
            pipe.hgetall(meta_key)
            
            results = await pipe.execute()
            
            if not results[0] and not results[1]:
                return None
            
            # Parser les résultats
            bids = []
            for entry in results[0]:
                price, quantity = entry.decode().split(':')
                bids.append((float(price), float(quantity)))
            
            asks = []
            for entry in results[1]:
                price, quantity = entry.decode().split(':')
                asks.append((float(price), float(quantity)))
            
            # Métadonnées
            meta = {k.decode(): v.decode() for k, v in results[2].items()} if results[2] else {}
            
            return {
                'symbol': symbol,
                'bids': bids,
                'asks': asks,
                'meta': meta
            }
            
        except Exception as e:
            self.logger.error("get_orderbook_error", symbol=symbol, error=str(e))
            return None
    
    async def cache_ohlcv(
        self,
        symbol: Symbol,
        interval: str,
        ohlcv_data: Dict[str, Any],
        ttl: int = 3600
    ) -> bool:
        """Cache des données OHLCV"""
        key = f"ohlcv:{symbol}:{interval}"
        
        # Utiliser un hash pour stocker plusieurs périodes
        timestamp = ohlcv_data.get('timestamp', datetime.now(timezone.utc).isoformat())
        field = timestamp
        
        try:
            pipe = self.redis_client.pipeline()
            pipe.hset(key, field, self._serialize(ohlcv_data, SerializationMethod.MSGPACK))
            pipe.expire(key, ttl)
            
            # Limiter le nombre d'entrées
            pipe.hlen(key)
            results = await pipe.execute()
            
            # Si trop d'entrées, supprimer les plus anciennes
            if results[2] > 1000:
                all_fields = await self.redis_client.hkeys(key)
                # Trier et garder les 900 plus récents
                sorted_fields = sorted(all_fields)
                to_delete = sorted_fields[:-900]
                if to_delete:
                    await self.redis_client.hdel(key, *to_delete)
            
            return True
            
        except Exception as e:
            self.logger.error("cache_ohlcv_error", 
                            symbol=symbol, 
                            interval=interval, 
                            error=str(e))
            return False
    
    # Pub/Sub pour mises à jour temps réel
    
    async def publish_market_update(
        self,
        channel: str,
        data: Dict[str, Any]
    ) -> int:
        """Publie une mise à jour de marché"""
        try:
            serialized = self._serialize(data, SerializationMethod.MSGPACK)
            return await self.pubsub_client.publish(channel, serialized)
        except Exception as e:
            self.logger.error("publish_error", channel=channel, error=str(e))
            return 0
    
    @asynccontextmanager
    async def subscribe_market_updates(
        self,
        channels: List[str]
    ) -> AsyncIterator[redis.client.PubSub]:
        """S'abonne aux mises à jour de marché"""
        pubsub = self.pubsub_client.pubsub()
        
        try:
            # S'abonner aux canaux
            await pubsub.subscribe(*channels)
            yield pubsub
        finally:
            # Se désabonner
            await pubsub.unsubscribe()
            await pubsub.close()
    
    # Méthodes pour features ML
    
    async def cache_features(
        self,
        feature_set_id: str,
        features: Union[Dict[str, Any], np.ndarray, pd.DataFrame],
        ttl: int = 600
    ) -> bool:
        """Cache des features pour ML"""
        key = f"features:{feature_set_id}"
        
        try:
            # Sérialiser selon le type
            if isinstance(features, np.ndarray):
                serialized = self._serialize(features, SerializationMethod.NUMPY)
            elif isinstance(features, pd.DataFrame):
                # Convertir DataFrame en dict pour sérialisation
                data = {
                    'columns': features.columns.tolist(),
                    'index': features.index.tolist(),
                    'data': features.values.tolist()
                }
                serialized = self._serialize(data, SerializationMethod.COMPRESSED)
            else:
                serialized = self._serialize(features, SerializationMethod.MSGPACK)
            
            return await self.redis_client.setex(key, ttl, serialized)
            
        except Exception as e:
            self.logger.error("cache_features_error", 
                            feature_set_id=feature_set_id, 
                            error=str(e))
            return False
    
    async def get_features(
        self,
        feature_set_id: str
    ) -> Optional[Union[Dict[str, Any], np.ndarray, pd.DataFrame]]:
        """Récupère des features du cache"""
        key = f"features:{feature_set_id}"
        
        try:
            data = await self.redis_client.get(key)
            if not data:
                return None
            
            # Essayer différentes désérialisations
            try:
                # Essayer numpy d'abord
                return self._deserialize(data, SerializationMethod.NUMPY)
            except:
                try:
                    # Essayer compressed (DataFrame)
                    decompressed = self._deserialize(data, SerializationMethod.COMPRESSED)
                    if isinstance(decompressed, dict) and 'columns' in decompressed:
                        # Reconstruire DataFrame
                        return pd.DataFrame(
                            decompressed['data'],
                            columns=decompressed['columns'],
                            index=decompressed['index']
                        )
                    return decompressed
                except:
                    # Fallback msgpack
                    return self._deserialize(data, SerializationMethod.MSGPACK)
                    
        except Exception as e:
            self.logger.error("get_features_error", 
                            feature_set_id=feature_set_id, 
                            error=str(e))
            return None
    
    # Transactions et pipelines
    
    @asynccontextmanager
    async def pipeline(self, transaction: bool = True):
        """Context manager pour pipeline/transaction"""
        pipe = self.redis_client.pipeline(transaction=transaction)
        try:
            yield pipe
            await pipe.execute()
        except Exception as e:
            self.logger.error("pipeline_error", error=str(e))
            raise
    
    # Méthodes utilitaires
    
    async def clear_namespace(self, namespace: CacheNamespace) -> int:
        """Efface toutes les clés d'un namespace"""
        pattern = f"{namespace.value}:*"
        
        try:
            # Utiliser SCAN pour éviter de bloquer Redis
            cursor = 0
            deleted = 0
            
            while True:
                cursor, keys = await self.redis_client.scan(
                    cursor, 
                    match=pattern, 
                    count=1000
                )
                
                if keys:
                    deleted += await self.redis_client.delete(*keys)
                
                if cursor == 0:
                    break
            
            self.logger.info("namespace_cleared", 
                           namespace=namespace.value, 
                           keys_deleted=deleted)
            return deleted
            
        except Exception as e:
            self.logger.error("clear_namespace_error", 
                            namespace=namespace.value, 
                            error=str(e))
            return 0
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Obtient l'utilisation mémoire par pattern"""
        try:
            info = await self.redis_client.info("memory")
            
            # Analyser l'utilisation par namespace
            namespace_usage = {}
            for ns in CacheNamespace:
                pattern = f"{ns.value}:*"
                
                # Compter les clés
                cursor = 0
                key_count = 0
                total_memory = 0
                
                while True:
                    cursor, keys = await self.redis_client.scan(
                        cursor, 
                        match=pattern, 
                        count=100
                    )
                    
                    key_count += len(keys)
                    
                    # Estimer la mémoire (sampling)
                    if keys and key_count <= 1000:  # Limiter l'échantillonnage
                        for key in keys[:10]:  # Échantillon
                            try:
                                memory = await self.redis_client.memory_usage(key)
                                if memory:
                                    total_memory += memory
                            except:
                                pass
                    
                    if cursor == 0:
                        break
                
                # Extrapoler si échantillonné
                if key_count > 100 and total_memory > 0:
                    avg_per_key = total_memory / min(key_count, 100)
                    estimated_total = avg_per_key * key_count
                else:
                    estimated_total = total_memory
                
                namespace_usage[ns.value] = {
                    'key_count': key_count,
                    'estimated_memory_mb': estimated_total / 1024 / 1024
                }
            
            return {
                'total_memory_mb': info.get('used_memory', 0) / 1024 / 1024,
                'peak_memory_mb': info.get('used_memory_peak', 0) / 1024 / 1024,
                'rss_memory_mb': info.get('used_memory_rss', 0) / 1024 / 1024,
                'evicted_keys': info.get('evicted_keys', 0),
                'namespace_usage': namespace_usage
            }
            
        except Exception as e:
            self.logger.error("memory_usage_error", error=str(e))
            return {}
    
    # Sérialisation
    
    def _serialize(self, value: Any, method: SerializationMethod) -> bytes:
        """Sérialise une valeur selon la méthode"""
        serializer = self._serializers.get(method, self._msgpack_serializer)
        return serializer(value)
    
    def _deserialize(self, data: bytes, method: SerializationMethod) -> Any:
        """Désérialise des données selon la méthode"""
        if method == SerializationMethod.JSON:
            return json.loads(data)
        elif method == SerializationMethod.MSGPACK:
            return msgpack.unpackb(data, raw=False, strict_map_key=False)
        elif method == SerializationMethod.PICKLE:
            return pickle.loads(data)
        elif method == SerializationMethod.NUMPY:
            return np.frombuffer(data, dtype=np.float64)
        elif method == SerializationMethod.COMPRESSED:
            decompressed = zlib.decompress(data)
            return msgpack.unpackb(decompressed, raw=False, strict_map_key=False)
        else:
            return data
    
    def _json_serializer(self, value: Any) -> bytes:
        """Sérialise en JSON"""
        return json.dumps(value, default=str).encode('utf-8')
    
    def _msgpack_serializer(self, value: Any) -> bytes:
        """Sérialise avec MessagePack"""
        # Convertir les types non supportés
        if isinstance(value, (Decimal, datetime)):
            value = str(value)
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        
        return msgpack.packb(value, use_bin_type=True)
    
    def _pickle_serializer(self, value: Any) -> bytes:
        """Sérialise avec Pickle"""
        return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    
    def _numpy_serializer(self, value: np.ndarray) -> bytes:
        """Sérialise un array NumPy"""
        return value.astype(np.float64).tobytes()
    
    def _compressed_serializer(self, value: Any) -> bytes:
        """Sérialise et compresse"""
        serialized = self._msgpack_serializer(value)
        
        # Compresser si au-dessus du seuil
        if len(serialized) > self.config.compression_threshold:
            return zlib.compress(serialized, level=self.config.compression_level)
        return serialized
    
    def _make_key(self, key: str, namespace: Optional[CacheNamespace] = None) -> str:
        """Construit une clé complète avec namespace"""
        if namespace:
            return f"{namespace.value}:{key}"
        return key
    
    # Monitoring
    
    async def _monitoring_loop(self) -> None:
        """Boucle de monitoring des métriques"""
        while True:
            try:
                await asyncio.sleep(self.config.stats_interval)
                
                # Calculer les moyennes
                if self._timing_buffer['get']:
                    self.stats.avg_get_time_ms = sum(self._timing_buffer['get']) / len(self._timing_buffer['get'])
                    self._timing_buffer['get'].clear()
                
                if self._timing_buffer['set']:
                    self.stats.avg_set_time_ms = sum(self._timing_buffer['set']) / len(self._timing_buffer['set'])
                    self._timing_buffer['set'].clear()
                
                # Obtenir les stats Redis
                info = await self.redis_client.info()
                self.stats.keys_count = info.get('db0', {}).get('keys', 0)
                self.stats.memory_usage_mb = info.get('used_memory', 0) / 1024 / 1024
                
                # Publier les métriques
                if self.metrics:
                    self.metrics.gauge("redis.hit_rate", self.stats.hit_rate)
                    self.metrics.gauge("redis.memory_usage_mb", self.stats.memory_usage_mb)
                    self.metrics.gauge("redis.keys_count", self.stats.keys_count)
                    self.metrics.gauge("redis.avg_get_time_ms", self.stats.avg_get_time_ms)
                    self.metrics.gauge("redis.avg_set_time_ms", self.stats.avg_set_time_ms)
                
                # Logger les stats
                self.logger.debug(
                    "cache_stats",
                    hit_rate=f"{self.stats.hit_rate:.2%}",
                    memory_mb=f"{self.stats.memory_usage_mb:.1f}",
                    keys=self.stats.keys_count,
                    avg_get_ms=f"{self.stats.avg_get_time_ms:.2f}",
                    avg_set_ms=f"{self.stats.avg_set_time_ms:.2f}"
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("monitoring_error", error=str(e))
    
    def get_stats(self) -> CacheStats:
        """Retourne les statistiques actuelles"""
        return self.stats
    
    async def reset_stats(self) -> None:
        """Réinitialise les statistiques"""
        self.stats = CacheStats()
        self._timing_buffer.clear()
        self.logger.info("cache_stats_reset")


# Classe spécialisée pour le cache L1 local
class LocalL1Cache:
    """Cache L1 en mémoire locale pour latence minimale"""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 60):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Récupère du cache L1"""
        async with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if time.time() < expiry:
                    self._access_times[key] = time.time()
                    return value
                else:
                    # Expiré
                    del self._cache[key]
                    del self._access_times[key]
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Définit dans le cache L1"""
        async with self._lock:
            # Éviction LRU si plein
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Trouver la clé la moins récemment accédée
                lru_key = min(self._access_times, key=self._access_times.get)
                del self._cache[lru_key]
                del self._access_times[lru_key]
            
            ttl = ttl or self.ttl_seconds
            expiry = time.time() + ttl
            self._cache[key] = (value, expiry)
            self._access_times[key] = time.time()
    
    async def clear(self) -> None:
        """Vide le cache L1"""
        async with self._lock:
            self._cache.clear()
            self._access_times.clear()


# Cache à deux niveaux
class TwoLevelCache:
    """Cache à deux niveaux : L1 local + L2 Redis"""
    
    def __init__(
        self,
        redis_cache: RedisCache,
        l1_max_size: int = 10000,
        l1_ttl: int = 30
    ):
        self.l1_cache = LocalL1Cache(max_size=l1_max_size, ttl_seconds=l1_ttl)
        self.l2_cache = redis_cache
        self.logger = get_structured_logger("two_level_cache")
    
    async def get(
        self,
        key: str,
        namespace: Optional[CacheNamespace] = None
    ) -> Optional[Any]:
        """Récupère d'abord de L1, puis L2"""
        # Essayer L1
        full_key = f"{namespace.value}:{key}" if namespace else key
        value = await self.l1_cache.get(full_key)
        
        if value is not None:
            return value
        
        # Essayer L2
        value = await self.l2_cache.get(key, namespace)
        
        if value is not None:
            # Mettre en L1 pour les prochains accès
            await self.l1_cache.set(full_key, value)
        
        return value
    
    async def set(
        self,
        key: str,
        value: Any,
        namespace: Optional[CacheNamespace] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Définit dans L1 et L2"""
        full_key = f"{namespace.value}:{key}" if namespace else key
        
        # Mettre en L1
        await self.l1_cache.set(full_key, value, ttl=min(ttl or 3600, 60))
        
        # Mettre en L2
        return await self.l2_cache.set(key, value, namespace, ttl)
    
    async def invalidate(
        self,
        key: str,
        namespace: Optional[CacheNamespace] = None
    ) -> bool:
        """Invalide dans L1 et L2"""
        full_key = f"{namespace.value}:{key}" if namespace else key
        
        # Supprimer de L1
        await self.l1_cache.set(full_key, None, ttl=1)  # Marquer comme invalide
        
        # Supprimer de L2
        return await self.l2_cache.delete(key, namespace)


# Factory pour créer des caches
async def create_redis_cache(
    host: str = "localhost",
    port: int = 6379,
    db: int = 0,
    password: Optional[str] = None,
    **kwargs
) -> RedisCache:
    """Crée et initialise un cache Redis"""
    config = CacheConfig(
        host=host,
        port=port,
        db=db,
        password=password,
        **kwargs
    )
    
    cache = RedisCache(config=config)
    await cache.initialize()
    
    return cache