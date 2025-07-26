"""
Feature Store pour Robot de Trading Algorithmique IA
===================================================

Ce module implémente un feature store haute performance pour stocker,
versionner et servir les features calculées pour le machine learning.
Optimisé pour accès temps réel avec cache Redis et historique TimescaleDB.

Architecture:
- Storage hybride : Redis (temps réel) + TimescaleDB (historique)
- Versioning automatique des features
- Support batch et streaming
- Compression intelligente pour économiser l'espace
- API unifiée pour lecture/écriture
- Partitionnement par symbole et timeframe
- Calcul incrémental et mise à jour temps réel
- Monitoring et métriques intégrés

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import asyncio
import gzip
import hashlib
import json
import pickle
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Set,
    Callable, Protocol, TypeVar, Generic
)
import warnings
from contextlib import asynccontextmanager

# Third-party imports
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.preprocessing import StandardScaler

# Imports internes
from config.settings import TradingConfig, get_config
from utils.logger import get_structured_logger
from utils.metrics import MetricsCollector
from utils.decorators import retry_async, circuit_breaker, rate_limit
from data.storage.timeseries_db import TimeSeriesDB
from data.storage.redis_cache import RedisCache
from ml.features.feature_engineering import FeatureEngineer


T = TypeVar('T')


class FeatureType(Enum):
    """Types de features supportées"""
    TECHNICAL = "technical"          # Indicateurs techniques
    MARKET_MICROSTRUCTURE = "microstructure"  # Microstructure
    SENTIMENT = "sentiment"          # Sentiment analysis
    FUNDAMENTAL = "fundamental"      # Données fondamentales
    ALTERNATIVE = "alternative"      # Données alternatives
    ENGINEERED = "engineered"       # Features ML créées
    REGIME = "regime"               # Régimes de marché
    CUSTOM = "custom"               # Features personnalisées


class StorageBackend(Enum):
    """Backends de stockage disponibles"""
    REDIS = "redis"                 # Cache temps réel
    TIMESCALEDB = "timescaledb"     # Historique compressé
    PARQUET = "parquet"             # Fichiers locaux
    MEMORY = "memory"               # Cache mémoire


class CompressionType(Enum):
    """Types de compression supportés"""
    NONE = "none"
    GZIP = "gzip"
    ZSTD = "zstd"
    LZ4 = "lz4"
    SNAPPY = "snappy"


@dataclass
class FeatureSet:
    """Ensemble de features avec métadonnées"""
    feature_id: str
    symbol: str
    timestamp: datetime
    features: Dict[str, Union[float, np.ndarray]]
    feature_type: FeatureType
    version: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation"""
        return {
            'feature_id': self.feature_id,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'features': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                        for k, v in self.features.items()},
            'feature_type': self.feature_type.value,
            'version': self.version,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureSet':
        """Crée une instance depuis un dictionnaire"""
        return cls(
            feature_id=data['feature_id'],
            symbol=data['symbol'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            features={k: np.array(v) if isinstance(v, list) else v 
                     for k, v in data['features'].items()},
            feature_type=FeatureType(data['feature_type']),
            version=data['version'],
            metadata=data.get('metadata', {})
        )


@dataclass
class FeatureSchema:
    """Schéma définissant la structure d'un ensemble de features"""
    name: str
    features: List[str]
    dtypes: Dict[str, type]
    version: str
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    update_frequency: str = "1m"
    
    def validate(self, feature_dict: Dict[str, Any]) -> bool:
        """Valide qu'un dictionnaire respecte le schéma"""
        # Vérifier toutes les features requises
        missing = set(self.features) - set(feature_dict.keys())
        if missing:
            return False
        
        # Vérifier les types
        for feature, expected_type in self.dtypes.items():
            if feature in feature_dict:
                value = feature_dict[feature]
                if not isinstance(value, expected_type):
                    if expected_type == float and isinstance(value, (int, np.number)):
                        continue  # Acceptable
                    return False
        
        return True


class FeatureStore:
    """
    Feature Store principal pour la gestion centralisée des features
    
    Fournit une interface unifiée pour stocker et récupérer des features
    avec support pour multiple backends et stratégies de cache.
    """
    
    def __init__(
        self,
        redis_cache: Optional[RedisCache] = None,
        timeseries_db: Optional[TimeSeriesDB] = None,
        cache_ttl: int = 3600,  # 1 heure
        compression: CompressionType = CompressionType.GZIP,
        enable_versioning: bool = True,
        memory_cache_size: int = 10000
    ):
        """
        Initialise le Feature Store
        
        Args:
            redis_cache: Instance Redis pour cache temps réel
            timeseries_db: Instance TimescaleDB pour historique
            cache_ttl: Durée de vie du cache en secondes
            compression: Type de compression à utiliser
            enable_versioning: Activer le versioning des features
            memory_cache_size: Taille du cache mémoire local
        """
        self.redis_cache = redis_cache
        self.timeseries_db = timeseries_db
        self.cache_ttl = cache_ttl
        self.compression = compression
        self.enable_versioning = enable_versioning
        
        # Logging et métriques
        self.logger = get_structured_logger(self.__class__.__name__)
        self.metrics = MetricsCollector("feature_store")
        
        # Cache mémoire local (LRU)
        self._memory_cache: Dict[str, Tuple[FeatureSet, float]] = {}
        self._cache_order = deque(maxlen=memory_cache_size)
        
        # Registre des schémas
        self._schemas: Dict[str, FeatureSchema] = {}
        self._register_default_schemas()
        
        # Feature engineering
        self.feature_engineer = FeatureEngineer()
        
        # Statistiques
        self.stats = {
            'writes': 0,
            'reads': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'compression_ratio': 1.0
        }
        
        # Scalers pour normalisation
        self._scalers: Dict[str, StandardScaler] = {}
        
        # Configuration
        self.config = get_config()
        
        # Lock pour opérations concurrentes
        self._write_lock = asyncio.Lock()
        
        self.logger.info(
            "Feature Store initialisé",
            compression=compression.value,
            versioning=enable_versioning,
            cache_size=memory_cache_size
        )
    
    def _register_default_schemas(self) -> None:
        """Enregistre les schémas de features par défaut"""
        # Schéma pour features techniques
        technical_schema = FeatureSchema(
            name="technical_indicators",
            features=[
                'sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20',
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'atr', 'adx', 'volume_sma', 'vwap', 'obv'
            ],
            dtypes={f: float for f in [
                'sma_10', 'sma_20', 'sma_50', 'ema_10', 'ema_20',
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'atr', 'adx', 'volume_sma', 'vwap', 'obv'
            ]},
            version="1.0.0",
            description="Indicateurs techniques standards"
        )
        self.register_schema(technical_schema)
        
        # Schéma pour microstructure
        microstructure_schema = FeatureSchema(
            name="market_microstructure",
            features=[
                'bid_ask_spread', 'spread_percentage', 'mid_price',
                'bid_volume', 'ask_volume', 'order_imbalance',
                'trade_intensity', 'kyle_lambda', 'price_impact'
            ],
            dtypes={f: float for f in [
                'bid_ask_spread', 'spread_percentage', 'mid_price',
                'bid_volume', 'ask_volume', 'order_imbalance',
                'trade_intensity', 'kyle_lambda', 'price_impact'
            ]},
            version="1.0.0",
            description="Features de microstructure de marché"
        )
        self.register_schema(microstructure_schema)
        
        # Schéma pour régimes de marché
        regime_schema = FeatureSchema(
            name="market_regime",
            features=[
                'regime_label', 'regime_probability', 'trend_strength',
                'volatility_regime', 'correlation_regime'
            ],
            dtypes={
                'regime_label': str,
                'regime_probability': float,
                'trend_strength': float,
                'volatility_regime': str,
                'correlation_regime': float
            },
            version="1.0.0",
            description="Détection de régimes de marché"
        )
        self.register_schema(regime_schema)
    
    def register_schema(self, schema: FeatureSchema) -> None:
        """
        Enregistre un nouveau schéma de features
        
        Args:
            schema: Schéma à enregistrer
        """
        self._schemas[schema.name] = schema
        self.logger.info(f"Schéma enregistré: {schema.name} v{schema.version}")
    
    async def write_features(
        self,
        symbol: str,
        timestamp: datetime,
        features: Dict[str, Any],
        feature_type: FeatureType = FeatureType.TECHNICAL,
        schema_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Écrit un ensemble de features dans le store
        
        Args:
            symbol: Symbole de l'actif
            timestamp: Timestamp des features
            features: Dictionnaire des features
            feature_type: Type de features
            schema_name: Nom du schéma à valider
            metadata: Métadonnées additionnelles
            
        Returns:
            ID unique du feature set
        """
        async with self._write_lock:
            start_time = time.time()
            
            # Valider contre le schéma si fourni
            if schema_name and schema_name in self._schemas:
                schema = self._schemas[schema_name]
                if not schema.validate(features):
                    raise ValueError(f"Features ne respectent pas le schéma {schema_name}")
            
            # Générer l'ID unique
            feature_id = self._generate_feature_id(symbol, timestamp, feature_type)
            
            # Créer le feature set
            feature_set = FeatureSet(
                feature_id=feature_id,
                symbol=symbol,
                timestamp=timestamp,
                features=features,
                feature_type=feature_type,
                version=self._get_current_version() if self.enable_versioning else "latest",
                metadata=metadata or {}
            )
            
            # Écrire dans les différents backends
            tasks = []
            
            # Redis (cache temps réel)
            if self.redis_cache:
                tasks.append(self._write_to_redis(feature_set))
            
            # TimescaleDB (historique)
            if self.timeseries_db:
                tasks.append(self._write_to_timescaledb(feature_set))
            
            # Cache mémoire local
            self._update_memory_cache(feature_id, feature_set)
            
            # Exécuter toutes les écritures en parallèle
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Métriques
            write_time = time.time() - start_time
            self.stats['writes'] += 1
            
            self.metrics.record_histogram(
                "feature_write_duration",
                write_time,
                labels={"symbol": symbol, "type": feature_type.value}
            )
            
            self.logger.debug(
                f"Features écrites",
                symbol=symbol,
                feature_id=feature_id,
                feature_count=len(features),
                duration_ms=write_time * 1000
            )
            
            return feature_id
    
    async def read_features(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        feature_names: Optional[List[str]] = None,
        feature_type: Optional[FeatureType] = None
    ) -> pd.DataFrame:
        """
        Lit les features pour une période donnée
        
        Args:
            symbol: Symbole de l'actif
            start_time: Début de la période
            end_time: Fin de la période
            feature_names: Liste des features à récupérer (None = toutes)
            feature_type: Type de features à filtrer
            
        Returns:
            DataFrame avec les features
        """
        start = time.time()
        
        # Vérifier le cache mémoire d'abord
        cached_data = self._check_memory_cache_range(
            symbol, start_time, end_time, feature_type
        )
        
        if cached_data is not None:
            self.stats['cache_hits'] += 1
            self.metrics.increment_counter("cache_hits", labels={"level": "memory"})
            return self._filter_features(cached_data, feature_names)
        
        # Essayer Redis ensuite
        if self.redis_cache:
            redis_data = await self._read_from_redis_range(
                symbol, start_time, end_time, feature_type
            )
            
            if redis_data is not None:
                self.stats['cache_hits'] += 1
                self.metrics.increment_counter("cache_hits", labels={"level": "redis"})
                return self._filter_features(redis_data, feature_names)
        
        # Finalement, lire depuis TimescaleDB
        if self.timeseries_db:
            db_data = await self._read_from_timescaledb(
                symbol, start_time, end_time, feature_type
            )
            
            if db_data is not None:
                self.stats['cache_misses'] += 1
                self.metrics.increment_counter("cache_misses")
                
                # Mettre en cache pour les prochaines lectures
                if self.redis_cache:
                    await self._cache_to_redis(symbol, db_data)
                
                return self._filter_features(db_data, feature_names)
        
        # Aucune donnée trouvée
        self.logger.warning(
            f"Aucune feature trouvée",
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        return pd.DataFrame()
    
    async def read_latest_features(
        self,
        symbol: str,
        feature_names: Optional[List[str]] = None,
        feature_type: Optional[FeatureType] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Lit les dernières features disponibles
        
        Args:
            symbol: Symbole de l'actif
            feature_names: Features spécifiques à récupérer
            feature_type: Type de features
            
        Returns:
            Dictionnaire des dernières features ou None
        """
        # Clé pour le cache des dernières features
        cache_key = f"latest:{symbol}:{feature_type.value if feature_type else 'all'}"
        
        # Vérifier le cache mémoire
        if cache_key in self._memory_cache:
            feature_set, timestamp = self._memory_cache[cache_key]
            if time.time() - timestamp < 60:  # Cache de 1 minute pour latest
                features = feature_set.features
                if feature_names:
                    features = {k: v for k, v in features.items() if k in feature_names}
                return features
        
        # Essayer Redis
        if self.redis_cache:
            latest_key = f"features:latest:{symbol}"
            if feature_type:
                latest_key += f":{feature_type.value}"
            
            cached_data = await self.redis_cache.get(latest_key)
            if cached_data:
                feature_set = FeatureSet.from_dict(json.loads(cached_data))
                self._update_memory_cache(cache_key, feature_set)
                
                features = feature_set.features
                if feature_names:
                    features = {k: v for k, v in features.items() if k in feature_names}
                return features
        
        # Requête TimescaleDB pour la dernière entrée
        if self.timeseries_db:
            query = f"""
            SELECT * FROM features
            WHERE symbol = %s
            {f"AND feature_type = %s" if feature_type else ""}
            ORDER BY timestamp DESC
            LIMIT 1
            """
            
            params = [symbol]
            if feature_type:
                params.append(feature_type.value)
            
            result = await self.timeseries_db.fetch_one(query, params)
            
            if result:
                features = json.loads(result['features'])
                if feature_names:
                    features = {k: v for k, v in features.items() if k in feature_names}
                
                # Mettre en cache
                if self.redis_cache:
                    await self.redis_cache.set(
                        latest_key,
                        json.dumps({
                            'features': features,
                            'timestamp': result['timestamp'].isoformat()
                        }),
                        expire=self.cache_ttl
                    )
                
                return features
        
        return None
    
    async def compute_and_store_features(
        self,
        symbol: str,
        market_data: pd.DataFrame,
        feature_configs: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Calcule et stocke automatiquement les features
        
        Args:
            symbol: Symbole de l'actif
            market_data: Données de marché
            feature_configs: Configuration des features à calculer
            
        Returns:
            ID du feature set créé
        """
        # Utiliser la configuration par défaut si non fournie
        if feature_configs is None:
            feature_configs = self._get_default_feature_config()
        
        # Calculer les features via FeatureEngineer
        features = self.feature_engineer.calculate_all_features(
            market_data,
            **feature_configs
        )
        
        # Déterminer le timestamp (dernière ligne des données)
        timestamp = market_data.index[-1] if isinstance(market_data.index, pd.DatetimeIndex) else datetime.now(timezone.utc)
        
        # Convertir en dictionnaire plat
        feature_dict = {}
        for category, values in features.items():
            if isinstance(values, dict):
                for name, value in values.items():
                    feature_dict[f"{category}_{name}"] = value
            else:
                feature_dict[category] = values
        
        # Stocker les features
        return await self.write_features(
            symbol=symbol,
            timestamp=timestamp,
            features=feature_dict,
            feature_type=FeatureType.ENGINEERED,
            metadata={'data_points': len(market_data)}
        )
    
    async def _write_to_redis(self, feature_set: FeatureSet) -> None:
        """Écrit dans Redis avec compression optionnelle"""
        try:
            # Clé principale
            key = f"features:{feature_set.symbol}:{feature_set.timestamp.timestamp()}"
            
            # Sérialiser et compresser
            data = json.dumps(feature_set.to_dict())
            if self.compression != CompressionType.NONE:
                data = self._compress_data(data.encode())
            
            # Écrire avec TTL
            await self.redis_cache.set(key, data, expire=self.cache_ttl)
            
            # Mettre à jour la clé "latest"
            latest_key = f"features:latest:{feature_set.symbol}:{feature_set.feature_type.value}"
            await self.redis_cache.set(latest_key, data, expire=self.cache_ttl)
            
            # Ajouter à l'index par timerange
            index_key = f"features:index:{feature_set.symbol}:{feature_set.timestamp.date()}"
            await self.redis_cache.zadd(
                index_key,
                {key: feature_set.timestamp.timestamp()}
            )
            
        except Exception as e:
            self.logger.error(f"Erreur écriture Redis", error=str(e))
    
    async def _write_to_timescaledb(self, feature_set: FeatureSet) -> None:
        """Écrit dans TimescaleDB avec compression"""
        try:
            # Préparer les données
            features_json = json.dumps(feature_set.features)
            
            # Compresser si nécessaire
            if self.compression != CompressionType.NONE:
                features_json = self._compress_data(features_json.encode()).decode('latin-1')
            
            # Requête d'insertion
            query = """
            INSERT INTO features (
                feature_id, symbol, timestamp, features, 
                feature_type, version, metadata, compressed
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (feature_id) DO UPDATE
            SET features = EXCLUDED.features,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
            """
            
            await self.timeseries_db.execute(
                query,
                feature_set.feature_id,
                feature_set.symbol,
                feature_set.timestamp,
                features_json,
                feature_set.feature_type.value,
                feature_set.version,
                json.dumps(feature_set.metadata),
                self.compression != CompressionType.NONE
            )
            
        except Exception as e:
            self.logger.error(f"Erreur écriture TimescaleDB", error=str(e))
    
    def _update_memory_cache(self, key: str, feature_set: FeatureSet) -> None:
        """Met à jour le cache mémoire LRU"""
        # Supprimer l'ancienne entrée si elle existe
        if key in self._memory_cache:
            self._cache_order.remove(key)
        
        # Ajouter la nouvelle entrée
        self._memory_cache[key] = (feature_set, time.time())
        self._cache_order.append(key)
        
        # Nettoyer si dépassement de taille
        while len(self._memory_cache) > self._cache_order.maxlen:
            oldest_key = self._cache_order.popleft()
            del self._memory_cache[oldest_key]
    
    def _check_memory_cache_range(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        feature_type: Optional[FeatureType]
    ) -> Optional[pd.DataFrame]:
        """Vérifie le cache mémoire pour une plage de temps"""
        # Pour l'instant, retourner None (amélioration future)
        return None
    
    async def _read_from_redis_range(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        feature_type: Optional[FeatureType]
    ) -> Optional[pd.DataFrame]:
        """Lit une plage depuis Redis"""
        try:
            # Utiliser l'index pour trouver les clés
            current_date = start_time.date()
            end_date = end_time.date()
            
            all_keys = []
            
            while current_date <= end_date:
                index_key = f"features:index:{symbol}:{current_date}"
                
                # Récupérer les clés dans la plage de temps
                keys = await self.redis_cache.zrangebyscore(
                    index_key,
                    start_time.timestamp(),
                    end_time.timestamp()
                )
                
                all_keys.extend(keys)
                current_date += timedelta(days=1)
            
            if not all_keys:
                return None
            
            # Récupérer toutes les données
            features_list = []
            
            for key in all_keys:
                data = await self.redis_cache.get(key)
                if data:
                    # Décompresser si nécessaire
                    if self.compression != CompressionType.NONE:
                        data = self._decompress_data(data)
                    
                    feature_set = FeatureSet.from_dict(json.loads(data))
                    
                    # Filtrer par type si nécessaire
                    if feature_type and feature_set.feature_type != feature_type:
                        continue
                    
                    features_list.append({
                        'timestamp': feature_set.timestamp,
                        'symbol': feature_set.symbol,
                        **feature_set.features
                    })
            
            if features_list:
                return pd.DataFrame(features_list).set_index('timestamp')
            
        except Exception as e:
            self.logger.error(f"Erreur lecture Redis", error=str(e))
        
        return None
    
    async def _read_from_timescaledb(
        self,
        symbol: str,
        start_time: datetime,
        end_time: datetime,
        feature_type: Optional[FeatureType]
    ) -> Optional[pd.DataFrame]:
        """Lit depuis TimescaleDB"""
        try:
            query = """
            SELECT timestamp, features, compressed
            FROM features
            WHERE symbol = $1
            AND timestamp >= $2
            AND timestamp <= $3
            """
            
            params = [symbol, start_time, end_time]
            
            if feature_type:
                query += " AND feature_type = $4"
                params.append(feature_type.value)
            
            query += " ORDER BY timestamp"
            
            results = await self.timeseries_db.fetch_all(query, params)
            
            if not results:
                return None
            
            # Convertir en DataFrame
            data_list = []
            
            for row in results:
                features_data = row['features']
                
                # Décompresser si nécessaire
                if row['compressed']:
                    features_data = self._decompress_data(
                        features_data.encode('latin-1')
                    )
                
                features = json.loads(features_data)
                
                data_list.append({
                    'timestamp': row['timestamp'],
                    'symbol': symbol,
                    **features
                })
            
            return pd.DataFrame(data_list).set_index('timestamp')
            
        except Exception as e:
            self.logger.error(f"Erreur lecture TimescaleDB", error=str(e))
            return None
    
    async def _cache_to_redis(self, symbol: str, data: pd.DataFrame) -> None:
        """Met en cache les données dans Redis"""
        try:
            for timestamp, row in data.iterrows():
                feature_set = FeatureSet(
                    feature_id=self._generate_feature_id(symbol, timestamp, FeatureType.TECHNICAL),
                    symbol=symbol,
                    timestamp=timestamp,
                    features=row.to_dict(),
                    feature_type=FeatureType.TECHNICAL,
                    version="cached"
                )
                
                await self._write_to_redis(feature_set)
                
        except Exception as e:
            self.logger.error(f"Erreur mise en cache Redis", error=str(e))
    
    def _filter_features(
        self,
        data: pd.DataFrame,
        feature_names: Optional[List[str]]
    ) -> pd.DataFrame:
        """Filtre les colonnes du DataFrame"""
        if feature_names is None:
            return data
        
        # Garder toujours timestamp et symbol
        columns_to_keep = ['symbol'] if 'symbol' in data.columns else []
        columns_to_keep.extend([col for col in feature_names if col in data.columns])
        
        return data[columns_to_keep]
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compresse les données selon le type configuré"""
        if self.compression == CompressionType.GZIP:
            compressed = gzip.compress(data)
            
            # Calculer le ratio de compression
            ratio = len(data) / len(compressed)
            self.stats['compression_ratio'] = (
                0.9 * self.stats['compression_ratio'] + 0.1 * ratio
            )
            
            return compressed
        
        # Autres types de compression à implémenter
        return data
    
    def _decompress_data(self, data: bytes) -> str:
        """Décompresse les données"""
        if self.compression == CompressionType.GZIP:
            return gzip.decompress(data).decode()
        
        return data.decode()
    
    def _generate_feature_id(
        self,
        symbol: str,
        timestamp: datetime,
        feature_type: FeatureType
    ) -> str:
        """Génère un ID unique pour un feature set"""
        content = f"{symbol}:{timestamp.isoformat()}:{feature_type.value}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_current_version(self) -> str:
        """Retourne la version actuelle pour le versioning"""
        # Pour l'instant, utiliser la date
        return datetime.now(timezone.utc).strftime("%Y%m%d")
    
    def _get_default_feature_config(self) -> Dict[str, Any]:
        """Configuration par défaut pour le calcul de features"""
        return {
            'technical_indicators': {
                'sma_periods': [10, 20, 50],
                'ema_periods': [10, 20],
                'rsi_period': 14,
                'bb_period': 20,
                'bb_std': 2,
                'atr_period': 14,
                'adx_period': 14
            },
            'microstructure': {
                'calculate_spread': True,
                'calculate_imbalance': True,
                'calculate_kyle_lambda': True
            },
            'regime': {
                'lookback_period': 100,
                'n_regimes': 3
            }
        }
    
    async def delete_old_features(
        self,
        older_than: datetime,
        feature_type: Optional[FeatureType] = None
    ) -> int:
        """
        Supprime les features plus anciennes qu'une date
        
        Args:
            older_than: Date limite
            feature_type: Type spécifique à supprimer
            
        Returns:
            Nombre de features supprimées
        """
        deleted_count = 0
        
        try:
            # Supprimer de TimescaleDB
            if self.timeseries_db:
                query = "DELETE FROM features WHERE timestamp < $1"
                params = [older_than]
                
                if feature_type:
                    query += " AND feature_type = $2"
                    params.append(feature_type.value)
                
                result = await self.timeseries_db.execute(query, params)
                deleted_count = result.split()[-1] if result else 0
            
            # Nettoyer Redis (les clés expirent automatiquement)
            
            self.logger.info(
                f"Features supprimées",
                count=deleted_count,
                older_than=older_than
            )
            
        except Exception as e:
            self.logger.error(f"Erreur suppression features", error=str(e))
        
        return int(deleted_count)
    
    async def get_feature_statistics(
        self,
        symbol: str,
        feature_names: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calcule des statistiques sur les features
        
        Args:
            symbol: Symbole de l'actif
            feature_names: Noms des features
            start_time: Début de la période
            end_time: Fin de la période
            
        Returns:
            Dictionnaire avec stats par feature
        """
        # Utiliser les 30 derniers jours par défaut
        if end_time is None:
            end_time = datetime.now(timezone.utc)
        if start_time is None:
            start_time = end_time - timedelta(days=30)
        
        # Récupérer les données
        data = await self.read_features(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            feature_names=feature_names
        )
        
        if data.empty:
            return {}
        
        # Calculer les statistiques
        stats = {}
        
        for feature in feature_names:
            if feature in data.columns:
                feature_data = data[feature].dropna()
                
                if len(feature_data) > 0:
                    stats[feature] = {
                        'mean': float(feature_data.mean()),
                        'std': float(feature_data.std()),
                        'min': float(feature_data.min()),
                        'max': float(feature_data.max()),
                        'median': float(feature_data.median()),
                        'skew': float(feature_data.skew()),
                        'kurtosis': float(feature_data.kurtosis()),
                        'count': len(feature_data),
                        'missing_pct': (len(data) - len(feature_data)) / len(data) * 100
                    }
        
        return stats
    
    def get_storage_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de stockage"""
        return {
            'total_writes': self.stats['writes'],
            'total_reads': self.stats['reads'],
            'cache_hit_rate': (
                self.stats['cache_hits'] / max(self.stats['reads'], 1) * 100
            ),
            'compression_ratio': self.stats['compression_ratio'],
            'memory_cache_size': len(self._memory_cache),
            'registered_schemas': len(self._schemas)
        }
    
    async def optimize_storage(self) -> Dict[str, Any]:
        """
        Optimise le stockage (compression, nettoyage, etc.)
        
        Returns:
            Résultats de l'optimisation
        """
        results = {
            'memory_cache_cleared': 0,
            'old_features_deleted': 0,
            'indexes_optimized': False
        }
        
        try:
            # Nettoyer le cache mémoire des entrées expirées
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in self._memory_cache.items()
                if current_time - timestamp > self.cache_ttl
            ]
            
            for key in expired_keys:
                del self._memory_cache[key]
                if key in self._cache_order:
                    self._cache_order.remove(key)
            
            results['memory_cache_cleared'] = len(expired_keys)
            
            # Supprimer les vieilles features (> 90 jours)
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=90)
            deleted = await self.delete_old_features(cutoff_date)
            results['old_features_deleted'] = deleted
            
            # Optimiser les indexes TimescaleDB
            if self.timeseries_db:
                await self.timeseries_db.execute("REINDEX TABLE features")
                results['indexes_optimized'] = True
            
        except Exception as e:
            self.logger.error(f"Erreur optimisation storage", error=str(e))
        
        return results
    
    async def close(self) -> None:
        """Ferme proprement le feature store"""
        self.logger.info("Fermeture du Feature Store")
        
        # Flush les métriques
        self.metrics.flush()
        
        # Nettoyer le cache mémoire
        self._memory_cache.clear()
        self._cache_order.clear()
        
        self.logger.info("Feature Store fermé")


# Fonctions utilitaires

async def create_feature_store(
    config: Optional[TradingConfig] = None
) -> FeatureStore:
    """
    Crée une instance de FeatureStore avec la configuration par défaut
    
    Args:
        config: Configuration du trading (optionnelle)
        
    Returns:
        Instance configurée du FeatureStore
    """
    if config is None:
        config = get_config()
    
    # Initialiser Redis et TimescaleDB
    redis_cache = RedisCache(
        host=config.database.redis_host,
        port=config.database.redis_port,
        db=config.database.redis_db
    )
    
    timeseries_db = TimeSeriesDB(
        connection_string=config.get_db_url()
    )
    
    # Créer le feature store
    feature_store = FeatureStore(
        redis_cache=redis_cache,
        timeseries_db=timeseries_db,
        cache_ttl=3600,  # 1 heure
        compression=CompressionType.GZIP,
        enable_versioning=True
    )
    
    return feature_store


# Schema SQL pour TimescaleDB
FEATURE_STORE_SCHEMA = """
-- Table principale des features
CREATE TABLE IF NOT EXISTS features (
    feature_id VARCHAR(32) PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    features JSONB NOT NULL,
    feature_type VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL DEFAULT 'latest',
    metadata JSONB DEFAULT '{}',
    compressed BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convertir en hypertable
SELECT create_hypertable('features', 'timestamp', if_not_exists => TRUE);

-- Index pour requêtes rapides
CREATE INDEX IF NOT EXISTS idx_features_symbol_time 
ON features (symbol, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_features_type 
ON features (feature_type, timestamp DESC);

CREATE INDEX IF NOT EXISTS idx_features_version 
ON features (version);

-- Index GIN pour requêtes JSONB
CREATE INDEX IF NOT EXISTS idx_features_jsonb 
ON features USING gin(features);

-- Politique de rétention (garder 1 an)
SELECT add_retention_policy('features', INTERVAL '1 year', if_not_exists => TRUE);

-- Politique de compression (compresser après 7 jours)
ALTER TABLE features SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol,feature_type'
);

SELECT add_compression_policy('features', INTERVAL '7 days', if_not_exists => TRUE);
"""


if __name__ == "__main__":
    # Tests de base du Feature Store
    import asyncio
    
    async def test_feature_store():
        print("🧪 Test du Feature Store...")
        
        # Créer un feature store de test (en mémoire)
        feature_store = FeatureStore(
            redis_cache=None,  # Pas de Redis pour le test
            timeseries_db=None,  # Pas de DB pour le test
            memory_cache_size=100
        )
        
        # Données de test
        symbol = "BTCUSDT"
        timestamp = datetime.now(timezone.utc)
        
        # Features techniques de test
        features = {
            'sma_10': 45000.5,
            'sma_20': 44800.2,
            'rsi': 65.3,
            'macd': 120.5,
            'macd_signal': 115.2,
            'bb_upper': 46000.0,
            'bb_lower': 44000.0,
            'volume_sma': 1500000.0,
            'atr': 800.5
        }
        
        print("\n📝 Écriture des features...")
        feature_id = await feature_store.write_features(
            symbol=symbol,
            timestamp=timestamp,
            features=features,
            feature_type=FeatureType.TECHNICAL,
            schema_name="technical_indicators"
        )
        print(f"✅ Features écrites avec ID: {feature_id}")
        
        # Lecture des dernières features
        print("\n📖 Lecture des dernières features...")
        latest = await feature_store.read_latest_features(
            symbol=symbol,
            feature_names=['rsi', 'macd', 'sma_20']
        )
        
        if latest:
            print("✅ Features récupérées:")
            for name, value in latest.items():
                print(f"  {name}: {value}")
        
        # Test des statistiques
        print("\n📊 Statistiques de stockage:")
        stats = feature_store.get_storage_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test avec plusieurs timestamps
        print("\n📝 Écriture de série temporelle...")
        for i in range(5):
            ts = timestamp - timedelta(minutes=i)
            
            # Simuler des variations
            test_features = {
                'sma_10': 45000.5 + np.random.randn() * 100,
                'sma_20': 44800.2 + np.random.randn() * 100,
                'rsi': 65.3 + np.random.randn() * 5,
                'volume_sma': 1500000.0 + np.random.randn() * 100000
            }
            
            await feature_store.write_features(
                symbol=symbol,
                timestamp=ts,
                features=test_features,
                feature_type=FeatureType.TECHNICAL
            )
        
        print("✅ Série temporelle créée")
        
        # Lire une plage
        print("\n📖 Lecture d'une plage temporelle...")
        start_time = timestamp - timedelta(minutes=10)
        end_time = timestamp
        
        # Note: Ceci retournera un DataFrame vide car pas de backend
        data = await feature_store.read_features(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time
        )
        
        print(f"✅ Données lues: {len(data)} lignes")
        
        # Optimisation
        print("\n🔧 Optimisation du storage...")
        optimization_results = await feature_store.optimize_storage()
        print(f"✅ Résultats: {optimization_results}")
        
        await feature_store.close()
        print("\n✅ Test terminé!")
    
    # Exécuter le test
    asyncio.run(test_feature_store())