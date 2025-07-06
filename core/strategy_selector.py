"""
Sélecteur Intelligent de Stratégies pour Robot de Trading Algorithmique IA
==========================================================================

Ce module implémente un sélecteur de stratégies sophistiqué utilisant l'apprentissage
automatique pour détecter les régimes de marché et optimiser dynamiquement l'allocation
des stratégies. Support pour hot-swapping et adaptation temps réel.

Architecture:
- Détection de régimes de marché par ML (clustering, HMM, ensemble)
- Sélection adaptative basée sur performance historique et conditions
- Hot-swapping sans interruption du trading
- Allocation de capital dynamique entre stratégies
- Meta-learning pour optimisation continue
- Registry pattern avec factory pour extensibilité
- Monitoring de performance avec alertes automatiques

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import asyncio
import inspect
import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Set, Union, Callable, 
    Type, Protocol, Tuple, NamedTuple
)
import importlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Imports internes
from config.settings import TradingConfig, StrategyType
from utils.logger import get_structured_logger, log_context
from utils.decorators import retry_async, circuit_breaker, measure_performance
from utils.metrics import MetricsCollector
from monitoring.alerts import AlertManager, AlertSeverity, AlertCategory


class MarketRegime(Enum):
    """Types de régimes de marché détectés"""
    TRENDING_BULL = "trending_bull"       # Tendance haussière forte
    TRENDING_BEAR = "trending_bear"       # Tendance baissière forte  
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"  # Latéral volatil
    SIDEWAYS_LOW_VOL = "sideways_low_vol"    # Latéral calme
    VOLATILE_CHAOTIC = "volatile_chaotic"     # Volatilité extrême
    BREAKOUT_BULL = "breakout_bull"          # Cassure haussière
    BREAKOUT_BEAR = "breakout_bear"          # Cassure baissière
    UNKNOWN = "unknown"                      # Régime indéterminé


class StrategyStatus(Enum):
    """États possibles d'une stratégie"""
    INACTIVE = "inactive"         # Inactive
    LOADING = "loading"          # En cours de chargement
    ACTIVE = "active"            # Active et trading
    PAUSED = "paused"           # En pause
    STOPPING = "stopping"       # En cours d'arrêt
    ERROR = "error"             # En erreur
    HOT_SWAPPING = "hot_swapping"  # En cours de hot-swap


@dataclass
class MarketFeatures:
    """Features de marché pour détection de régime"""
    timestamp: float
    
    # Prix et volatilité
    price: float
    returns_1m: float
    returns_5m: float
    returns_15m: float
    volatility_1h: float
    volatility_4h: float
    volatility_24h: float
    
    # Tendance et momentum
    sma_20: float
    sma_50: float
    sma_200: float
    rsi_14: float
    macd: float
    macd_signal: float
    
    # Volume et liquidité
    volume: float
    volume_sma_20: float
    bid_ask_spread: float
    order_book_imbalance: float
    
    # Microstructure
    trade_count_1m: int
    avg_trade_size: float
    price_impact: float
    
    # Correlations et sentiment
    btc_correlation: float = 0.0
    fear_greed_index: float = 50.0
    funding_rate: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convertit en array numpy pour ML"""
        return np.array([
            self.returns_1m, self.returns_5m, self.returns_15m,
            self.volatility_1h, self.volatility_4h, self.volatility_24h,
            self.sma_20, self.sma_50, self.sma_200,
            self.rsi_14, self.macd, self.macd_signal,
            self.volume, self.volume_sma_20,
            self.bid_ask_spread, self.order_book_imbalance,
            self.trade_count_1m, self.avg_trade_size, self.price_impact,
            self.btc_correlation, self.fear_greed_index, self.funding_rate
        ])


@dataclass
class StrategyPerformance:
    """Métriques de performance d'une stratégie"""
    strategy_id: str
    
    # Performance financière
    total_pnl: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Métriques opérationnelles
    total_trades: int = 0
    avg_trade_duration_minutes: float = 0.0
    success_rate: float = 1.0  # Taux de succès technique
    avg_execution_time_ms: float = 0.0
    
    # Performance par régime
    performance_by_regime: Dict[MarketRegime, float] = field(default_factory=dict)
    
    # Historique
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    performance_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    
    def update_performance(self, pnl: float, regime: MarketRegime):
        """Met à jour les métriques de performance"""
        self.total_pnl += pnl
        self.last_updated = datetime.now(timezone.utc)
        
        # Mise à jour performance par régime
        if regime not in self.performance_by_regime:
            self.performance_by_regime[regime] = 0.0
        self.performance_by_regime[regime] += pnl
        
        # Historique
        self.performance_history.append({
            'timestamp': time.time(),
            'pnl': pnl,
            'regime': regime.value,
            'cumulative_pnl': self.total_pnl
        })
    
    def get_regime_fitness(self, regime: MarketRegime) -> float:
        """Calcule la fitness d'une stratégie pour un régime donné"""
        if regime not in self.performance_by_regime:
            return 0.0
        
        # Combinaison de performance et success rate
        regime_pnl = self.performance_by_regime[regime]
        fitness = regime_pnl * self.success_rate * (1 - self.max_drawdown)
        
        # Bonus pour Sharpe ratio élevé
        if self.sharpe_ratio > 1.0:
            fitness *= (1 + self.sharpe_ratio * 0.1)
        
        return max(0.0, fitness)


class StrategyProtocol(Protocol):
    """Interface que doivent implémenter toutes les stratégies"""
    
    strategy_id: str
    strategy_type: StrategyType
    
    async def start(self) -> None:
        """Démarre la stratégie"""
        ...
    
    async def stop(self) -> None:
        """Arrête la stratégie"""
        ...
    
    async def pause(self) -> None:
        """Met en pause la stratégie"""
        ...
    
    async def resume(self) -> None:
        """Reprend la stratégie"""
        ...
    
    def get_status(self) -> StrategyStatus:
        """Retourne le statut actuel"""
        ...
    
    def get_performance(self) -> StrategyPerformance:
        """Retourne les métriques de performance"""
        ...
    
    async def on_market_data(self, event: Any) -> None:
        """Traite les données de marché"""
        ...


class MarketRegimeDetector:
    """Détecteur de régimes de marché par ML"""
    
    def __init__(self, lookback_periods: int = 200, n_regimes: int = 7):
        self.lookback_periods = lookback_periods
        self.n_regimes = n_regimes
        self.logger = get_structured_logger("market_regime_detector")
        
        # Modèles ML
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        
        # Données historiques
        self.features_history: deque = deque(maxlen=1000)
        self.regime_history: deque = deque(maxlen=1000)
        
        # État du modèle
        self.is_trained = False
        self.last_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        
        # Cache et performance
        self._last_detection_time = 0.0
        self._detection_cache = {}
        
        # Mapping des clusters aux régimes
        self.cluster_to_regime = {}
    
    def add_market_features(self, features: MarketFeatures):
        """Ajoute des features de marché à l'historique"""
        self.features_history.append(features)
        
        # Réentraîner si assez de données
        if len(self.features_history) >= self.lookback_periods and not self.is_trained:
            self._train_models()
        elif len(self.features_history) >= self.lookback_periods and len(self.features_history) % 50 == 0:
            # Réentraînement périodique
            self._retrain_models()
    
    def _train_models(self):
        """Entraîne les modèles de détection de régime"""
        try:
            if len(self.features_history) < self.lookback_periods:
                return
            
            # Préparer les données
            features_array = np.array([f.to_array() for f in self.features_history])
            
            # Normalisation
            features_scaled = self.scaler.fit_transform(features_array)
            
            # Clustering K-means pour identifier les régimes
            clusters = self.kmeans.fit_predict(features_scaled)
            
            # Mapper les clusters aux régimes basé sur les caractéristiques
            self._map_clusters_to_regimes(features_array, clusters)
            
            # Entraîner le classificateur
            regime_labels = [self._cluster_to_regime_label(c) for c in clusters]
            self.rf_classifier.fit(features_scaled, regime_labels)
            
            self.is_trained = True
            
            # Évaluer la qualité du clustering
            silhouette_avg = silhouette_score(features_scaled, clusters)
            
            self.logger.info("market_regime_models_trained",
                           lookback_periods=len(self.features_history),
                           n_clusters=self.n_regimes,
                           silhouette_score=silhouette_avg)
            
        except Exception as e:
            self.logger.error("market_regime_training_error", error=str(e))
    
    def _retrain_models(self):
        """Réentraîne les modèles avec nouvelles données"""
        try:
            # Utiliser les dernières données pour adaptation
            recent_features = list(self.features_history)[-self.lookback_periods:]
            features_array = np.array([f.to_array() for f in recent_features])
            
            # Réentraînement incrémental
            features_scaled = self.scaler.transform(features_array)
            clusters = self.kmeans.predict(features_scaled)
            
            # Mise à jour du mapping si nécessaire
            self._update_cluster_mapping(features_array, clusters)
            
            self.logger.debug("market_regime_models_retrained",
                            data_points=len(recent_features))
            
        except Exception as e:
            self.logger.error("market_regime_retraining_error", error=str(e))
    
    def _map_clusters_to_regimes(self, features: np.ndarray, clusters: np.ndarray):
        """Mappe les clusters aux régimes de marché"""
        cluster_stats = {}
        
        # Calculer les statistiques par cluster
        for cluster_id in range(self.n_regimes):
            mask = clusters == cluster_id
            if not np.any(mask):
                continue
            
            cluster_features = features[mask]
            
            # Indices des features importantes
            returns_1m_idx = 0
            volatility_1h_idx = 3
            rsi_idx = 9
            volume_idx = 12
            
            avg_returns = np.mean(cluster_features[:, returns_1m_idx])
            avg_volatility = np.mean(cluster_features[:, volatility_1h_idx])
            avg_rsi = np.mean(cluster_features[:, rsi_idx])
            avg_volume = np.mean(cluster_features[:, volume_idx])
            
            cluster_stats[cluster_id] = {
                'returns': avg_returns,
                'volatility': avg_volatility,
                'rsi': avg_rsi,
                'volume': avg_volume,
                'count': np.sum(mask)
            }
        
        # Mapping basé sur les caractéristiques
        for cluster_id, stats in cluster_stats.items():
            regime = self._classify_regime(stats)
            self.cluster_to_regime[cluster_id] = regime
        
        self.logger.debug("cluster_regime_mapping", mapping=self.cluster_to_regime)
    
    def _classify_regime(self, stats: Dict[str, float]) -> MarketRegime:
        """Classifie un cluster en régime de marché"""
        returns = stats['returns']
        volatility = stats['volatility']
        rsi = stats['rsi']
        
        # Seuils adaptatifs
        high_vol_threshold = 0.02  # 2% volatilité horaire
        trend_threshold = 0.001    # 0.1% returns
        
        if volatility > high_vol_threshold * 2:
            return MarketRegime.VOLATILE_CHAOTIC
        elif returns > trend_threshold and rsi > 60:
            if volatility > high_vol_threshold:
                return MarketRegime.BREAKOUT_BULL
            else:
                return MarketRegime.TRENDING_BULL
        elif returns < -trend_threshold and rsi < 40:
            if volatility > high_vol_threshold:
                return MarketRegime.BREAKOUT_BEAR
            else:
                return MarketRegime.TRENDING_BEAR
        elif volatility > high_vol_threshold:
            return MarketRegime.SIDEWAYS_HIGH_VOL
        else:
            return MarketRegime.SIDEWAYS_LOW_VOL
    
    def _cluster_to_regime_label(self, cluster: int) -> int:
        """Convertit un cluster en label numérique pour le classificateur"""
        regime = self.cluster_to_regime.get(cluster, MarketRegime.UNKNOWN)
        return list(MarketRegime).index(regime)
    
    def _update_cluster_mapping(self, features: np.ndarray, clusters: np.ndarray):
        """Met à jour le mapping des clusters (apprentissage adaptatif)"""
        # Pour l'instant, utiliser le mapping existant
        # Future amélioration : apprentissage adaptatif du mapping
        pass
    
    def detect_regime(self, current_features: MarketFeatures) -> Tuple[MarketRegime, float]:
        """Détecte le régime de marché actuel"""
        if not self.is_trained:
            return MarketRegime.UNKNOWN, 0.0
        
        try:
            # Cache pour éviter les détections trop fréquentes
            current_time = time.time()
            if current_time - self._last_detection_time < 10.0:  # Cache 10 secondes
                return self.last_regime, self.regime_confidence
            
            # Préparer les features
            features_array = current_features.to_array().reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            # Prédiction avec le classificateur
            regime_proba = self.rf_classifier.predict_proba(features_scaled)[0]
            regime_label = np.argmax(regime_proba)
            confidence = np.max(regime_proba)
            
            # Convertir en régime
            regime = list(MarketRegime)[regime_label]
            
            # Mise à jour de l'état
            self.last_regime = regime
            self.regime_confidence = confidence
            self._last_detection_time = current_time
            
            # Historique
            self.regime_history.append({
                'timestamp': current_time,
                'regime': regime.value,
                'confidence': confidence
            })
            
            return regime, confidence
            
        except Exception as e:
            self.logger.error("regime_detection_error", error=str(e))
            return MarketRegime.UNKNOWN, 0.0
    
    def get_regime_distribution(self, window_minutes: int = 60) -> Dict[MarketRegime, float]:
        """Retourne la distribution des régimes sur une fenêtre temporelle"""
        cutoff = time.time() - (window_minutes * 60)
        recent_regimes = [
            r for r in self.regime_history 
            if r['timestamp'] >= cutoff
        ]
        
        if not recent_regimes:
            return {}
        
        regime_counts = defaultdict(int)
        for r in recent_regimes:
            regime = MarketRegime(r['regime'])
            regime_counts[regime] += 1
        
        total = len(recent_regimes)
        return {regime: count / total for regime, count in regime_counts.items()}


class StrategyRegistry:
    """Registry des stratégies disponibles avec factory pattern"""
    
    def __init__(self):
        self.strategies: Dict[StrategyType, Type[StrategyProtocol]] = {}
        self.strategy_configs: Dict[str, Dict[str, Any]] = {}
        self.logger = get_structured_logger("strategy_registry")
    
    def register_strategy(self, strategy_type: StrategyType, 
                         strategy_class: Type[StrategyProtocol],
                         config: Optional[Dict[str, Any]] = None):
        """Enregistre une stratégie dans le registry"""
        self.strategies[strategy_type] = strategy_class
        if config:
            self.strategy_configs[strategy_type.value] = config
        
        self.logger.info("strategy_registered", 
                        strategy_type=strategy_type.value,
                        strategy_class=strategy_class.__name__)
    
    def create_strategy(self, strategy_type: StrategyType, 
                       strategy_id: str, **kwargs) -> StrategyProtocol:
        """Crée une instance de stratégie"""
        if strategy_type not in self.strategies:
            raise ValueError(f"Strategy type {strategy_type.value} not registered")
        
        strategy_class = self.strategies[strategy_type]
        
        # Merge configuration
        config = self.strategy_configs.get(strategy_type.value, {})
        config.update(kwargs)
        
        # Créer l'instance
        if 'strategy_id' not in config:
            config['strategy_id'] = strategy_id
        
        return strategy_class(**config)
    
    def get_available_strategies(self) -> List[StrategyType]:
        """Retourne la liste des stratégies disponibles"""
        return list(self.strategies.keys())


class StrategySelector:
    """
    Sélecteur intelligent de stratégies avec ML et adaptation temps réel
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = get_structured_logger("strategy_selector")
        
        # Composants internes
        self.regime_detector = MarketRegimeDetector()
        self.strategy_registry = StrategyRegistry()
        self.metrics_collector = MetricsCollector("strategy_selector")
        self.alert_manager = AlertManager()
        
        # État des stratégies
        self.active_strategies: Dict[str, StrategyProtocol] = {}
        self.strategy_performances: Dict[str, StrategyPerformance] = {}
        self.strategy_allocations: Dict[str, float] = {}  # % du capital
        
        # Sélection et optimisation
        self.current_regime = MarketRegime.UNKNOWN
        self.regime_confidence = 0.0
        self.last_rebalance = datetime.now(timezone.utc)
        self.rebalance_interval = timedelta(minutes=15)  # Rééquilibrage toutes les 15min
        
        # Meta-learning
        self.regime_strategy_performance: Dict[Tuple[MarketRegime, str], deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        
        # Configuration et limites
        self.max_active_strategies = 3
        self.min_strategy_allocation = 0.05  # 5% minimum
        self.max_strategy_allocation = 0.60  # 60% maximum
        
        # Background tasks
        self._running = False
        self._monitor_task = None
        self._rebalance_task = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Initialisation
        self._init_default_strategies()
    
    def _init_default_strategies(self):
        """Initialise les stratégies par défaut depuis la configuration"""
        for strategy_id, strategy_config in self.config.strategies.items():
            strategy_type = strategy_config.strategy_type
            
            # Pour l'instant, créer des performances vides
            # Les vraies stratégies seront injectées via le registry
            performance = StrategyPerformance(strategy_id=strategy_id)
            self.strategy_performances[strategy_id] = performance
            
            # Allocation initiale depuis la config
            self.strategy_allocations[strategy_id] = float(strategy_config.capital_allocation)
        
        self.logger.info("default_strategies_initialized", 
                        strategies=list(self.strategy_performances.keys()))
    
    async def start(self):
        """Démarre le sélecteur de stratégies"""
        if self._running:
            return
        
        self._running = True
        
        # Démarrer les tâches de surveillance
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self._rebalance_task = asyncio.create_task(self._rebalancing_loop())
        
        self.logger.info("strategy_selector_started")
    
    async def stop(self):
        """Arrête le sélecteur de stratégies"""
        self._running = False
        
        # Arrêter toutes les stratégies actives
        for strategy_id, strategy in list(self.active_strategies.items()):
            await self._stop_strategy(strategy_id)
        
        # Arrêter les tâches
        if self._monitor_task:
            self._monitor_task.cancel()
        if self._rebalance_task:
            self._rebalance_task.cancel()
        
        self.logger.info("strategy_selector_stopped")
    
    @measure_performance(track_memory=True)
    async def process_market_data(self, market_data: Dict[str, Any]):
        """Traite les données de marché et met à jour la détection de régime"""
        try:
            # Extraire les features de marché
            features = self._extract_market_features(market_data)
            
            # Ajouter à l'historique pour entraînement
            self.regime_detector.add_market_features(features)
            
            # Détecter le régime actuel
            regime, confidence = self.regime_detector.detect_regime(features)
            
            # Mise à jour si changement de régime
            if regime != self.current_regime and confidence > 0.7:
                await self._handle_regime_change(regime, confidence)
            
            self.current_regime = regime
            self.regime_confidence = confidence
            
            # Métriques
            self.metrics_collector.gauge("current_regime", list(MarketRegime).index(regime))
            self.metrics_collector.gauge("regime_confidence", confidence)
            
            # Transférer les données aux stratégies actives
            await self._distribute_market_data_to_strategies(market_data)
            
        except Exception as e:
            self.logger.error("market_data_processing_error", error=str(e))
            await self.alert_manager.send_warning_alert(
                "Market Data Processing Error",
                f"Error processing market data: {str(e)}"
            )
    
    def _extract_market_features(self, market_data: Dict[str, Any]) -> MarketFeatures:
        """Extrait les features de marché depuis les données brutes"""
        # Cette fonction sera étendue selon le format des données de marché
        # Pour l'instant, une implémentation basique
        
        return MarketFeatures(
            timestamp=time.time(),
            price=market_data.get('price', 0.0),
            returns_1m=market_data.get('returns_1m', 0.0),
            returns_5m=market_data.get('returns_5m', 0.0),
            returns_15m=market_data.get('returns_15m', 0.0),
            volatility_1h=market_data.get('volatility_1h', 0.0),
            volatility_4h=market_data.get('volatility_4h', 0.0),
            volatility_24h=market_data.get('volatility_24h', 0.0),
            sma_20=market_data.get('sma_20', 0.0),
            sma_50=market_data.get('sma_50', 0.0),
            sma_200=market_data.get('sma_200', 0.0),
            rsi_14=market_data.get('rsi_14', 50.0),
            macd=market_data.get('macd', 0.0),
            macd_signal=market_data.get('macd_signal', 0.0),
            volume=market_data.get('volume', 0.0),
            volume_sma_20=market_data.get('volume_sma_20', 0.0),
            bid_ask_spread=market_data.get('bid_ask_spread', 0.0),
            order_book_imbalance=market_data.get('order_book_imbalance', 0.0),
            trade_count_1m=market_data.get('trade_count_1m', 0),
            avg_trade_size=market_data.get('avg_trade_size', 0.0),
            price_impact=market_data.get('price_impact', 0.0)
        )
    
    async def _handle_regime_change(self, new_regime: MarketRegime, confidence: float):
        """Gère un changement de régime de marché"""
        old_regime = self.current_regime
        
        self.logger.info("market_regime_changed",
                        old_regime=old_regime.value,
                        new_regime=new_regime.value,
                        confidence=confidence)
        
        # Alerte sur changement de régime significatif
        await self.alert_manager.send_alert(
            AlertSeverity.INFO,
            AlertCategory.STRATEGY,
            "Market Regime Change",
            f"Market regime changed from {old_regime.value} to {new_regime.value} "
            f"(confidence: {confidence:.2f})",
            source="strategy_selector",
            metrics={
                "old_regime": old_regime.value,
                "new_regime": new_regime.value,
                "confidence": confidence
            }
        )
        
        # Déclencher rééquilibrage immédiat
        await self._rebalance_strategies(force=True)
        
        # Métriques
        self.metrics_collector.increment("regime_changes_total", 
                                        tags={"from": old_regime.value, "to": new_regime.value})
    
    async def _distribute_market_data_to_strategies(self, market_data: Dict[str, Any]):
        """Distribue les données de marché aux stratégies actives"""
        tasks = []
        
        with self._lock:
            for strategy_id, strategy in self.active_strategies.items():
                if strategy.get_status() == StrategyStatus.ACTIVE:
                    task = asyncio.create_task(
                        self._safe_strategy_call(strategy, 'on_market_data', market_data)
                    )
                    tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _safe_strategy_call(self, strategy: StrategyProtocol, 
                                 method_name: str, *args, **kwargs):
        """Appel sécurisé d'une méthode de stratégie avec gestion d'erreur"""
        try:
            method = getattr(strategy, method_name)
            if asyncio.iscoroutinefunction(method):
                return await method(*args, **kwargs)
            else:
                return method(*args, **kwargs)
        except Exception as e:
            self.logger.error("strategy_method_call_error",
                            strategy_id=strategy.strategy_id,
                            method=method_name,
                            error=str(e))
            
            # Alertes pour erreurs critiques
            await self.alert_manager.send_warning_alert(
                f"Strategy {strategy.strategy_id} Error",
                f"Error calling {method_name}: {str(e)}"
            )
    
    @circuit_breaker(failure_threshold=3, recovery_timeout=60.0)
    async def register_strategy_instance(self, strategy: StrategyProtocol):
        """Enregistre une instance de stratégie"""
        strategy_id = strategy.strategy_id
        
        with self._lock:
            # Arrêter l'ancienne stratégie si elle existe
            if strategy_id in self.active_strategies:
                await self._stop_strategy(strategy_id)
            
            # Enregistrer la nouvelle stratégie
            self.active_strategies[strategy_id] = strategy
            
            # Initialiser les métriques de performance si nécessaire
            if strategy_id not in self.strategy_performances:
                self.strategy_performances[strategy_id] = StrategyPerformance(strategy_id)
            
            # Allocation par défaut
            if strategy_id not in self.strategy_allocations:
                self.strategy_allocations[strategy_id] = self.min_strategy_allocation
        
        self.logger.info("strategy_registered",
                        strategy_id=strategy_id,
                        strategy_type=strategy.strategy_type.value)
    
    async def hot_swap_strategy(self, old_strategy_id: str, new_strategy: StrategyProtocol) -> bool:
        """Hot-swap d'une stratégie sans interruption"""
        try:
            with log_context(correlation_id=f"hotswap_{old_strategy_id}"):
                self.logger.info("strategy_hot_swap_starting",
                               old_strategy_id=old_strategy_id,
                               new_strategy_id=new_strategy.strategy_id)
                
                with self._lock:
                    # Marquer l'ancienne stratégie en hot-swapping
                    if old_strategy_id in self.active_strategies:
                        old_strategy = self.active_strategies[old_strategy_id]
                        # Mettre en pause plutôt qu'arrêter
                        await self._safe_strategy_call(old_strategy, 'pause')
                    
                    # Transférer l'allocation
                    if old_strategy_id in self.strategy_allocations:
                        allocation = self.strategy_allocations[old_strategy_id]
                        self.strategy_allocations[new_strategy.strategy_id] = allocation
                        del self.strategy_allocations[old_strategy_id]
                    
                    # Transférer les performances
                    if old_strategy_id in self.strategy_performances:
                        old_perf = self.strategy_performances[old_strategy_id]
                        new_perf = StrategyPerformance(strategy_id=new_strategy.strategy_id)
                        # Copier certaines métriques importantes
                        new_perf.performance_by_regime = old_perf.performance_by_regime.copy()
                        self.strategy_performances[new_strategy.strategy_id] = new_perf
                        del self.strategy_performances[old_strategy_id]
                    
                    # Remplacer la stratégie
                    self.active_strategies[new_strategy.strategy_id] = new_strategy
                    if old_strategy_id in self.active_strategies and old_strategy_id != new_strategy.strategy_id:
                        del self.active_strategies[old_strategy_id]
                    
                    # Démarrer la nouvelle stratégie
                    await self._safe_strategy_call(new_strategy, 'start')
                
                self.logger.info("strategy_hot_swap_completed",
                               old_strategy_id=old_strategy_id,
                               new_strategy_id=new_strategy.strategy_id)
                
                # Alerte de succès
                await self.alert_manager.send_alert(
                    AlertSeverity.INFO,
                    AlertCategory.STRATEGY,
                    "Strategy Hot-Swap Completed",
                    f"Successfully swapped {old_strategy_id} with {new_strategy.strategy_id}",
                    source="strategy_selector"
                )
                
                return True
                
        except Exception as e:
            self.logger.error("strategy_hot_swap_failed",
                            old_strategy_id=old_strategy_id,
                            new_strategy_id=new_strategy.strategy_id,
                            error=str(e))
            
            await self.alert_manager.send_critical_alert(
                "Strategy Hot-Swap Failed",
                f"Failed to hot-swap {old_strategy_id}: {str(e)}"
            )
            
            return False
    
    async def _stop_strategy(self, strategy_id: str):
        """Arrête une stratégie en toute sécurité"""
        if strategy_id not in self.active_strategies:
            return
        
        strategy = self.active_strategies[strategy_id]
        
        try:
            await self._safe_strategy_call(strategy, 'stop')
            
            with self._lock:
                del self.active_strategies[strategy_id]
            
            self.logger.info("strategy_stopped", strategy_id=strategy_id)
            
        except Exception as e:
            self.logger.error("strategy_stop_error",
                            strategy_id=strategy_id,
                            error=str(e))
    
    async def _monitoring_loop(self):
        """Boucle de surveillance des stratégies"""
        while self._running:
            try:
                await self._monitor_strategy_performance()
                await asyncio.sleep(30)  # Surveillance toutes les 30 secondes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("strategy_monitoring_error", error=str(e))
                await asyncio.sleep(60)
    
    async def _monitor_strategy_performance(self):
        """Surveille les performances des stratégies"""
        current_time = datetime.now(timezone.utc)
        
        with self._lock:
            for strategy_id, strategy in self.active_strategies.items():
                try:
                    # Récupérer les performances
                    performance = await self._safe_strategy_call(strategy, 'get_performance')
                    if performance:
                        self.strategy_performances[strategy_id] = performance
                        
                        # Métriques
                        self.metrics_collector.gauge(f"strategy_pnl", performance.total_pnl,
                                                   tags={"strategy_id": strategy_id})
                        self.metrics_collector.gauge(f"strategy_sharpe", performance.sharpe_ratio,
                                                   tags={"strategy_id": strategy_id})
                        self.metrics_collector.gauge(f"strategy_drawdown", performance.max_drawdown,
                                                   tags={"strategy_id": strategy_id})
                        
                        # Alertes sur performance dégradée
                        if performance.sharpe_ratio < 0.5 and performance.total_trades > 10:
                            await self.alert_manager.send_warning_alert(
                                f"Poor Strategy Performance: {strategy_id}",
                                f"Sharpe ratio {performance.sharpe_ratio:.2f} below threshold",
                                strategy_id=strategy_id,
                                metrics={"sharpe_ratio": performance.sharpe_ratio}
                            )
                        
                        # Alerte sur drawdown excessif
                        if performance.max_drawdown > 0.2:  # 20%
                            await self.alert_manager.send_critical_alert(
                                f"Excessive Drawdown: {strategy_id}",
                                f"Max drawdown {performance.max_drawdown:.1%} exceeds limit",
                                strategy_id=strategy_id
                            )
                
                except Exception as e:
                    self.logger.error("strategy_performance_monitoring_error",
                                    strategy_id=strategy_id,
                                    error=str(e))
    
    async def _rebalancing_loop(self):
        """Boucle de rééquilibrage des stratégies"""
        while self._running:
            try:
                # Vérifier si rééquilibrage nécessaire
                current_time = datetime.now(timezone.utc)
                if current_time - self.last_rebalance >= self.rebalance_interval:
                    await self._rebalance_strategies()
                
                await asyncio.sleep(60)  # Vérifier toutes les minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("rebalancing_loop_error", error=str(e))
                await asyncio.sleep(300)  # 5 minutes en cas d'erreur
    
    async def _rebalance_strategies(self, force: bool = False):
        """Rééquilibre l'allocation entre stratégies"""
        current_time = datetime.now(timezone.utc)
        
        if not force and current_time - self.last_rebalance < self.rebalance_interval:
            return
        
        try:
            self.logger.info("strategy_rebalancing_started", regime=self.current_regime.value)
            
            # Calculer les nouvelles allocations
            new_allocations = self._calculate_optimal_allocations()
            
            # Appliquer les changements significatifs
            changes_made = False
            with self._lock:
                for strategy_id, new_allocation in new_allocations.items():
                    old_allocation = self.strategy_allocations.get(strategy_id, 0.0)
                    
                    # Appliquer si changement > 5%
                    if abs(new_allocation - old_allocation) > 0.05:
                        self.strategy_allocations[strategy_id] = new_allocation
                        changes_made = True
                        
                        self.logger.info("strategy_allocation_changed",
                                       strategy_id=strategy_id,
                                       old_allocation=old_allocation,
                                       new_allocation=new_allocation)
            
            if changes_made:
                # Normaliser pour s'assurer que la somme = 1.0
                total_allocation = sum(self.strategy_allocations.values())
                if total_allocation > 0:
                    for strategy_id in self.strategy_allocations:
                        self.strategy_allocations[strategy_id] /= total_allocation
                
                # Alerter des changements importants
                await self.alert_manager.send_alert(
                    AlertSeverity.INFO,
                    AlertCategory.STRATEGY,
                    "Strategy Allocation Rebalanced",
                    f"Rebalanced allocations for regime {self.current_regime.value}",
                    source="strategy_selector",
                    metrics=dict(self.strategy_allocations)
                )
            
            self.last_rebalance = current_time
            
        except Exception as e:
            self.logger.error("strategy_rebalancing_error", error=str(e))
            await self.alert_manager.send_warning_alert(
                "Strategy Rebalancing Error",
                f"Error during rebalancing: {str(e)}"
            )
    
    def _calculate_optimal_allocations(self) -> Dict[str, float]:
        """Calcule les allocations optimales basées sur les performances par régime"""
        allocations = {}
        
        # Calculer les fitness scores pour le régime actuel
        fitness_scores = {}
        total_fitness = 0.0
        
        for strategy_id, performance in self.strategy_performances.items():
            if strategy_id in self.active_strategies:
                fitness = performance.get_regime_fitness(self.current_regime)
                # Bonus pour stratégies récemment performantes
                if performance.total_trades > 0:
                    recent_performance = sum(
                        h['pnl'] for h in list(performance.performance_history)[-10:]
                    )
                    fitness += recent_performance * 0.1
                
                fitness_scores[strategy_id] = max(0.01, fitness)  # Minimum positif
                total_fitness += fitness_scores[strategy_id]
        
        # Allocation proportionnelle aux fitness scores
        if total_fitness > 0:
            for strategy_id, fitness in fitness_scores.items():
                base_allocation = fitness / total_fitness
                
                # Appliquer les contraintes
                allocation = max(self.min_strategy_allocation, 
                               min(self.max_strategy_allocation, base_allocation))
                allocations[strategy_id] = allocation
        else:
            # Allocation égale si pas de données de performance
            n_strategies = len(self.active_strategies)
            if n_strategies > 0:
                equal_allocation = 1.0 / n_strategies
                for strategy_id in self.active_strategies:
                    allocations[strategy_id] = equal_allocation
        
        return allocations
    
    def get_strategy_status(self) -> Dict[str, Any]:
        """Retourne le statut complet du sélecteur"""
        with self._lock:
            regime_distribution = self.regime_detector.get_regime_distribution()
            
            strategy_info = {}
            for strategy_id, strategy in self.active_strategies.items():
                performance = self.strategy_performances.get(strategy_id)
                strategy_info[strategy_id] = {
                    "status": strategy.get_status().value,
                    "allocation": self.strategy_allocations.get(strategy_id, 0.0),
                    "total_pnl": performance.total_pnl if performance else 0.0,
                    "sharpe_ratio": performance.sharpe_ratio if performance else 0.0,
                    "total_trades": performance.total_trades if performance else 0
                }
            
            return {
                "current_regime": self.current_regime.value,
                "regime_confidence": self.regime_confidence,
                "regime_distribution_1h": {r.value: p for r, p in regime_distribution.items()},
                "active_strategies": len(self.active_strategies),
                "strategy_details": strategy_info,
                "last_rebalance": self.last_rebalance.isoformat(),
                "total_allocation": sum(self.strategy_allocations.values())
            }


# Factory function pour simplifier la création
async def create_strategy_selector(config: TradingConfig) -> StrategySelector:
    """Factory function pour créer et configurer le sélecteur de stratégies"""
    selector = StrategySelector(config)
    await selector.start()
    return selector


# Exports principaux
__all__ = [
    'StrategySelector',
    'MarketRegimeDetector',
    'StrategyRegistry',
    'StrategyProtocol',
    'MarketRegime',
    'StrategyStatus',
    'MarketFeatures',
    'StrategyPerformance',
    'create_strategy_selector'
]


if __name__ == "__main__":
    # Test du sélecteur de stratégies
    import asyncio
    from config.settings import create_development_config
    
    # Mock strategy pour test
    class MockStrategy:
        def __init__(self, strategy_id: str, strategy_type: StrategyType = StrategyType.SCALPING):
            self.strategy_id = strategy_id
            self.strategy_type = strategy_type
            self._status = StrategyStatus.INACTIVE
            self._performance = StrategyPerformance(strategy_id)
        
        async def start(self):
            self._status = StrategyStatus.ACTIVE
            
        async def stop(self):
            self._status = StrategyStatus.INACTIVE
            
        async def pause(self):
            self._status = StrategyStatus.PAUSED
            
        async def resume(self):
            self._status = StrategyStatus.ACTIVE
            
        def get_status(self):
            return self._status
            
        def get_performance(self):
            return self._performance
            
        async def on_market_data(self, event):
            # Simuler un trade avec PnL aléatoire
            import random
            pnl = random.uniform(-10, 20)
            self._performance.update_performance(pnl, MarketRegime.TRENDING_BULL)
    
    async def test_strategy_selector():
        print("🚀 Testing Strategy Selector System...")
        
        # Configuration
        config = create_development_config()
        
        # Créer le sélecteur
        selector = StrategySelector(config)
        await selector.start()
        
        # Créer et enregistrer des stratégies de test
        strategy1 = MockStrategy("test_strategy_1", StrategyType.SCALPING)
        strategy2 = MockStrategy("test_strategy_2", StrategyType.MARKET_MAKING)
        
        await selector.register_strategy_instance(strategy1)
        await selector.register_strategy_instance(strategy2)
        
        print(f"✅ Strategies registered: {len(selector.active_strategies)}")
        
        # Simuler des données de marché
        market_data = {
            'price': 45000.0,
            'returns_1m': 0.001,
            'returns_5m': 0.005,
            'returns_15m': 0.01,
            'volatility_1h': 0.02,
            'volatility_4h': 0.025,
            'volatility_24h': 0.03,
            'sma_20': 44800.0,
            'sma_50': 44500.0,
            'sma_200': 43000.0,
            'rsi_14': 65.0,
            'macd': 100.0,
            'macd_signal': 80.0,
            'volume': 1000000.0,
            'volume_sma_20': 800000.0,
            'bid_ask_spread': 1.0,
            'order_book_imbalance': 0.1,
            'trade_count_1m': 50,
            'avg_trade_size': 2.5,
            'price_impact': 0.001
        }
        
        # Traiter les données de marché plusieurs fois
        for i in range(10):
            # Varier légèrement les données
            market_data['price'] += random.uniform(-50, 50)
            market_data['returns_1m'] = random.uniform(-0.002, 0.002)
            market_data['volatility_1h'] = random.uniform(0.01, 0.05)
            
            await selector.process_market_data(market_data)
            await asyncio.sleep(0.1)
        
        print(f"✅ Market data processed, current regime: {selector.current_regime.value}")
        
        # Test hot-swap
        strategy3 = MockStrategy("test_strategy_3", StrategyType.STATISTICAL_ARBITRAGE)
        success = await selector.hot_swap_strategy("test_strategy_1", strategy3)
        print(f"✅ Hot-swap {'successful' if success else 'failed'}")
        
        # Attendre un peu pour les tâches de surveillance
        await asyncio.sleep(3.0)
        
        # Statut final
        status = selector.get_strategy_status()
        print(f"\n📊 Strategy Selector Status:")
        print(f"  Current regime: {status['current_regime']} (confidence: {status['regime_confidence']:.2f})")
        print(f"  Active strategies: {status['active_strategies']}")
        print(f"  Total allocation: {status['total_allocation']:.2f}")
        
        for strategy_id, details in status['strategy_details'].items():
            print(f"  {strategy_id}: {details['status']} "
                  f"(allocation: {details['allocation']:.1%}, "
                  f"PnL: {details['total_pnl']:.2f})")
        
        # Arrêt
        await selector.stop()
        print("\n✅ All strategy selector tests completed!")
    
    # Run test
    import random
    asyncio.run(test_strategy_selector())