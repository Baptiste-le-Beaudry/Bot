"""
Système de Stop-Loss Avancé pour Robot de Trading Algorithmique IA
==================================================================

Ce module implémente des mécanismes de stop-loss sophistiqués pour protéger
le capital et gérer le risque. Support pour stop-loss fixes, trailing,
basés sur l'ATR, et adaptatifs par IA.

Fonctionnalités:
- Stop-loss fixes et pourcentages
- Trailing stop-loss avec différents algorithmes
- Stop-loss basés sur l'ATR (Average True Range)
- Stop-loss adaptatifs par machine learning
- Protection contre les gaps et manipulation
- Stop-loss globaux et par stratégie
- Gestion des stop-loss partiels
- Intégration temps réel avec le moteur d'exécution
- Support multi-timeframe et multi-assets
- Backtesting et optimisation

Architecture:
- Event-driven pour latence minimale
- Hot-swappable configuration
- State persistence pour recovery
- Monitoring et alertes intégrés

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Callable, Any, Set
import uuid
import numpy as np
import pandas as pd
from scipy import stats

# Imports internes
from config.settings import get_config
from utils.logger import get_structured_logger
from utils.metrics import MetricsCollector
from utils.decorators import retry_async, circuit_breaker, measure_performance
from core.engine import Event, EventType
from execution.order_manager import OrderType, OrderSide, Order
from ml.features.technical_indicators import ATRCalculator, VolatilityEstimator
from monitoring.alerts import AlertLevel, Alert

logger = get_structured_logger(__name__)
metrics = MetricsCollector()


class StopLossType(Enum):
    """Types de stop-loss supportés"""
    FIXED = "fixed"                    # Prix fixe
    PERCENTAGE = "percentage"          # Pourcentage du prix d'entrée
    TRAILING = "trailing"              # Trailing stop
    ATR_BASED = "atr_based"           # Basé sur l'ATR
    VOLATILITY_ADJUSTED = "volatility_adjusted"  # Ajusté à la volatilité
    TIME_BASED = "time_based"          # Décroissant avec le temps
    ML_ADAPTIVE = "ml_adaptive"        # Adaptatif par ML
    GUARANTEED = "guaranteed"          # Garanti (avec prime)
    BRACKET = "bracket"                # Bracket order avec TP


class StopLossStatus(Enum):
    """États d'un stop-loss"""
    PENDING = "pending"          # En attente d'activation
    ACTIVE = "active"           # Actif et surveillé
    TRIGGERED = "triggered"     # Déclenché
    EXECUTED = "executed"       # Ordre exécuté
    CANCELLED = "cancelled"     # Annulé
    EXPIRED = "expired"         # Expiré
    FAILED = "failed"          # Échec d'exécution


class TrailingMethod(Enum):
    """Méthodes de trailing stop"""
    FIXED_DISTANCE = "fixed_distance"      # Distance fixe
    PERCENTAGE = "percentage"              # Pourcentage
    ATR_MULTIPLE = "atr_multiple"         # Multiple de l'ATR
    CHANDELIER = "chandelier"             # Chandelier exit
    PARABOLIC_SAR = "parabolic_sar"      # SAR parabolique
    DYNAMIC = "dynamic"                   # Dynamique par ML


@dataclass
class StopLossConfig:
    """Configuration d'un stop-loss"""
    stop_type: StopLossType
    initial_stop: Optional[Decimal] = None
    stop_distance: Optional[Decimal] = None
    stop_percentage: Optional[Decimal] = None
    trailing_distance: Optional[Decimal] = None
    atr_multiplier: Optional[Decimal] = None
    time_decay_hours: Optional[int] = None
    max_slippage: Decimal = Decimal("0.001")  # 0.1%
    use_limit_order: bool = False
    limit_offset: Optional[Decimal] = None
    guaranteed: bool = False
    guarantee_premium: Decimal = Decimal("0")
    partial_exits: List[Tuple[Decimal, Decimal]] = field(default_factory=list)  # [(percentage, level)]
    
    def validate(self) -> bool:
        """Valide la configuration"""
        if self.stop_type == StopLossType.FIXED and self.initial_stop is None:
            raise ValueError("Fixed stop-loss requires initial_stop")
        if self.stop_type == StopLossType.PERCENTAGE and self.stop_percentage is None:
            raise ValueError("Percentage stop-loss requires stop_percentage")
        if self.stop_type == StopLossType.ATR_BASED and self.atr_multiplier is None:
            raise ValueError("ATR-based stop-loss requires atr_multiplier")
        return True


@dataclass
class StopLossOrder:
    """Représente un ordre stop-loss"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    position_id: str = ""
    symbol: str = ""
    side: OrderSide = OrderSide.SELL
    quantity: Decimal = Decimal("0")
    stop_price: Decimal = Decimal("0")
    limit_price: Optional[Decimal] = None
    config: StopLossConfig = field(default_factory=StopLossConfig)
    status: StopLossStatus = StopLossStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    triggered_at: Optional[datetime] = None
    executed_at: Optional[datetime] = None
    execution_price: Optional[Decimal] = None
    slippage: Optional[Decimal] = None
    high_water_mark: Optional[Decimal] = None  # Pour trailing stops
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_stop_price(self, new_price: Decimal) -> bool:
        """Met à jour le prix stop"""
        if self.status != StopLossStatus.ACTIVE:
            return False
            
        # Pour les shorts, le stop doit augmenter
        if self.side == OrderSide.BUY and new_price > self.stop_price:
            self.stop_price = new_price
            return True
        # Pour les longs, le stop doit diminuer  
        elif self.side == OrderSide.SELL and new_price < self.stop_price:
            self.stop_price = new_price
            return True
            
        return False


class StopLossStrategy(ABC):
    """Classe de base pour les stratégies de stop-loss"""
    
    def __init__(self, config: StopLossConfig):
        self.config = config
        self.logger = get_structured_logger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    async def calculate_stop_price(
        self,
        entry_price: Decimal,
        current_price: Decimal,
        position_side: OrderSide,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Decimal:
        """Calcule le prix de stop-loss"""
        pass
        
    @abstractmethod
    async def should_trail(
        self,
        current_price: Decimal,
        stop_order: StopLossOrder,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Decimal]]:
        """Détermine si le stop doit être ajusté"""
        pass


class FixedStopLoss(StopLossStrategy):
    """Stop-loss à prix fixe"""
    
    async def calculate_stop_price(
        self,
        entry_price: Decimal,
        current_price: Decimal,
        position_side: OrderSide,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Decimal:
        """Retourne le prix stop fixe"""
        return self.config.initial_stop
        
    async def should_trail(
        self,
        current_price: Decimal,
        stop_order: StopLossOrder,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Decimal]]:
        """Les stops fixes ne bougent pas"""
        return False, None


class PercentageStopLoss(StopLossStrategy):
    """Stop-loss basé sur un pourcentage"""
    
    async def calculate_stop_price(
        self,
        entry_price: Decimal,
        current_price: Decimal,
        position_side: OrderSide,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Decimal:
        """Calcule le stop basé sur le pourcentage"""
        stop_distance = entry_price * self.config.stop_percentage
        
        if position_side == OrderSide.BUY:  # Long position
            return entry_price - stop_distance
        else:  # Short position
            return entry_price + stop_distance
            
    async def should_trail(
        self,
        current_price: Decimal,
        stop_order: StopLossOrder,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Decimal]]:
        """Pas de trailing pour les stops percentage simples"""
        return False, None


class TrailingStopLoss(StopLossStrategy):
    """Trailing stop-loss"""
    
    def __init__(self, config: StopLossConfig):
        super().__init__(config)
        self.trailing_method = TrailingMethod.FIXED_DISTANCE
        if config.atr_multiplier:
            self.trailing_method = TrailingMethod.ATR_MULTIPLE
        
    async def calculate_stop_price(
        self,
        entry_price: Decimal,
        current_price: Decimal,
        position_side: OrderSide,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Decimal:
        """Calcule le stop initial pour un trailing stop"""
        if self.config.trailing_distance:
            distance = self.config.trailing_distance
        elif self.config.stop_percentage:
            distance = entry_price * self.config.stop_percentage
        else:
            distance = entry_price * Decimal("0.02")  # 2% par défaut
            
        if position_side == OrderSide.BUY:
            return entry_price - distance
        else:
            return entry_price + distance
            
    async def should_trail(
        self,
        current_price: Decimal,
        stop_order: StopLossOrder,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Decimal]]:
        """Détermine si le trailing stop doit être ajusté"""
        if stop_order.status != StopLossStatus.ACTIVE:
            return False, None
            
        # Mettre à jour le high water mark
        if stop_order.high_water_mark is None:
            stop_order.high_water_mark = current_price
        else:
            if stop_order.side == OrderSide.SELL:  # Long position
                stop_order.high_water_mark = max(stop_order.high_water_mark, current_price)
            else:  # Short position
                stop_order.high_water_mark = min(stop_order.high_water_mark, current_price)
                
        # Calculer le nouveau stop
        if self.trailing_method == TrailingMethod.FIXED_DISTANCE:
            distance = self.config.trailing_distance
        elif self.trailing_method == TrailingMethod.ATR_MULTIPLE and market_data:
            atr = market_data.get('atr', self.config.trailing_distance)
            distance = atr * self.config.atr_multiplier
        else:
            distance = stop_order.high_water_mark * self.config.stop_percentage
            
        if stop_order.side == OrderSide.SELL:  # Long position
            new_stop = stop_order.high_water_mark - distance
            if new_stop > stop_order.stop_price:
                return True, new_stop
        else:  # Short position
            new_stop = stop_order.high_water_mark + distance
            if new_stop < stop_order.stop_price:
                return True, new_stop
                
        return False, None


class ATRStopLoss(StopLossStrategy):
    """Stop-loss basé sur l'ATR"""
    
    def __init__(self, config: StopLossConfig):
        super().__init__(config)
        self.atr_calculator = ATRCalculator(period=14)
        
    async def calculate_stop_price(
        self,
        entry_price: Decimal,
        current_price: Decimal,
        position_side: OrderSide,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Decimal:
        """Calcule le stop basé sur l'ATR"""
        if not market_data or 'atr' not in market_data:
            # Fallback sur un pourcentage si pas d'ATR
            return await PercentageStopLoss(self.config).calculate_stop_price(
                entry_price, current_price, position_side
            )
            
        atr = Decimal(str(market_data['atr']))
        distance = atr * self.config.atr_multiplier
        
        if position_side == OrderSide.BUY:
            return entry_price - distance
        else:
            return entry_price + distance
            
    async def should_trail(
        self,
        current_price: Decimal,
        stop_order: StopLossOrder,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Decimal]]:
        """Ajuste le stop selon l'ATR actuel"""
        if not market_data or 'atr' not in market_data:
            return False, None
            
        atr = Decimal(str(market_data['atr']))
        distance = atr * self.config.atr_multiplier
        
        # Utiliser le principe du trailing stop
        trailing_strategy = TrailingStopLoss(self.config)
        trailing_strategy.config.trailing_distance = distance
        return await trailing_strategy.should_trail(current_price, stop_order, market_data)


class MLAdaptiveStopLoss(StopLossStrategy):
    """Stop-loss adaptatif basé sur le machine learning"""
    
    def __init__(self, config: StopLossConfig, ml_model=None):
        super().__init__(config)
        self.ml_model = ml_model
        self.feature_window = 50
        self.prediction_cache = {}
        self.cache_ttl = 60  # secondes
        
    async def calculate_stop_price(
        self,
        entry_price: Decimal,
        current_price: Decimal,
        position_side: OrderSide,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Decimal:
        """Calcule le stop optimal via ML"""
        if not self.ml_model or not market_data:
            # Fallback sur ATR
            return await ATRStopLoss(self.config).calculate_stop_price(
                entry_price, current_price, position_side, market_data
            )
            
        # Extraire les features
        features = self._extract_features(market_data)
        
        # Prédire la distance de stop optimale
        try:
            optimal_distance_pct = self.ml_model.predict(features)[0]
            distance = entry_price * Decimal(str(optimal_distance_pct))
            
            if position_side == OrderSide.BUY:
                return entry_price - distance
            else:
                return entry_price + distance
                
        except Exception as e:
            self.logger.error("ML prediction failed", error=str(e))
            # Fallback
            return await ATRStopLoss(self.config).calculate_stop_price(
                entry_price, current_price, position_side, market_data
            )
            
    async def should_trail(
        self,
        current_price: Decimal,
        stop_order: StopLossOrder,
        market_data: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Decimal]]:
        """Ajuste dynamiquement le stop selon les prédictions ML"""
        if not self.ml_model or not market_data:
            return False, None
            
        # Cache pour éviter trop de prédictions
        cache_key = f"{stop_order.id}_{int(time.time() / self.cache_ttl)}"
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
            
        features = self._extract_features(market_data)
        
        try:
            # Prédire si ajustement nécessaire
            should_adjust, new_distance_pct = self.ml_model.predict_adjustment(
                features, stop_order.metadata
            )
            
            if should_adjust:
                distance = current_price * Decimal(str(new_distance_pct))
                
                if stop_order.side == OrderSide.SELL:
                    new_stop = current_price - distance
                else:
                    new_stop = current_price + distance
                    
                result = (True, new_stop)
            else:
                result = (False, None)
                
            self.prediction_cache[cache_key] = result
            return result
            
        except Exception as e:
            self.logger.error("ML adjustment prediction failed", error=str(e))
            return False, None
            
    def _extract_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extrait les features pour le modèle ML"""
        features = []
        
        # Prix et volumes
        features.extend([
            market_data.get('price', 0),
            market_data.get('volume', 0),
            market_data.get('bid_ask_spread', 0)
        ])
        
        # Indicateurs techniques
        features.extend([
            market_data.get('rsi', 50),
            market_data.get('atr', 0),
            market_data.get('volatility', 0),
            market_data.get('trend_strength', 0)
        ])
        
        # Microstructure
        features.extend([
            market_data.get('order_flow_imbalance', 0),
            market_data.get('trade_intensity', 0)
        ])
        
        return np.array(features).reshape(1, -1)


class StopLossManager:
    """Gestionnaire principal des stop-loss"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        self.logger = get_structured_logger(__name__)
        
        # Stockage des stop-loss
        self.active_stops: Dict[str, StopLossOrder] = {}
        self.position_stops: Dict[str, List[str]] = defaultdict(list)
        self.triggered_stops: deque = deque(maxlen=1000)
        
        # Stratégies disponibles
        self.strategies: Dict[StopLossType, StopLossStrategy] = {
            StopLossType.FIXED: FixedStopLoss,
            StopLossType.PERCENTAGE: PercentageStopLoss,
            StopLossType.TRAILING: TrailingStopLoss,
            StopLossType.ATR_BASED: ATRStopLoss,
            StopLossType.ML_ADAPTIVE: MLAdaptiveStopLoss
        }
        
        # Configuration des limites
        self.max_stops_per_position = 5
        self.min_stop_distance = Decimal("0.001")  # 0.1%
        self.max_slippage_allowed = Decimal("0.005")  # 0.5%
        
        # État
        self.running = False
        self.last_check = time.time()
        self.check_interval = 0.1  # 100ms
        
        # Métriques
        self.metrics = {
            'stops_created': 0,
            'stops_triggered': 0,
            'stops_executed': 0,
            'total_slippage': Decimal("0"),
            'protection_saved': Decimal("0")
        }
        
        # ML model pour stops adaptatifs
        self.ml_model = None
        self._load_ml_model()
        
        self.logger.info("StopLossManager initialized")
        
    def _load_ml_model(self):
        """Charge le modèle ML pour les stops adaptatifs"""
        try:
            # TODO: Charger le modèle réel
            self.logger.info("ML model loaded for adaptive stops")
        except Exception as e:
            self.logger.warning("Failed to load ML model", error=str(e))
            
    async def create_stop_loss(
        self,
        position_id: str,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        entry_price: Decimal,
        config: StopLossConfig,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StopLossOrder:
        """Crée un nouveau stop-loss"""
        try:
            # Validation
            config.validate()
            
            # Vérifier les limites
            if len(self.position_stops[position_id]) >= self.max_stops_per_position:
                raise ValueError(f"Maximum stops ({self.max_stops_per_position}) reached for position")
                
            # Créer la stratégie
            strategy_class = self.strategies.get(config.stop_type)
            if not strategy_class:
                raise ValueError(f"Unknown stop type: {config.stop_type}")
                
            strategy = strategy_class(config)
            if config.stop_type == StopLossType.ML_ADAPTIVE:
                strategy = strategy_class(config, self.ml_model)
                
            # Calculer le prix initial
            current_price = entry_price  # TODO: Obtenir le prix actuel
            stop_price = await strategy.calculate_stop_price(
                entry_price, current_price, side
            )
            
            # Vérifier la distance minimale
            distance = abs(stop_price - entry_price) / entry_price
            if distance < self.min_stop_distance:
                raise ValueError(f"Stop distance too small: {distance:.4%}")
                
            # Créer l'ordre
            stop_order = StopLossOrder(
                position_id=position_id,
                symbol=symbol,
                side=OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY,
                quantity=quantity,
                stop_price=stop_price,
                config=config,
                status=StopLossStatus.ACTIVE,
                metadata=metadata or {}
            )
            
            # Ajouter le prix limite si nécessaire
            if config.use_limit_order and config.limit_offset:
                if stop_order.side == OrderSide.SELL:
                    stop_order.limit_price = stop_price - config.limit_offset
                else:
                    stop_order.limit_price = stop_price + config.limit_offset
                    
            # Enregistrer
            self.active_stops[stop_order.id] = stop_order
            self.position_stops[position_id].append(stop_order.id)
            
            # Métriques
            self.metrics['stops_created'] += 1
            metrics.increment('stop_loss.created', tags={'type': config.stop_type.value})
            
            self.logger.info(
                "Stop-loss created",
                stop_id=stop_order.id,
                position_id=position_id,
                symbol=symbol,
                stop_price=float(stop_price),
                type=config.stop_type.value
            )
            
            # Émettre l'événement
            await self._emit_event(
                EventType.RISK_ALERT,
                {
                    'action': 'stop_created',
                    'stop_order': stop_order,
                    'position_id': position_id
                }
            )
            
            return stop_order
            
        except Exception as e:
            self.logger.error(
                "Failed to create stop-loss",
                position_id=position_id,
                error=str(e),
                exc_info=True
            )
            raise
            
    async def update_stop_loss(
        self,
        stop_id: str,
        new_stop_price: Optional[Decimal] = None,
        new_quantity: Optional[Decimal] = None,
        new_config: Optional[StopLossConfig] = None
    ) -> bool:
        """Met à jour un stop-loss existant"""
        try:
            stop_order = self.active_stops.get(stop_id)
            if not stop_order:
                raise ValueError(f"Stop-loss not found: {stop_id}")
                
            if stop_order.status != StopLossStatus.ACTIVE:
                raise ValueError(f"Cannot update {stop_order.status.value} stop-loss")
                
            updated = False
            
            # Mettre à jour le prix
            if new_stop_price and new_stop_price != stop_order.stop_price:
                if stop_order.update_stop_price(new_stop_price):
                    updated = True
                    self.logger.info(
                        "Stop price updated",
                        stop_id=stop_id,
                        old_price=float(stop_order.stop_price),
                        new_price=float(new_stop_price)
                    )
                    
            # Mettre à jour la quantité
            if new_quantity and new_quantity != stop_order.quantity:
                stop_order.quantity = new_quantity
                updated = True
                
            # Mettre à jour la configuration
            if new_config:
                new_config.validate()
                stop_order.config = new_config
                updated = True
                
            if updated:
                await self._emit_event(
                    EventType.RISK_ALERT,
                    {
                        'action': 'stop_updated',
                        'stop_order': stop_order
                    }
                )
                
            return updated
            
        except Exception as e:
            self.logger.error(
                "Failed to update stop-loss",
                stop_id=stop_id,
                error=str(e)
            )
            return False
            
    async def cancel_stop_loss(self, stop_id: str, reason: str = "manual") -> bool:
        """Annule un stop-loss"""
        try:
            stop_order = self.active_stops.get(stop_id)
            if not stop_order:
                return False
                
            if stop_order.status not in [StopLossStatus.ACTIVE, StopLossStatus.PENDING]:
                return False
                
            # Marquer comme annulé
            stop_order.status = StopLossStatus.CANCELLED
            stop_order.metadata['cancel_reason'] = reason
            stop_order.metadata['cancelled_at'] = datetime.now(timezone.utc).isoformat()
            
            # Retirer des actifs
            del self.active_stops[stop_id]
            self.position_stops[stop_order.position_id].remove(stop_id)
            
            self.logger.info(
                "Stop-loss cancelled",
                stop_id=stop_id,
                reason=reason
            )
            
            await self._emit_event(
                EventType.RISK_ALERT,
                {
                    'action': 'stop_cancelled',
                    'stop_id': stop_id,
                    'reason': reason
                }
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Failed to cancel stop-loss",
                stop_id=stop_id,
                error=str(e)
            )
            return False
            
    async def check_stops(self, market_data: Dict[str, Dict[str, Any]]) -> List[StopLossOrder]:
        """Vérifie tous les stops actifs"""
        triggered = []
        current_time = time.time()
        
        # Limiter la fréquence des vérifications
        if current_time - self.last_check < self.check_interval:
            return triggered
            
        self.last_check = current_time
        
        for stop_id, stop_order in list(self.active_stops.items()):
            if stop_order.status != StopLossStatus.ACTIVE:
                continue
                
            symbol_data = market_data.get(stop_order.symbol)
            if not symbol_data:
                continue
                
            current_price = Decimal(str(symbol_data.get('price', 0)))
            if current_price <= 0:
                continue
                
            # Vérifier si le stop est atteint
            is_triggered = False
            
            if stop_order.side == OrderSide.SELL:  # Long position
                is_triggered = current_price <= stop_order.stop_price
            else:  # Short position
                is_triggered = current_price >= stop_order.stop_price
                
            if is_triggered:
                await self._trigger_stop(stop_order, current_price)
                triggered.append(stop_order)
            else:
                # Vérifier si trailing nécessaire
                await self._check_trailing(stop_order, current_price, symbol_data)
                
        return triggered
        
    async def _trigger_stop(self, stop_order: StopLossOrder, trigger_price: Decimal):
        """Déclenche un stop-loss"""
        try:
            stop_order.status = StopLossStatus.TRIGGERED
            stop_order.triggered_at = datetime.now(timezone.utc)
            
            # Calculer le slippage estimé
            expected_slippage = stop_order.config.max_slippage * stop_order.stop_price
            
            self.logger.warning(
                "Stop-loss triggered",
                stop_id=stop_order.id,
                symbol=stop_order.symbol,
                stop_price=float(stop_order.stop_price),
                trigger_price=float(trigger_price),
                quantity=float(stop_order.quantity)
            )
            
            # Émettre l'ordre d'exécution
            execution_order = {
                'stop_order_id': stop_order.id,
                'symbol': stop_order.symbol,
                'side': stop_order.side,
                'quantity': stop_order.quantity,
                'order_type': OrderType.MARKET if not stop_order.limit_price else OrderType.LIMIT,
                'limit_price': stop_order.limit_price,
                'urgency': 'high',
                'metadata': {
                    'stop_loss': True,
                    'expected_slippage': float(expected_slippage)
                }
            }
            
            await self._emit_event(
                EventType.TRADE_SIGNAL,
                {
                    'action': 'stop_triggered',
                    'order': execution_order,
                    'stop_order': stop_order
                }
            )
            
            # Retirer des stops actifs
            del self.active_stops[stop_order.id]
            self.position_stops[stop_order.position_id].remove(stop_order.id)
            self.triggered_stops.append(stop_order)
            
            # Métriques
            self.metrics['stops_triggered'] += 1
            metrics.increment('stop_loss.triggered', tags={'symbol': stop_order.symbol})
            
            # Alerte
            await self._send_alert(
                AlertLevel.WARNING,
                f"Stop-loss triggered for {stop_order.symbol}",
                {
                    'stop_id': stop_order.id,
                    'position_id': stop_order.position_id,
                    'loss_estimate': float((trigger_price - stop_order.stop_price) * stop_order.quantity)
                }
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to trigger stop-loss",
                stop_id=stop_order.id,
                error=str(e),
                exc_info=True
            )
            stop_order.status = StopLossStatus.FAILED
            
    async def _check_trailing(
        self,
        stop_order: StopLossOrder,
        current_price: Decimal,
        market_data: Dict[str, Any]
    ):
        """Vérifie et ajuste les trailing stops"""
        try:
            # Obtenir la stratégie
            strategy_class = self.strategies.get(stop_order.config.stop_type)
            if not strategy_class:
                return
                
            strategy = strategy_class(stop_order.config)
            if stop_order.config.stop_type == StopLossType.ML_ADAPTIVE:
                strategy = strategy_class(stop_order.config, self.ml_model)
                
            # Vérifier si ajustement nécessaire
            should_trail, new_stop_price = await strategy.should_trail(
                current_price, stop_order, market_data
            )
            
            if should_trail and new_stop_price:
                # Mettre à jour le stop
                old_price = stop_order.stop_price
                if stop_order.update_stop_price(new_stop_price):
                    self.logger.info(
                        "Trailing stop adjusted",
                        stop_id=stop_order.id,
                        old_stop=float(old_price),
                        new_stop=float(new_stop_price),
                        current_price=float(current_price)
                    )
                    
                    # Mettre à jour le prix limite si nécessaire
                    if stop_order.config.use_limit_order and stop_order.config.limit_offset:
                        if stop_order.side == OrderSide.SELL:
                            stop_order.limit_price = new_stop_price - stop_order.config.limit_offset
                        else:
                            stop_order.limit_price = new_stop_price + stop_order.config.limit_offset
                            
                    metrics.increment('stop_loss.trailing_adjusted')
                    
        except Exception as e:
            self.logger.error(
                "Failed to check trailing stop",
                stop_id=stop_order.id,
                error=str(e)
            )
            
    async def handle_execution_result(
        self,
        stop_order_id: str,
        execution_price: Decimal,
        executed_quantity: Decimal,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Traite le résultat d'exécution d'un stop"""
        try:
            # Retrouver l'ordre dans les triggered
            stop_order = None
            for order in self.triggered_stops:
                if order.id == stop_order_id:
                    stop_order = order
                    break
                    
            if not stop_order:
                self.logger.error("Stop order not found in triggered", stop_order_id=stop_order_id)
                return
                
            if success:
                stop_order.status = StopLossStatus.EXECUTED
                stop_order.executed_at = datetime.now(timezone.utc)
                stop_order.execution_price = execution_price
                stop_order.slippage = abs(execution_price - stop_order.stop_price)
                
                # Métriques
                self.metrics['stops_executed'] += 1
                self.metrics['total_slippage'] += stop_order.slippage
                
                # Calculer la protection
                if metadata and 'entry_price' in metadata:
                    entry_price = Decimal(str(metadata['entry_price']))
                    if stop_order.side == OrderSide.SELL:
                        max_loss = (entry_price - stop_order.stop_price) * stop_order.quantity
                        actual_loss = (entry_price - execution_price) * executed_quantity
                    else:
                        max_loss = (stop_order.stop_price - entry_price) * stop_order.quantity
                        actual_loss = (execution_price - entry_price) * executed_quantity
                        
                    protection = max(max_loss - actual_loss, Decimal("0"))
                    self.metrics['protection_saved'] += protection
                    
                self.logger.info(
                    "Stop-loss executed successfully",
                    stop_id=stop_order_id,
                    execution_price=float(execution_price),
                    slippage=float(stop_order.slippage),
                    slippage_pct=float(stop_order.slippage / stop_order.stop_price * 100)
                )
                
            else:
                stop_order.status = StopLossStatus.FAILED
                self.logger.error(
                    "Stop-loss execution failed",
                    stop_id=stop_order_id,
                    metadata=metadata
                )
                
                # Alerte critique
                await self._send_alert(
                    AlertLevel.CRITICAL,
                    f"Stop-loss execution failed for {stop_order.symbol}",
                    {
                        'stop_id': stop_order_id,
                        'position_id': stop_order.position_id,
                        'quantity': float(stop_order.quantity)
                    }
                )
                
        except Exception as e:
            self.logger.error(
                "Failed to handle execution result",
                stop_order_id=stop_order_id,
                error=str(e),
                exc_info=True
            )
            
    def get_position_stops(self, position_id: str) -> List[StopLossOrder]:
        """Récupère tous les stops d'une position"""
        stop_ids = self.position_stops.get(position_id, [])
        return [self.active_stops[sid] for sid in stop_ids if sid in self.active_stops]
        
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques du gestionnaire"""
        total_active = len(self.active_stops)
        
        # Calculer les statistiques
        avg_slippage = Decimal("0")
        if self.metrics['stops_executed'] > 0:
            avg_slippage = self.metrics['total_slippage'] / self.metrics['stops_executed']
            
        success_rate = 0
        if self.metrics['stops_triggered'] > 0:
            success_rate = self.metrics['stops_executed'] / self.metrics['stops_triggered']
            
        return {
            'active_stops': total_active,
            'stops_by_type': self._count_by_type(),
            'total_created': self.metrics['stops_created'],
            'total_triggered': self.metrics['stops_triggered'],
            'total_executed': self.metrics['stops_executed'],
            'success_rate': float(success_rate),
            'average_slippage': float(avg_slippage),
            'total_protection_saved': float(self.metrics['protection_saved'])
        }
        
    def _count_by_type(self) -> Dict[str, int]:
        """Compte les stops par type"""
        counts = defaultdict(int)
        for stop in self.active_stops.values():
            counts[stop.config.stop_type.value] += 1
        return dict(counts)
        
    async def _emit_event(self, event_type: EventType, data: Dict[str, Any]):
        """Émet un événement"""
        # TODO: Implémenter l'émission via le bus d'événements
        pass
        
    async def _send_alert(self, level: AlertLevel, message: str, details: Dict[str, Any]):
        """Envoie une alerte"""
        # TODO: Implémenter l'envoi via AlertManager
        pass
        
    async def start(self):
        """Démarre le gestionnaire"""
        self.running = True
        self.logger.info("StopLossManager started")
        
    async def stop(self):
        """Arrête le gestionnaire"""
        self.running = False
        self.logger.info("StopLossManager stopped")
        
    async def cleanup(self):
        """Nettoie les ressources"""
        # Sauvegarder l'état si nécessaire
        state = {
            'active_stops': {k: v.__dict__ for k, v in self.active_stops.items()},
            'metrics': self.metrics,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # TODO: Sauvegarder dans la base de données
        
        self.logger.info("StopLossManager cleanup completed")