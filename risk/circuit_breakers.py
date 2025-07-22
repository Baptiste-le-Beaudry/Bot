"""
Circuit Breakers - Système de Coupe-Circuits Automatiques pour Trading
=====================================================================

Ce module implémente des mécanismes de sécurité critiques qui arrêtent
automatiquement le trading en cas de conditions dangereuses. Conçu pour
prévenir les pertes catastrophiques et protéger le capital.

Types de circuit breakers:
- Drawdown: Arrêt si perte dépasse seuil critique
- Loss Rate: Arrêt si taux de perte trop rapide
- Volatility: Arrêt si volatilité excessive
- Error Rate: Arrêt si trop d'erreurs système
- Market Anomaly: Arrêt si comportement marché anormal
- System Health: Arrêt si problèmes techniques

Architecture:
- État distribué avec synchronisation
- Décisions sub-milliseconde
- Recovery automatique avec cooldown
- Audit trail complet pour compliance
- Integration avec alerting système

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Callable, Awaitable
import json
import numpy as np
from abc import ABC, abstractmethod

# Imports internes
from core.portfolio_manager import Symbol, Price, Quantity
from utils.logger import get_structured_logger
from utils.metrics import MetricsCollector
from monitoring.alerts import AlertManager, AlertLevel


# Types spécifiques
CircuitBreakerId = str
TriggerReason = str


class CircuitBreakerState(Enum):
    """États possibles d'un circuit breaker"""
    CLOSED = "closed"          # Fonctionnement normal
    OPEN = "open"             # Circuit ouvert (trading arrêté)
    HALF_OPEN = "half_open"   # Test de récupération
    FORCED_OPEN = "forced_open"  # Ouvert manuellement


class CircuitBreakerType(Enum):
    """Types de circuit breakers"""
    DRAWDOWN = "drawdown"
    LOSS_RATE = "loss_rate"
    VOLATILITY = "volatility"
    ERROR_RATE = "error_rate"
    MARKET_ANOMALY = "market_anomaly"
    SYSTEM_HEALTH = "system_health"
    CORRELATION = "correlation"
    LIQUIDITY = "liquidity"
    MANUAL = "manual"


class TripAction(Enum):
    """Actions lors du déclenchement"""
    STOP_ALL_TRADING = "stop_all_trading"
    STOP_SYMBOL = "stop_symbol"
    STOP_STRATEGY = "stop_strategy"
    REDUCE_POSITION_SIZE = "reduce_position_size"
    CLOSE_ALL_POSITIONS = "close_all_positions"
    ALERT_ONLY = "alert_only"


@dataclass
class CircuitBreakerConfig:
    """Configuration d'un circuit breaker"""
    breaker_type: CircuitBreakerType
    enabled: bool = True
    
    # Seuils de déclenchement
    threshold: float = 0.0
    time_window: timedelta = timedelta(minutes=5)
    
    # Actions
    trip_action: TripAction = TripAction.STOP_ALL_TRADING
    auto_reset: bool = True
    cooldown_period: timedelta = timedelta(minutes=30)
    
    # Paramètres spécifiques par type
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Notification
    alert_on_trip: bool = True
    alert_level: AlertLevel = AlertLevel.CRITICAL


@dataclass
class CircuitBreakerEvent:
    """Événement de circuit breaker"""
    breaker_id: CircuitBreakerId
    breaker_type: CircuitBreakerType
    event_type: str  # "tripped", "reset", "test"
    timestamp: datetime
    state_before: CircuitBreakerState
    state_after: CircuitBreakerState
    trigger_value: float
    threshold: float
    reason: str
    affected_symbols: List[Symbol] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """État de santé du système"""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_latency: float
    api_response_time: float
    error_count: int
    last_heartbeat: datetime
    is_healthy: bool = True
    
    def get_health_score(self) -> float:
        """Calcule un score de santé global (0-1)"""
        scores = []
        
        # CPU (inverse - moins c'est mieux)
        scores.append(1.0 - min(self.cpu_usage / 100, 1.0))
        
        # Mémoire (inverse)
        scores.append(1.0 - min(self.memory_usage / 100, 1.0))
        
        # Latence réseau (inverse, normalisée sur 1000ms)
        scores.append(1.0 - min(self.network_latency / 1000, 1.0))
        
        # Temps de réponse API (inverse, normalisé sur 500ms)
        scores.append(1.0 - min(self.api_response_time / 500, 1.0))
        
        # Erreurs (inverse, normalisé sur 100)
        scores.append(1.0 - min(self.error_count / 100, 1.0))
        
        return float(np.mean(scores))


class BaseCircuitBreaker(ABC):
    """Classe de base pour tous les circuit breakers"""
    
    def __init__(
        self,
        breaker_id: CircuitBreakerId,
        config: CircuitBreakerConfig,
        logger: Any,
        metrics: MetricsCollector,
        alert_manager: Optional[AlertManager] = None
    ):
        self.breaker_id = breaker_id
        self.config = config
        self.logger = logger
        self.metrics = metrics
        self.alert_manager = alert_manager
        
        # État
        self.state = CircuitBreakerState.CLOSED
        self.last_trip_time: Optional[datetime] = None
        self.trip_count = 0
        self.test_count = 0
        
        # Historique
        self.event_history: deque = deque(maxlen=1000)
        self.state_changes: deque = deque(maxlen=100)
        
    @abstractmethod
    async def evaluate(self, data: Any) -> Tuple[bool, Optional[str]]:
        """
        Évalue si le circuit doit être déclenché
        
        Returns:
            (should_trip, reason)
        """
        pass
    
    async def check_and_trip(self, data: Any) -> bool:
        """
        Vérifie et déclenche le circuit si nécessaire
        
        Returns:
            True si le circuit a été déclenché
        """
        if not self.config.enabled:
            return False
        
        if self.state == CircuitBreakerState.OPEN:
            # Vérifier si on peut passer en half-open
            if self.config.auto_reset and self.last_trip_time:
                time_since_trip = datetime.now(timezone.utc) - self.last_trip_time
                if time_since_trip >= self.config.cooldown_period:
                    await self._transition_to_half_open()
            else:
                return False
        
        # Évaluer la condition
        should_trip, reason = await self.evaluate(data)
        
        if should_trip:
            await self._trip(reason)
            return True
        
        # Si half-open et pas de déclenchement, fermer le circuit
        if self.state == CircuitBreakerState.HALF_OPEN:
            await self._close()
        
        return False
    
    async def _trip(self, reason: str) -> None:
        """Déclenche le circuit breaker"""
        old_state = self.state
        self.state = CircuitBreakerState.OPEN
        self.last_trip_time = datetime.now(timezone.utc)
        self.trip_count += 1
        
        # Créer l'événement
        event = CircuitBreakerEvent(
            breaker_id=self.breaker_id,
            breaker_type=self.config.breaker_type,
            event_type="tripped",
            timestamp=self.last_trip_time,
            state_before=old_state,
            state_after=self.state,
            trigger_value=0.0,  # À override dans les sous-classes
            threshold=self.config.threshold,
            reason=reason
        )
        
        self.event_history.append(event)
        self.state_changes.append((self.last_trip_time, old_state, self.state))
        
        # Logger et alerter
        self.logger.critical(
            "circuit_breaker_tripped",
            breaker_id=self.breaker_id,
            breaker_type=self.config.breaker_type.value,
            reason=reason,
            action=self.config.trip_action.value
        )
        
        # Métriques
        self.metrics.increment(
            "circuit_breaker.trips",
            tags={
                "type": self.config.breaker_type.value,
                "action": self.config.trip_action.value
            }
        )
        
        # Alerter si configuré
        if self.config.alert_on_trip and self.alert_manager:
            await self.alert_manager.send_alert(
                level=self.config.alert_level,
                title=f"Circuit Breaker Déclenché: {self.config.breaker_type.value}",
                message=f"Raison: {reason}\nAction: {self.config.trip_action.value}",
                metadata={
                    "breaker_id": self.breaker_id,
                    "trip_count": self.trip_count
                }
            )
    
    async def _transition_to_half_open(self) -> None:
        """Passe en état half-open pour tester la récupération"""
        old_state = self.state
        self.state = CircuitBreakerState.HALF_OPEN
        self.test_count += 1
        
        self.logger.info(
            "circuit_breaker_half_open",
            breaker_id=self.breaker_id,
            test_count=self.test_count
        )
        
        self.state_changes.append(
            (datetime.now(timezone.utc), old_state, self.state)
        )
    
    async def _close(self) -> None:
        """Ferme le circuit (retour à la normale)"""
        old_state = self.state
        self.state = CircuitBreakerState.CLOSED
        
        self.logger.info(
            "circuit_breaker_closed",
            breaker_id=self.breaker_id,
            time_open=str(datetime.now(timezone.utc) - self.last_trip_time)
            if self.last_trip_time else "N/A"
        )
        
        self.state_changes.append(
            (datetime.now(timezone.utc), old_state, self.state)
        )
        
        # Alerter du retour à la normale
        if self.alert_manager:
            await self.alert_manager.send_alert(
                level=AlertLevel.INFO,
                title=f"Circuit Breaker Fermé: {self.config.breaker_type.value}",
                message="Retour au fonctionnement normal"
            )
    
    async def force_open(self, reason: str = "Manual override") -> None:
        """Force l'ouverture du circuit (arrêt manuel)"""
        old_state = self.state
        self.state = CircuitBreakerState.FORCED_OPEN
        self.last_trip_time = datetime.now(timezone.utc)
        
        self.logger.warning(
            "circuit_breaker_forced_open",
            breaker_id=self.breaker_id,
            reason=reason
        )
        
        await self._trip(f"FORCED: {reason}")
    
    async def reset(self) -> None:
        """Réinitialise le circuit breaker"""
        self.state = CircuitBreakerState.CLOSED
        self.last_trip_time = None
        self.test_count = 0
        
        self.logger.info(
            "circuit_breaker_reset",
            breaker_id=self.breaker_id
        )


class DrawdownCircuitBreaker(BaseCircuitBreaker):
    """Circuit breaker basé sur le drawdown"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_balance = Decimal("0")
        self.current_drawdown = 0.0
        
    async def evaluate(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Évalue le drawdown actuel"""
        current_balance = data.get('balance', Decimal("0"))
        
        # Mettre à jour le pic
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # Calculer le drawdown
        if self.peak_balance > 0:
            self.current_drawdown = float(
                (self.peak_balance - current_balance) / self.peak_balance
            )
        
        # Vérifier le seuil
        if self.current_drawdown > self.config.threshold:
            return True, f"Drawdown {self.current_drawdown:.2%} dépasse le seuil {self.config.threshold:.2%}"
        
        return False, None


class LossRateCircuitBreaker(BaseCircuitBreaker):
    """Circuit breaker basé sur le taux de perte"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_history: deque = deque(maxlen=100)
        
    async def evaluate(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Évalue le taux de perte récent"""
        pnl = data.get('pnl', Decimal("0"))
        timestamp = data.get('timestamp', datetime.now(timezone.utc))
        
        # Ajouter à l'historique
        self.loss_history.append((timestamp, pnl))
        
        # Calculer les pertes dans la fenêtre de temps
        cutoff_time = timestamp - self.config.time_window
        recent_losses = [
            float(pnl) for ts, pnl in self.loss_history
            if ts >= cutoff_time and pnl < 0
        ]
        
        if not recent_losses:
            return False, None
        
        # Calculer le taux de perte
        total_loss = abs(sum(recent_losses))
        loss_rate = total_loss / self.config.time_window.total_seconds() * 3600  # Par heure
        
        # Vérifier le seuil
        if loss_rate > self.config.threshold:
            return True, f"Taux de perte {loss_rate:.2f}/h dépasse le seuil {self.config.threshold:.2f}/h"
        
        return False, None


class VolatilityCircuitBreaker(BaseCircuitBreaker):
    """Circuit breaker basé sur la volatilité excessive"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.price_history: Dict[Symbol, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        
    async def evaluate(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Évalue la volatilité du marché"""
        symbol = data.get('symbol')
        price = data.get('price')
        timestamp = data.get('timestamp', datetime.now(timezone.utc))
        
        if not symbol or not price:
            return False, None
        
        # Ajouter à l'historique
        self.price_history[symbol].append((timestamp, float(price)))
        
        # Calculer la volatilité sur la fenêtre
        cutoff_time = timestamp - self.config.time_window
        recent_prices = [
            p for ts, p in self.price_history[symbol]
            if ts >= cutoff_time
        ]
        
        if len(recent_prices) < 10:  # Pas assez de données
            return False, None
        
        # Calculer la volatilité annualisée
        returns = np.diff(np.log(recent_prices))
        volatility = np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualisée (minutes)
        
        # Vérifier le seuil
        if volatility > self.config.threshold:
            return True, f"Volatilité {volatility:.2%} dépasse le seuil {self.config.threshold:.2%}"
        
        return False, None


class ErrorRateCircuitBreaker(BaseCircuitBreaker):
    """Circuit breaker basé sur le taux d'erreur système"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.error_history: deque = deque(maxlen=1000)
        self.request_count = 0
        
    async def evaluate(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Évalue le taux d'erreur"""
        is_error = data.get('is_error', False)
        error_type = data.get('error_type', 'unknown')
        timestamp = data.get('timestamp', datetime.now(timezone.utc))
        
        # Incrémenter les compteurs
        self.request_count += 1
        if is_error:
            self.error_history.append((timestamp, error_type))
        
        # Calculer le taux d'erreur sur la fenêtre
        cutoff_time = timestamp - self.config.time_window
        recent_errors = [
            err for ts, err in self.error_history
            if ts >= cutoff_time
        ]
        
        # Calculer le taux
        window_seconds = self.config.time_window.total_seconds()
        error_rate = len(recent_errors) / max(window_seconds, 1) * 60  # Par minute
        
        # Vérifier le seuil
        if error_rate > self.config.threshold:
            error_types = defaultdict(int)
            for _, err_type in recent_errors:
                error_types[err_type] += 1
            
            return True, f"Taux d'erreur {error_rate:.1f}/min dépasse le seuil. Types: {dict(error_types)}"
        
        return False, None


class MarketAnomalyCircuitBreaker(BaseCircuitBreaker):
    """Circuit breaker pour détecter les anomalies de marché"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.anomaly_scores: Dict[Symbol, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        
    async def evaluate(self, data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Détecte les anomalies de marché"""
        symbol = data.get('symbol')
        
        # Vérifier les conditions d'anomalie
        anomalies = []
        
        # 1. Flash crash detection
        price_drop = data.get('price_drop_pct', 0)
        if abs(price_drop) > self.config.params.get('flash_crash_threshold', 0.05):
            anomalies.append(f"Flash crash détecté: {price_drop:.2%}")
        
        # 2. Spread anormal
        spread_pct = data.get('spread_pct', 0)
        normal_spread = self.config.params.get('normal_spread_pct', 0.001)
        if spread_pct > normal_spread * 10:
            anomalies.append(f"Spread anormal: {spread_pct:.3%}")
        
        # 3. Volume spike
        volume_ratio = data.get('volume_ratio', 1.0)  # Current/Average
        if volume_ratio > self.config.params.get('volume_spike_threshold', 5.0):
            anomalies.append(f"Spike de volume: {volume_ratio:.1f}x")
        
        # 4. Corrélation breakdown
        correlation_break = data.get('correlation_break', False)
        if correlation_break:
            anomalies.append("Rupture de corrélation détectée")
        
        # Score d'anomalie global
        anomaly_score = len(anomalies) / 4.0  # Normaliser
        
        if symbol:
            self.anomaly_scores[symbol].append(
                (datetime.now(timezone.utc), anomaly_score)
            )
        
        # Déclencher si score dépasse le seuil
        if anomaly_score > self.config.threshold:
            return True, f"Anomalies de marché détectées: {', '.join(anomalies)}"
        
        return False, None


class SystemHealthCircuitBreaker(BaseCircuitBreaker):
    """Circuit breaker basé sur la santé du système"""
    
    async def evaluate(self, data: SystemHealth) -> Tuple[bool, Optional[str]]:
        """Évalue la santé du système"""
        if not isinstance(data, SystemHealth):
            return False, None
        
        health_score = data.get_health_score()
        issues = []
        
        # Vérifier les composants individuels
        if data.cpu_usage > 90:
            issues.append(f"CPU critique: {data.cpu_usage:.1f}%")
        
        if data.memory_usage > 85:
            issues.append(f"Mémoire critique: {data.memory_usage:.1f}%")
        
        if data.network_latency > 500:
            issues.append(f"Latence réseau élevée: {data.network_latency:.0f}ms")
        
        if data.api_response_time > 1000:
            issues.append(f"API lent: {data.api_response_time:.0f}ms")
        
        # Vérifier le heartbeat
        time_since_heartbeat = datetime.now(timezone.utc) - data.last_heartbeat
        if time_since_heartbeat > timedelta(minutes=5):
            issues.append(f"Pas de heartbeat depuis {time_since_heartbeat}")
        
        # Déclencher si score trop bas
        if health_score < self.config.threshold:
            return True, f"Santé système dégradée ({health_score:.2f}). Issues: {', '.join(issues)}"
        
        return False, None


class CircuitBreakerManager:
    """
    Gestionnaire central de tous les circuit breakers
    Coordonne les décisions et les actions
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_manager: AlertManager,
        config: Optional[Dict[str, Any]] = None
    ):
        self.metrics = metrics_collector
        self.alert_manager = alert_manager
        self.config = config or {}
        
        # Logger
        self.logger = get_structured_logger(
            "circuit_breaker_manager",
            module="risk"
        )
        
        # Circuit breakers
        self.breakers: Dict[CircuitBreakerId, BaseCircuitBreaker] = {}
        self.breaker_groups: Dict[str, Set[CircuitBreakerId]] = defaultdict(set)
        
        # État global
        self.global_trading_enabled = True
        self.symbol_trading_enabled: Dict[Symbol, bool] = defaultdict(lambda: True)
        self.strategy_trading_enabled: Dict[str, bool] = defaultdict(lambda: True)
        
        # Historique et statistiques
        self.trip_history: deque = deque(maxlen=1000)
        self.action_history: deque = deque(maxlen=1000)
        
        # Callbacks
        self.action_callbacks: Dict[TripAction, List[Callable]] = defaultdict(list)
        
        # Configuration par défaut
        self._setup_default_breakers()
        
        self.logger.info("circuit_breaker_manager_initialized")
    
    def _setup_default_breakers(self) -> None:
        """Configure les circuit breakers par défaut"""
        # Drawdown breaker
        self.add_circuit_breaker(
            "drawdown_main",
            DrawdownCircuitBreaker,
            CircuitBreakerConfig(
                breaker_type=CircuitBreakerType.DRAWDOWN,
                threshold=self.config.get('max_drawdown', 0.20),  # 20%
                trip_action=TripAction.STOP_ALL_TRADING,
                cooldown_period=timedelta(hours=1)
            )
        )
        
        # Loss rate breaker
        self.add_circuit_breaker(
            "loss_rate_main",
            LossRateCircuitBreaker,
            CircuitBreakerConfig(
                breaker_type=CircuitBreakerType.LOSS_RATE,
                threshold=self.config.get('max_loss_per_hour', 10000),
                time_window=timedelta(hours=1),
                trip_action=TripAction.STOP_ALL_TRADING
            )
        )
        
        # Volatility breaker
        self.add_circuit_breaker(
            "volatility_main",
            VolatilityCircuitBreaker,
            CircuitBreakerConfig(
                breaker_type=CircuitBreakerType.VOLATILITY,
                threshold=self.config.get('max_volatility', 1.0),  # 100% annualisée
                time_window=timedelta(minutes=5),
                trip_action=TripAction.REDUCE_POSITION_SIZE
            )
        )
        
        # Error rate breaker
        self.add_circuit_breaker(
            "error_rate_main",
            ErrorRateCircuitBreaker,
            CircuitBreakerConfig(
                breaker_type=CircuitBreakerType.ERROR_RATE,
                threshold=self.config.get('max_errors_per_minute', 10),
                time_window=timedelta(minutes=1),
                trip_action=TripAction.STOP_ALL_TRADING
            )
        )
        
        # System health breaker
        self.add_circuit_breaker(
            "system_health_main",
            SystemHealthCircuitBreaker,
            CircuitBreakerConfig(
                breaker_type=CircuitBreakerType.SYSTEM_HEALTH,
                threshold=0.3,  # Score minimum de santé
                trip_action=TripAction.STOP_ALL_TRADING,
                alert_level=AlertLevel.EMERGENCY
            )
        )
    
    def add_circuit_breaker(
        self,
        breaker_id: CircuitBreakerId,
        breaker_class: type,
        config: CircuitBreakerConfig,
        group: Optional[str] = None
    ) -> None:
        """Ajoute un nouveau circuit breaker"""
        breaker = breaker_class(
            breaker_id=breaker_id,
            config=config,
            logger=self.logger,
            metrics=self.metrics,
            alert_manager=self.alert_manager
        )
        
        self.breakers[breaker_id] = breaker
        
        if group:
            self.breaker_groups[group].add(breaker_id)
        
        self.logger.info(
            "circuit_breaker_added",
            breaker_id=breaker_id,
            breaker_type=config.breaker_type.value,
            group=group
        )
    
    async def check_all_breakers(self, data: Dict[str, Any]) -> List[CircuitBreakerEvent]:
        """
        Vérifie tous les circuit breakers actifs
        
        Returns:
            Liste des événements de déclenchement
        """
        events = []
        
        for breaker_id, breaker in self.breakers.items():
            if not breaker.config.enabled:
                continue
            
            # Préparer les données selon le type
            breaker_data = self._prepare_breaker_data(breaker.config.breaker_type, data)
            
            # Vérifier et déclencher si nécessaire
            tripped = await breaker.check_and_trip(breaker_data)
            
            if tripped:
                # Exécuter l'action
                await self._execute_trip_action(breaker)
                
                # Créer l'événement
                event = CircuitBreakerEvent(
                    breaker_id=breaker_id,
                    breaker_type=breaker.config.breaker_type,
                    event_type="tripped",
                    timestamp=datetime.now(timezone.utc),
                    state_before=CircuitBreakerState.CLOSED,
                    state_after=CircuitBreakerState.OPEN,
                    trigger_value=0.0,  # TODO: obtenir la vraie valeur
                    threshold=breaker.config.threshold,
                    reason=f"{breaker.config.breaker_type.value} threshold exceeded"
                )
                
                events.append(event)
                self.trip_history.append(event)
        
        # Métriques
        if events:
            self.metrics.gauge(
                "circuit_breakers.active",
                len([b for b in self.breakers.values() if b.state == CircuitBreakerState.OPEN])
            )
        
        return events
    
    def _prepare_breaker_data(
        self,
        breaker_type: CircuitBreakerType,
        raw_data: Dict[str, Any]
    ) -> Any:
        """Prépare les données pour un type de breaker spécifique"""
        if breaker_type == CircuitBreakerType.SYSTEM_HEALTH:
            # Convertir en SystemHealth
            return SystemHealth(
                cpu_usage=raw_data.get('cpu_usage', 0),
                memory_usage=raw_data.get('memory_usage', 0),
                disk_usage=raw_data.get('disk_usage', 0),
                network_latency=raw_data.get('network_latency', 0),
                api_response_time=raw_data.get('api_response_time', 0),
                error_count=raw_data.get('error_count', 0),
                last_heartbeat=raw_data.get('last_heartbeat', datetime.now(timezone.utc))
            )
        
        # Pour les autres types, passer les données brutes
        return raw_data
    
    async def _execute_trip_action(self, breaker: BaseCircuitBreaker) -> None:
        """Exécute l'action configurée lors du déclenchement"""
        action = breaker.config.trip_action
        
        self.logger.warning(
            "executing_circuit_breaker_action",
            action=action.value,
            breaker_id=breaker.breaker_id
        )
        
        # Enregistrer l'action
        self.action_history.append({
            'timestamp': datetime.now(timezone.utc),
            'breaker_id': breaker.breaker_id,
            'action': action,
            'state_before': {
                'global_trading': self.global_trading_enabled,
                'active_symbols': sum(1 for enabled in self.symbol_trading_enabled.values() if enabled)
            }
        })
        
        # Exécuter selon le type d'action
        if action == TripAction.STOP_ALL_TRADING:
            await self.stop_all_trading(f"Circuit breaker: {breaker.breaker_id}")
            
        elif action == TripAction.STOP_SYMBOL:
            # Obtenir les symboles affectés depuis les données
            symbols = breaker.event_history[-1].affected_symbols if breaker.event_history else []
            for symbol in symbols:
                await self.stop_symbol_trading(symbol, f"Circuit breaker: {breaker.breaker_id}")
        
        elif action == TripAction.STOP_STRATEGY:
            # TODO: Implémenter l'arrêt par stratégie
            pass
        
        elif action == TripAction.REDUCE_POSITION_SIZE:
            # Notifier le position sizer de réduire les tailles
            await self._notify_position_size_reduction(0.5)  # Réduire de 50%
        
        elif action == TripAction.CLOSE_ALL_POSITIONS:
            # Notifier le portfolio manager de fermer toutes les positions
            await self._notify_close_all_positions()
        
        elif action == TripAction.ALERT_ONLY:
            # Juste alerter, pas d'action
            pass
        
        # Exécuter les callbacks personnalisés
        for callback in self.action_callbacks.get(action, []):
            try:
                await callback(breaker)
            except Exception as e:
                self.logger.error(
                    "circuit_breaker_callback_error",
                    callback=callback.__name__,
                    error=str(e)
                )
    
    async def stop_all_trading(self, reason: str) -> None:
        """Arrête tout le trading"""
        self.global_trading_enabled = False
        
        # Désactiver tous les symboles
        for symbol in self.symbol_trading_enabled:
            self.symbol_trading_enabled[symbol] = False
        
        # Désactiver toutes les stratégies
        for strategy in self.strategy_trading_enabled:
            self.strategy_trading_enabled[strategy] = False
        
        self.logger.critical(
            "all_trading_stopped",
            reason=reason
        )
        
        # Alerte d'urgence
        await self.alert_manager.send_alert(
            level=AlertLevel.EMERGENCY,
            title="TRADING ARRÊTÉ - Circuit Breaker Déclenché",
            message=f"Tout le trading a été arrêté. Raison: {reason}",
            metadata={
                'requires_manual_intervention': True,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        )
        
        self.metrics.increment("circuit_breaker.global_stops")
    
    async def stop_symbol_trading(self, symbol: Symbol, reason: str) -> None:
        """Arrête le trading pour un symbole spécifique"""
        self.symbol_trading_enabled[symbol] = False
        
        self.logger.warning(
            "symbol_trading_stopped",
            symbol=symbol,
            reason=reason
        )
        
        self.metrics.increment(
            "circuit_breaker.symbol_stops",
            tags={"symbol": symbol}
        )
    
    async def resume_trading(
        self,
        scope: str = "all",  # "all", "symbol", "strategy"
        target: Optional[str] = None
    ) -> bool:
        """
        Reprend le trading après vérifications
        
        Args:
            scope: Portée de la reprise
            target: Symbole ou stratégie spécifique
            
        Returns:
            True si la reprise est autorisée
        """
        # Vérifier que tous les breakers sont fermés ou half-open
        open_breakers = [
            bid for bid, b in self.breakers.items()
            if b.state == CircuitBreakerState.OPEN and b.config.breaker_type != CircuitBreakerType.MANUAL
        ]
        
        if open_breakers:
            self.logger.warning(
                "resume_trading_blocked",
                reason="Circuit breakers still open",
                open_breakers=open_breakers
            )
            return False
        
        # Reprendre selon la portée
        if scope == "all":
            self.global_trading_enabled = True
            for symbol in self.symbol_trading_enabled:
                self.symbol_trading_enabled[symbol] = True
            for strategy in self.strategy_trading_enabled:
                self.strategy_trading_enabled[strategy] = True
                
            self.logger.info("trading_resumed_globally")
            
        elif scope == "symbol" and target:
            self.symbol_trading_enabled[Symbol(target)] = True
            self.logger.info("symbol_trading_resumed", symbol=target)
            
        elif scope == "strategy" and target:
            self.strategy_trading_enabled[target] = True
            self.logger.info("strategy_trading_resumed", strategy=target)
        
        return True
    
    def is_trading_allowed(
        self,
        symbol: Optional[Symbol] = None,
        strategy: Optional[str] = None
    ) -> bool:
        """Vérifie si le trading est autorisé"""
        if not self.global_trading_enabled:
            return False
        
        if symbol and not self.symbol_trading_enabled.get(symbol, True):
            return False
        
        if strategy and not self.strategy_trading_enabled.get(strategy, True):
            return False
        
        return True
    
    def register_action_callback(
        self,
        action: TripAction,
        callback: Callable[[BaseCircuitBreaker], Awaitable[None]]
    ) -> None:
        """Enregistre un callback pour une action spécifique"""
        self.action_callbacks[action].append(callback)
    
    async def _notify_position_size_reduction(self, factor: float) -> None:
        """Notifie la réduction de taille de position"""
        # Cette méthode serait connectée au position sizer
        self.logger.info(
            "position_size_reduction_notified",
            reduction_factor=factor
        )
    
    async def _notify_close_all_positions(self) -> None:
        """Notifie la fermeture de toutes les positions"""
        # Cette méthode serait connectée au portfolio manager
        self.logger.info("close_all_positions_notified")
    
    def get_status(self) -> Dict[str, Any]:
        """Retourne le statut de tous les circuit breakers"""
        return {
            "global_trading_enabled": self.global_trading_enabled,
            "breakers": {
                bid: {
                    "type": b.config.breaker_type.value,
                    "state": b.state.value,
                    "enabled": b.config.enabled,
                    "trip_count": b.trip_count,
                    "last_trip": b.last_trip_time.isoformat() if b.last_trip_time else None
                }
                for bid, b in self.breakers.items()
            },
            "disabled_symbols": [
                s for s, enabled in self.symbol_trading_enabled.items()
                if not enabled
            ],
            "disabled_strategies": [
                s for s, enabled in self.strategy_trading_enabled.items()
                if not enabled
            ],
            "recent_trips": len([
                e for e in self.trip_history
                if datetime.now(timezone.utc) - e.timestamp < timedelta(hours=24)
            ])
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du système de circuit breakers"""
        health = {
            "healthy": True,
            "checks": {}
        }
        
        # Vérifier chaque breaker
        for bid, breaker in self.breakers.items():
            if breaker.state == CircuitBreakerState.OPEN:
                time_open = datetime.now(timezone.utc) - breaker.last_trip_time
                if time_open > timedelta(hours=2):
                    health["healthy"] = False
                    health["checks"][bid] = f"Open for {time_open}"
        
        # Vérifier les trips fréquents
        recent_trips = [
            e for e in self.trip_history
            if datetime.now(timezone.utc) - e.timestamp < timedelta(hours=1)
        ]
        
        if len(recent_trips) > 5:
            health["healthy"] = False
            health["checks"]["trip_frequency"] = f"{len(recent_trips)} trips in last hour"
        
        return health


# Fonction helper pour créer un manager préconfiguré
def create_circuit_breaker_manager(
    config: Dict[str, Any],
    metrics_collector: MetricsCollector,
    alert_manager: AlertManager
) -> CircuitBreakerManager:
    """
    Crée un gestionnaire de circuit breakers préconfiguré
    
    Args:
        config: Configuration du système
        metrics_collector: Collecteur de métriques
        alert_manager: Gestionnaire d'alertes
        
    Returns:
        Instance configurée du manager
    """
    manager = CircuitBreakerManager(
        metrics_collector=metrics_collector,
        alert_manager=alert_manager,
        config=config
    )
    
    # Ajouter des breakers supplémentaires selon la config
    if config.get('enable_correlation_breaker', False):
        manager.add_circuit_breaker(
            "correlation_monitor",
            MarketAnomalyCircuitBreaker,  # Réutiliser pour l'instant
            CircuitBreakerConfig(
                breaker_type=CircuitBreakerType.CORRELATION,
                threshold=0.5,
                params={
                    'correlation_threshold': 0.8,
                    'min_correlation': 0.3
                }
            )
        )
    
    return manager