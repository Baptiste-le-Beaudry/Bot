"""
Moteur Principal du Robot de Trading Algorithmique IA
====================================================

Ce module implémente le moteur central event-driven qui orchestre tous les composants
du système de trading. Il suit les meilleures pratiques 2025 pour une architecture
haute performance avec support pour l'apprentissage par renforcement profond.

Architecture:
- Event-driven avec asyncio pour performance sub-milliseconde
- Plugin-based pour hot-swapping des stratégies
- Circuit breakers et retry patterns pour résilience
- Event sourcing pour audit trail complet
- Monitoring et observabilité intégrés

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import asyncio
import logging
import signal
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Protocol, Set, Union, 
    Callable, Awaitable, TypeVar, Generic
)
from collections import defaultdict, deque
import json

# Third-party imports pour performance et monitoring
import aiohttp
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
import structlog

# Imports internes (seront créés dans les prochains fichiers)
from config.settings import TradingConfig
from utils.logger import get_structured_logger
from utils.metrics import MetricsCollector
from utils.decorators import retry_async, circuit_breaker
from monitoring.alerts import AlertManager

# Type definitions pour la flexibilité
T = TypeVar('T')
EventHandler = Callable[[Any], Awaitable[None]]


class EngineState(Enum):
    """États possibles du moteur de trading"""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    EMERGENCY_STOP = "emergency_stop"


class EventType(Enum):
    """Types d'événements dans le système"""
    MARKET_DATA = "market_data"
    TRADE_SIGNAL = "trade_signal"
    ORDER_UPDATE = "order_update"
    POSITION_UPDATE = "position_update"
    RISK_ALERT = "risk_alert"
    STRATEGY_CHANGE = "strategy_change"
    SYSTEM_STATUS = "system_status"
    ERROR_EVENT = "error_event"


@dataclass
class TradingEvent:
    """Événement standardisé dans le système"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: EventType = EventType.SYSTEM_STATUS
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "unknown"
    data: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit l'événement en dictionnaire pour logging"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "data": self.data,
            "correlation_id": self.correlation_id
        }


class ComponentProtocol(Protocol):
    """Interface pour tous les composants du système"""
    
    async def start(self) -> None:
        """Démarre le composant"""
        ...
    
    async def stop(self) -> None:
        """Arrête le composant"""
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """Vérifie la santé du composant"""
        ...


class EventBus:
    """Event bus haute performance pour communication inter-composants"""
    
    def __init__(self, max_queue_size: int = 10000):
        self.subscribers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue_size)
        self.metrics = MetricsCollector("event_bus")
        self.logger = get_structured_logger("event_bus")
        self._running = False
        
    async def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Abonne un handler à un type d'événement"""
        self.subscribers[event_type].append(handler)
        self.logger.info(
            "handler_subscribed",
            event_type=event_type.value,
            handler=handler.__name__
        )
    
    async def publish(self, event: TradingEvent) -> None:
        """Publie un événement de façon asynchrone"""
        try:
            await self.event_queue.put(event)
            self.metrics.increment("events_published", tags={"type": event.event_type.value})
        except asyncio.QueueFull:
            self.logger.error("event_queue_full", event_id=event.event_id)
            self.metrics.increment("events_dropped")
    
    async def start_processing(self) -> None:
        """Démarre le processus de traitement des événements"""
        self._running = True
        while self._running:
            try:
                # Timeout pour permettre l'arrêt propre
                event = await asyncio.wait_for(self.event_queue.get(), timeout=1.0)
                await self._process_event(event)
                self.event_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error("event_processing_error", error=str(e))
    
    async def _process_event(self, event: TradingEvent) -> None:
        """Traite un événement en appelant tous les handlers abonnés"""
        handlers = self.subscribers.get(event.event_type, [])
        if not handlers:
            return
            
        start_time = time.perf_counter()
        tasks = [handler(event) for handler in handlers]
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            processing_time = time.perf_counter() - start_time
            self.metrics.histogram("event_processing_time", processing_time)
        except Exception as e:
            self.logger.error("event_handler_error", event_id=event.event_id, error=str(e))
    
    async def stop(self) -> None:
        """Arrête le traitement des événements"""
        self._running = False


class TradingEngine:
    """
    Moteur principal du système de trading algorithmique IA
    
    Responsabilités:
    - Orchestration de tous les composants
    - Gestion du cycle de vie et des états
    - Event bus central et communication
    - Monitoring et observabilité
    - Gestion des erreurs et récupération
    - Hot-swapping des stratégies
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.state = EngineState.INITIALIZING
        self.session_id = str(uuid.uuid4())
        
        # Logging et monitoring
        self.logger = get_structured_logger("trading_engine", session_id=self.session_id)
        self.metrics = MetricsCollector("trading_engine")
        self.alert_manager = AlertManager()
        
        # Event system
        self.event_bus = EventBus(max_queue_size=config.event_queue_size)
        self.event_store: deque = deque(maxlen=config.event_store_size)
        
        # Composants du système (seront injectés)
        self.components: Dict[str, ComponentProtocol] = {}
        self.data_collectors: Dict[str, Any] = {}
        self.strategies: Dict[str, Any] = {}
        self.risk_manager: Optional[Any] = None
        self.portfolio_manager: Optional[Any] = None
        self.execution_engine: Optional[Any] = None
        
        # Contrôle d'exécution
        self._shutdown_event = asyncio.Event()
        self._tasks: Set[asyncio.Task] = set()
        self._health_check_interval = config.health_check_interval
        
        # Circuit breakers pour résilience
        self._circuit_breakers: Dict[str, bool] = defaultdict(bool)
        
        self.logger.info("trading_engine_initialized", config=config.model_dump())
    
    @asynccontextmanager
    async def managed_task(self, coro):
        """Context manager pour gérer les tâches asyncio"""
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        try:
            yield task
        finally:
            self._tasks.discard(task)
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    async def register_component(self, name: str, component: ComponentProtocol) -> None:
        """Enregistre un composant dans le système"""
        self.components[name] = component
        
        # Subscribe aux événements de santé si le composant le supporte
        if hasattr(component, 'on_event'):
            await self.event_bus.subscribe(EventType.SYSTEM_STATUS, component.on_event)
        
        self.logger.info("component_registered", component_name=name)
    
    async def register_strategy(self, strategy_id: str, strategy: Any) -> None:
        """Enregistre une stratégie de trading"""
        self.strategies[strategy_id] = strategy
        
        # Subscribe aux événements de marché
        if hasattr(strategy, 'on_market_data'):
            await self.event_bus.subscribe(EventType.MARKET_DATA, strategy.on_market_data)
        
        await self._publish_event(
            EventType.STRATEGY_CHANGE,
            {"action": "registered", "strategy_id": strategy_id},
            source="engine"
        )
        
        self.logger.info("strategy_registered", strategy_id=strategy_id)
    
    @retry_async(max_attempts=3, backoff_factor=2.0)
    async def start(self) -> None:
        """Démarre le moteur de trading"""
        if self.state != EngineState.INITIALIZING:
            raise RuntimeError(f"Cannot start engine in state: {self.state}")
        
        self.state = EngineState.STARTING
        self.logger.info("trading_engine_starting")
        
        try:
            # 1. Démarrer l'event bus
            async with self.managed_task(self.event_bus.start_processing()):
                
                # 2. Démarrer tous les composants
                for name, component in self.components.items():
                    try:
                        await component.start()
                        self.logger.info("component_started", component_name=name)
                    except Exception as e:
                        self.logger.error("component_start_failed", 
                                        component_name=name, error=str(e))
                        raise
                
                # 3. Démarrer les services de monitoring
                async with self.managed_task(self._health_monitor()):
                    async with self.managed_task(self._metrics_collector()):
                        
                        # 4. Configurer les gestionnaires de signaux
                        self._setup_signal_handlers()
                        
                        # 5. État de fonctionnement
                        self.state = EngineState.RUNNING
                        await self._publish_event(
                            EventType.SYSTEM_STATUS,
                            {"state": self.state.value, "message": "Engine started successfully"},
                            source="engine"
                        )
                        
                        self.logger.info("trading_engine_started", session_id=self.session_id)
                        
                        # 6. Boucle principale - attendre l'arrêt
                        await self._shutdown_event.wait()
        
        except Exception as e:
            self.state = EngineState.ERROR
            self.logger.error("trading_engine_start_failed", error=str(e))
            await self.alert_manager.send_critical_alert(
                "Engine Start Failed", 
                f"Trading engine failed to start: {str(e)}"
            )
            raise
    
    async def stop(self) -> None:
        """Arrête le moteur de trading proprement"""
        if self.state in [EngineState.STOPPED, EngineState.STOPPING]:
            return
        
        self.state = EngineState.STOPPING
        self.logger.info("trading_engine_stopping")
        
        try:
            # 1. Arrêter toutes les stratégies
            for strategy_id, strategy in self.strategies.items():
                if hasattr(strategy, 'stop'):
                    await strategy.stop()
                self.logger.info("strategy_stopped", strategy_id=strategy_id)
            
            # 2. Arrêter tous les composants
            for name, component in self.components.items():
                try:
                    await component.stop()
                    self.logger.info("component_stopped", component_name=name)
                except Exception as e:
                    self.logger.error("component_stop_failed", 
                                    component_name=name, error=str(e))
            
            # 3. Arrêter l'event bus
            await self.event_bus.stop()
            
            # 4. Annuler toutes les tâches en cours
            for task in self._tasks:
                if not task.done():
                    task.cancel()
            
            if self._tasks:
                await asyncio.gather(*self._tasks, return_exceptions=True)
            
            self.state = EngineState.STOPPED
            self.logger.info("trading_engine_stopped")
            
        except Exception as e:
            self.state = EngineState.ERROR
            self.logger.error("trading_engine_stop_failed", error=str(e))
    
    @circuit_breaker(failure_threshold=5, recovery_timeout=30)
    async def emergency_stop(self, reason: str = "Manual emergency stop") -> None:
        """Arrêt d'urgence du système"""
        self.state = EngineState.EMERGENCY_STOP
        
        await self.alert_manager.send_critical_alert(
            "EMERGENCY STOP ACTIVATED",
            f"Trading engine emergency stop: {reason}"
        )
        
        self.logger.critical("emergency_stop_activated", reason=reason)
        
        # Liquider toutes les positions si possible
        if self.execution_engine and hasattr(self.execution_engine, 'emergency_liquidate'):
            try:
                await self.execution_engine.emergency_liquidate()
            except Exception as e:
                self.logger.error("emergency_liquidation_failed", error=str(e))
        
        # Arrêter le système
        self._shutdown_event.set()
    
    async def _health_monitor(self) -> None:
        """Monitore la santé de tous les composants"""
        while not self._shutdown_event.is_set():
            try:
                health_status = {}
                
                for name, component in self.components.items():
                    try:
                        health = await component.health_check()
                        health_status[name] = health
                        
                        # Alertes sur problèmes de santé
                        if not health.get('healthy', True):
                            await self.alert_manager.send_warning_alert(
                                f"Component Unhealthy: {name}",
                                f"Health check failed: {health}"
                            )
                    
                    except Exception as e:
                        health_status[name] = {"healthy": False, "error": str(e)}
                        self.logger.error("health_check_failed", 
                                        component_name=name, error=str(e))
                
                await self._publish_event(
                    EventType.SYSTEM_STATUS,
                    {"health_status": health_status},
                    source="health_monitor"
                )
                
                await asyncio.sleep(self._health_check_interval)
                
            except Exception as e:
                self.logger.error("health_monitor_error", error=str(e))
                await asyncio.sleep(5)  # Retry plus rapidement en cas d'erreur
    
    async def _metrics_collector(self) -> None:
        """Collecte et publie les métriques système"""
        while not self._shutdown_event.is_set():
            try:
                # Métriques système
                self.metrics.gauge("engine_state", 1, tags={"state": self.state.value})
                self.metrics.gauge("active_strategies", len(self.strategies))
                self.metrics.gauge("active_components", len(self.components))
                self.metrics.gauge("event_queue_size", self.event_bus.event_queue.qsize())
                
                await asyncio.sleep(5)  # Collecte toutes les 5 secondes
                
            except Exception as e:
                self.logger.error("metrics_collection_error", error=str(e))
                await asyncio.sleep(10)
    
    async def _publish_event(self, event_type: EventType, data: Dict[str, Any], 
                           source: str, correlation_id: Optional[str] = None) -> None:
        """Publie un événement et l'ajoute au store pour audit"""
        event = TradingEvent(
            event_type=event_type,
            data=data,
            source=source,
            correlation_id=correlation_id
        )
        
        # Event sourcing - ajouter au store d'audit
        self.event_store.append(event)
        
        # Publier sur l'event bus
        await self.event_bus.publish(event)
    
    def _setup_signal_handlers(self) -> None:
        """Configure les gestionnaires de signaux Unix"""
        def signal_handler(signum, frame):
            self.logger.info("signal_received", signal=signum)
            asyncio.create_task(self.stop())
        
        # Arrêt propre sur SIGTERM et SIGINT
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
        
        # Arrêt d'urgence sur SIGUSR1
        def emergency_signal_handler(signum, frame):
            self.logger.warning("emergency_signal_received", signal=signum)
            asyncio.create_task(self.emergency_stop("Signal-triggered emergency stop"))
        
        signal.signal(signal.SIGUSR1, emergency_signal_handler)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Retourne le status complet du système"""
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "uptime": time.time() - self.config.start_time,
            "components": list(self.components.keys()),
            "strategies": list(self.strategies.keys()),
            "event_queue_size": self.event_bus.event_queue.qsize(),
            "event_store_size": len(self.event_store),
            "active_tasks": len(self._tasks)
        }
    
    async def hot_swap_strategy(self, old_strategy_id: str, 
                              new_strategy_id: str, new_strategy: Any) -> None:
        """
        Hot-swap d'une stratégie en cours d'exécution
        Permet de changer de stratégie sans arrêter le système
        """
        self.logger.info("strategy_hot_swap_starting", 
                        old_strategy=old_strategy_id, 
                        new_strategy=new_strategy_id)
        
        try:
            # 1. Arrêter l'ancienne stratégie
            if old_strategy_id in self.strategies:
                old_strategy = self.strategies[old_strategy_id]
                if hasattr(old_strategy, 'stop'):
                    await old_strategy.stop()
                del self.strategies[old_strategy_id]
            
            # 2. Enregistrer la nouvelle stratégie
            await self.register_strategy(new_strategy_id, new_strategy)
            
            # 3. Démarrer la nouvelle stratégie
            if hasattr(new_strategy, 'start'):
                await new_strategy.start()
            
            await self._publish_event(
                EventType.STRATEGY_CHANGE,
                {
                    "action": "hot_swap_completed",
                    "old_strategy": old_strategy_id,
                    "new_strategy": new_strategy_id
                },
                source="engine"
            )
            
            self.logger.info("strategy_hot_swap_completed",
                           old_strategy=old_strategy_id,
                           new_strategy=new_strategy_id)
        
        except Exception as e:
            self.logger.error("strategy_hot_swap_failed",
                            old_strategy=old_strategy_id,
                            new_strategy=new_strategy_id,
                            error=str(e))
            raise


# Factory function pour simplifier la création
async def create_trading_engine(config: TradingConfig) -> TradingEngine:
    """Factory function pour créer et configurer le moteur de trading"""
    engine = TradingEngine(config)
    
    # Configuration des composants de base sera ajoutée ici
    # quand nous créerons les autres modules
    
    return engine


if __name__ == "__main__":
    # Code de test/démonstration
    import asyncio
    from config.settings import TradingConfig
    
    async def main():
        config = TradingConfig()
        engine = await create_trading_engine(config)
        
        try:
            await engine.start()
        except KeyboardInterrupt:
            await engine.stop()
    
    asyncio.run(main())