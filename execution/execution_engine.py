"""
Execution Engine - Moteur d'Exécution Haute Performance

Ce module implémente le moteur central d'exécution des ordres, orchestrant tous les
aspects de l'exécution from strategy signals to market execution, avec optimisation
des coûts et minimisation du market impact.

Fonctionnalités principales:
- Smart Order Routing avec algorithmes adaptatifs
- Algorithmes d'exécution (TWAP, VWAP, POV, Implementation Shortfall)
- Gestion complète du lifecycle des ordres
- Integration avec risk management en temps réel
- Fill aggregation et partial fills handling
- Performance monitoring et analytics
- Circuit breakers et error handling
- Event-driven architecture pour latence ultra-faible

Performance cible: <10ms order-to-market, 1000+ trades/second
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
from concurrent.futures import ThreadPoolExecutor
import heapq
from threading import Lock

import numpy as np
import pandas as pd
from decimal import Decimal
import numba
from pydantic import BaseModel, Field
import asyncio_throttle

# Imports internes
from utils.logger import get_logger
from utils.metrics import (
    calculate_vwap, calculate_twap, calculate_implementation_shortfall,
    calculate_fill_rate, calculate_execution_cost
)
from utils.helpers import (
    safe_divide, round_to_tick_size, create_correlation_id, 
    exponential_backoff, circuit_breaker
)
from utils.decorators import async_retry, performance_monitor, risk_check

from execution.smart_routing import (
    SmartRoutingEngine, Order, SubOrder, RoutingDecision, 
    OrderSide, ExecutionUrgency, VenueId
)
from execution.slippage_model import SlippageModel, MarketConditions
from execution.slippage_model import MarketRegime  # Ajout de l'import pour MarketRegime
from risk.risk_monitor import RiskMonitor
from risk.position_sizer import PositionSizer
from monitoring.performance_tracker import PerformanceTracker
from config.settings import EXECUTION_CONFIG

# Types
Symbol = str
Price = Decimal
Quantity = Decimal
OrderId = str
ExecutionId = str
Timestamp = datetime


class OrderStatus(Enum):
    """États possibles d'un ordre"""
    PENDING = "PENDING"                    # En attente de validation
    VALIDATED = "VALIDATED"                # Validé par risk management
    ROUTING = "ROUTING"                    # En cours de routage
    ROUTED = "ROUTED"                     # Routé vers venues
    EXECUTING = "EXECUTING"                # En cours d'exécution
    PARTIALLY_FILLED = "PARTIALLY_FILLED" # Partiellement exécuté
    FILLED = "FILLED"                     # Complètement exécuté
    CANCELLED = "CANCELLED"               # Annulé
    REJECTED = "REJECTED"                 # Rejeté
    FAILED = "FAILED"                     # Échec d'exécution
    EXPIRED = "EXPIRED"                   # Expiré


class ExecutionAlgorithm(Enum):
    """Algorithmes d'exécution disponibles"""
    MARKET = "market"                      # Market order immédiat
    LIMIT = "limit"                       # Limit order
    TWAP = "twap"                         # Time-Weighted Average Price
    VWAP = "vwap"                         # Volume-Weighted Average Price
    POV = "pov"                          # Percentage of Volume
    IMPLEMENTATION_SHORTFALL = "is"       # Implementation Shortfall
    ICEBERG = "iceberg"                   # Iceberg orders
    SNIPER = "sniper"                     # Aggressive execution
    STEALTH = "stealth"                   # Minimal market impact


class FillSide(Enum):
    """Côté du fill"""
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class ExecutionRequest:
    """Requête d'exécution d'un ordre"""
    request_id: str
    strategy_id: str
    symbol: Symbol
    side: OrderSide
    quantity: Quantity
    algorithm: ExecutionAlgorithm
    urgency: ExecutionUrgency
    
    # Contraintes d'exécution
    limit_price: Optional[Price] = None
    stop_price: Optional[Price] = None
    max_slippage_bps: float = 50.0
    max_participation_rate: float = 0.2  # 20% du volume
    time_in_force: str = "DAY"
    
    # Paramètres d'algorithme
    algorithm_params: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    client_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    created_at: Timestamp = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Fill:
    """Détails d'un fill (exécution partielle ou complète)"""
    fill_id: str
    execution_id: str
    order_id: str
    venue_id: VenueId
    symbol: Symbol
    side: FillSide
    quantity: Quantity
    price: Price
    commission: Decimal
    timestamp: Timestamp
    trade_id: Optional[str] = None
    
    @property
    def gross_amount(self) -> Decimal:
        """Montant brut du fill"""
        return self.quantity * self.price
    
    @property
    def net_amount(self) -> Decimal:
        """Montant net après commission"""
        return self.gross_amount - self.commission


@dataclass
class ExecutionReport:
    """Rapport d'exécution détaillé"""
    execution_id: str
    request: ExecutionRequest
    order_id: str
    status: OrderStatus
    
    # Détails d'exécution
    filled_quantity: Quantity
    remaining_quantity: Quantity
    average_fill_price: Optional[Price]
    total_commission: Decimal
    
    # Performance metrics
    implementation_shortfall_bps: Optional[float]
    slippage_bps: Optional[float]
    execution_cost_bps: Optional[float]
    fill_rate: float
    
    # Timing
    created_at: Timestamp
    first_fill_at: Optional[Timestamp]
    completed_at: Optional[Timestamp]
    execution_time_ms: Optional[int]
    
    # Fills
    fills: List[Fill] = field(default_factory=list)
    
    # Routing info
    routing_decision: Optional[RoutingDecision] = None
    venues_used: Set[VenueId] = field(default_factory=set)
    
    @property
    def fill_percentage(self) -> float:
        """Pourcentage de fill"""
        if self.request.quantity > 0:
            return float(self.filled_quantity / self.request.quantity)
        return 0.0
    
    @property
    def is_complete(self) -> bool:
        """Vérifie si l'exécution est complète"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]


class ExecutionContext:
    """Contexte d'exécution pour un ordre"""
    
    def __init__(self, request: ExecutionRequest):
        self.request = request
        self.execution_id = f"exec_{uuid.uuid4().hex[:8]}"
        self.order_id = f"ord_{uuid.uuid4().hex[:8]}"
        
        # État
        self.status = OrderStatus.PENDING
        self.fills: List[Fill] = []
        self.sub_orders: List[SubOrder] = []
        self.routing_decision: Optional[RoutingDecision] = None
        
        # Métriques
        self.start_time = time.time()
        self.execution_start_time: Optional[float] = None
        self.completion_time: Optional[float] = None
        
        # Contrôles
        self.cancel_requested = False
        self.error_count = 0
        self.retry_count = 0
        
        # Lock pour thread safety
        self._lock = Lock()
    
    def add_fill(self, fill: Fill) -> None:
        """Ajoute un fill thread-safe"""
        with self._lock:
            self.fills.append(fill)
            
            # Mettre à jour le statut
            total_filled = sum(f.quantity for f in self.fills)
            if total_filled >= self.request.quantity:
                self.status = OrderStatus.FILLED
                self.completion_time = time.time()
            elif total_filled > 0:
                self.status = OrderStatus.PARTIALLY_FILLED
    
    @property
    def filled_quantity(self) -> Quantity:
        """Quantité totale fillée"""
        return sum(f.quantity for f in self.fills)
    
    @property
    def remaining_quantity(self) -> Quantity:
        """Quantité restante à exécuter"""
        return self.request.quantity - self.filled_quantity
    
    @property
    def average_fill_price(self) -> Optional[Price]:
        """Prix moyen de fill"""
        if not self.fills:
            return None
        
        total_value = sum(f.quantity * f.price for f in self.fills)
        total_quantity = sum(f.quantity for f in self.fills)
        
        return Price(total_value / total_quantity) if total_quantity > 0 else None


class ExecutionEngine:
    """Moteur principal d'exécution des ordres"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or EXECUTION_CONFIG
        self.logger = get_logger(f"{__name__}.ExecutionEngine")
        
        # Composants intégrés
        self.smart_router = SmartRoutingEngine(self.config.get('routing', {}))
        self.slippage_model = SlippageModel(self.config.get('slippage', {}))
        self.risk_monitor = RiskMonitor(self.config.get('risk', {}))
        self.position_sizer = PositionSizer(self.config.get('position_sizing', {}))
        self.performance_tracker = PerformanceTracker()
        
        # État des exécutions
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.completed_executions: deque = deque(maxlen=10000)
        self.execution_queue = asyncio.PriorityQueue()
        
        # Performance et monitoring
        self.execution_metrics = defaultdict(lambda: deque(maxlen=1000))
        self.venue_performance = defaultdict(lambda: deque(maxlen=1000))
        
        # Contrôles de flux
        self.max_concurrent_executions = self.config.get('max_concurrent_executions', 100)
        self.execution_semaphore = asyncio.Semaphore(self.max_concurrent_executions)
        
        # Workers pool
        self.execution_workers = []
        self.is_running = False
        
        # Circuit breakers
        self.circuit_breakers = {}
        
        self.logger.info("ExecutionEngine initialisé")
    
    async def initialize(self) -> None:
        """Initialise le moteur d'exécution"""
        try:
            # Initialiser les composants
            await self.smart_router.initialize()
            
            # Démarrer les workers
            self.is_running = True
            num_workers = self.config.get('execution_workers', 4)
            
            for i in range(num_workers):
                worker = asyncio.create_task(self._execution_worker(f"worker_{i}"))
                self.execution_workers.append(worker)
            
            # Démarrer les tâches de monitoring
            asyncio.create_task(self._monitor_executions())
            asyncio.create_task(self._cleanup_completed_executions())
            asyncio.create_task(self._update_performance_metrics())
            
            self.logger.info(f"ExecutionEngine démarré avec {num_workers} workers")
            
        except Exception as e:
            self.logger.error(f"Erreur initialisation ExecutionEngine: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Arrête proprement le moteur d'exécution"""
        self.logger.info("Arrêt de l'ExecutionEngine...")
        
        self.is_running = False
        
        # Attendre la fin des exécutions actives
        if self.active_executions:
            self.logger.info(f"Attente de {len(self.active_executions)} exécutions actives...")
            
            # Timeout de 30 secondes
            timeout = 30
            start_time = time.time()
            
            while self.active_executions and time.time() - start_time < timeout:
                await asyncio.sleep(0.1)
        
        # Annuler les workers
        for worker in self.execution_workers:
            if not worker.done():
                worker.cancel()
        
        # Attendre l'arrêt des workers
        if self.execution_workers:
            await asyncio.gather(*self.execution_workers, return_exceptions=True)
        
        self.logger.info("ExecutionEngine arrêté")
    
    async def execute_order(self, request: ExecutionRequest) -> str:
        """
        Point d'entrée principal pour l'exécution d'un ordre.
        
        Args:
            request: Requête d'exécution
            
        Returns:
            execution_id: ID unique de l'exécution
        """
        execution_start = time.time()
        
        try:
            # Créer le contexte d'exécution
            context = ExecutionContext(request)
            
            self.logger.info(
                f"Nouvelle exécution {context.execution_id}: "
                f"{request.symbol} {request.side.value} {request.quantity} "
                f"algorithme={request.algorithm.value}"
            )
            
            # Validation pre-trade par risk management
            risk_approval = await self._validate_risk(request, context)
            if not risk_approval['approved']:
                context.status = OrderStatus.REJECTED
                await self._complete_execution(context, risk_approval['reason'])
                return context.execution_id
            
            # Ajuster la taille si nécessaire
            adjusted_quantity = await self._adjust_position_size(request, context)
            if adjusted_quantity != request.quantity:
                self.logger.info(f"Taille ajustée: {request.quantity} -> {adjusted_quantity}")
                request.quantity = adjusted_quantity
            
            # Enregistrer l'exécution
            self.active_executions[context.execution_id] = context
            context.status = OrderStatus.VALIDATED
            
            # Ajouter à la queue d'exécution avec priorité
            priority = self._calculate_execution_priority(request)
            await self.execution_queue.put((priority, time.time(), context))
            
            # Métriques de performance
            validation_time = (time.time() - execution_start) * 1000
            self.execution_metrics['validation_time_ms'].append(validation_time)
            
            return context.execution_id
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'exécution: {e}")
            raise
    
    def _calculate_execution_priority(self, request: ExecutionRequest) -> int:
        """Calcule la priorité d'exécution (plus petit = plus prioritaire)"""
        urgency_priority = {
            ExecutionUrgency.IMMEDIATE: 0,
            ExecutionUrgency.HIGH: 10,
            ExecutionUrgency.NORMAL: 20,
            ExecutionUrgency.LOW: 30
        }
        
        # Facteurs de priorité
        base_priority = urgency_priority.get(request.urgency, 20)
        
        # Ordre de marché = plus prioritaire
        if request.algorithm == ExecutionAlgorithm.MARKET:
            base_priority -= 5
        
        # Gros ordres = moins prioritaires
        if request.quantity > 1000:  # Seuil configurable
            base_priority += 5
        
        return base_priority
    
    async def _validate_risk(
        self, 
        request: ExecutionRequest, 
        context: ExecutionContext
    ) -> Dict[str, Any]:
        """Valide l'ordre avec le risk management"""
        try:
            # Vérifications pré-trade
            risk_check = await self.risk_monitor.validate_order(
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                price=request.limit_price,
                strategy_id=request.strategy_id
            )
            
            if not risk_check['approved']:
                self.logger.warning(f"Ordre rejeté par risk management: {risk_check['reason']}")
            
            return risk_check
            
        except Exception as e:
            self.logger.error(f"Erreur validation risque: {e}")
            return {
                'approved': False,
                'reason': f'Erreur validation risque: {str(e)}'
            }
    
    async def _adjust_position_size(
        self, 
        request: ExecutionRequest, 
        context: ExecutionContext
    ) -> Quantity:
        """Ajuste la taille de position selon les règles de sizing"""
        try:
            optimal_size = await self.position_sizer.calculate_optimal_size(
                symbol=request.symbol,
                side=request.side,
                target_quantity=request.quantity,
                strategy_id=request.strategy_id,
                market_conditions={}  # À enrichir avec données de marché
            )
            
            # Respecter la demande originale si plus petite
            return min(optimal_size, request.quantity)
            
        except Exception as e:
            self.logger.error(f"Erreur ajustement taille: {e}")
            return request.quantity
    
    async def _execution_worker(self, worker_name: str) -> None:
        """Worker pour traiter les exécutions de la queue"""
        self.logger.info(f"Worker {worker_name} démarré")
        
        while self.is_running:
            try:
                # Attendre un ordre à exécuter (timeout 1s)
                try:
                    priority, timestamp, context = await asyncio.wait_for(
                        self.execution_queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Limiter les exécutions concurrentes
                async with self.execution_semaphore:
                    await self._process_execution(context, worker_name)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Erreur dans worker {worker_name}: {e}")
                await asyncio.sleep(1)  # Éviter la boucle d'erreur
        
        self.logger.info(f"Worker {worker_name} arrêté")
    
    async def _process_execution(self, context: ExecutionContext, worker_name: str) -> None:
        """Traite une exécution complète"""
        execution_id = context.execution_id
        request = context.request
        
        try:
            self.logger.debug(f"[{worker_name}] Traitement exécution {execution_id}")
            
            context.execution_start_time = time.time()
            context.status = OrderStatus.ROUTING
            
            # Routage intelligent de l'ordre
            routing_decision = await self._route_order(context)
            if not routing_decision:
                context.status = OrderStatus.FAILED
                await self._complete_execution(context, "Échec du routage")
                return
            
            context.routing_decision = routing_decision
            context.sub_orders = routing_decision.sub_orders
            context.status = OrderStatus.ROUTED
            
            # Exécuter selon l'algorithme choisi
            success = await self._execute_algorithm(context, routing_decision)
            
            if not success:
                context.status = OrderStatus.FAILED
                await self._complete_execution(context, "Échec d'exécution")
                return
            
            # Vérifier si complètement exécuté
            if context.remaining_quantity <= 0:
                context.status = OrderStatus.FILLED
                await self._complete_execution(context, "Complètement exécuté")
            
        except Exception as e:
            self.logger.error(f"Erreur traitement exécution {execution_id}: {e}")
            context.status = OrderStatus.FAILED
            context.error_count += 1
            await self._complete_execution(context, f"Erreur: {str(e)}")
    
    async def _route_order(self, context: ExecutionContext) -> Optional[RoutingDecision]:
        """Route l'ordre avec le smart router"""
        try:
            request = context.request
            
            # Créer l'ordre pour le routeur
            order = Order(
                order_id=context.order_id,
                symbol=request.symbol,
                side=request.side,
                quantity=request.quantity,
                order_type=request.algorithm.value.upper(),
                urgency=request.urgency,
                max_slippage_bps=request.max_slippage_bps,
                time_in_force=request.time_in_force,
                client_id=request.strategy_id,
                strategy_id=request.strategy_id,
                limit_price=request.limit_price,
                max_participation_rate=request.max_participation_rate
            )
            
            # Router avec circuit breaker
            routing_decision = await circuit_breaker(
                self._route_with_circuit_breaker,
                order,
                max_failures=3,
                reset_timeout=60
            )
            
            self.logger.info(
                f"Ordre routé: {len(routing_decision.sub_orders) if routing_decision else 0} sous-ordres, "
                f"algorithme: {routing_decision.algorithm_used.value if routing_decision else 'N/A'}"
            )
            
            return routing_decision
            
        except Exception as e:
            self.logger.error(f"Erreur routage: {e}")
            return None
    
    async def _route_with_circuit_breaker(self, order: Order) -> RoutingDecision:
        """Wrapper pour routage avec circuit breaker"""
        return await self.smart_router.route_order(order)
    
    async def _execute_algorithm(
        self, 
        context: ExecutionContext, 
        routing_decision: RoutingDecision
    ) -> bool:
        """Exécute l'algorithme choisi"""
        algorithm = context.request.algorithm
        
        try:
            if algorithm == ExecutionAlgorithm.MARKET:
                return await self._execute_market_algorithm(context, routing_decision)
            
            elif algorithm == ExecutionAlgorithm.LIMIT:
                return await self._execute_limit_algorithm(context, routing_decision)
            
            elif algorithm == ExecutionAlgorithm.TWAP:
                return await self._execute_twap_algorithm(context, routing_decision)
            
            elif algorithm == ExecutionAlgorithm.VWAP:
                return await self._execute_vwap_algorithm(context, routing_decision)
            
            elif algorithm == ExecutionAlgorithm.POV:
                return await self._execute_pov_algorithm(context, routing_decision)
            
            elif algorithm == ExecutionAlgorithm.ICEBERG:
                return await self._execute_iceberg_algorithm(context, routing_decision)
            
            else:
                # Fallback vers market
                self.logger.warning(f"Algorithme non supporté {algorithm}, fallback vers MARKET")
                return await self._execute_market_algorithm(context, routing_decision)
                
        except Exception as e:
            self.logger.error(f"Erreur exécution algorithme {algorithm}: {e}")
            return False
    
    async def _execute_market_algorithm(
        self, 
        context: ExecutionContext, 
        routing_decision: RoutingDecision
    ) -> bool:
        """Exécute avec algorithme market (immédiat)"""
        context.status = OrderStatus.EXECUTING
        
        # Exécuter tous les sous-ordres en parallèle
        tasks = []
        for sub_order in routing_decision.sub_orders:
            task = asyncio.create_task(self._execute_sub_order(context, sub_order))
            tasks.append(task)
        
        # Attendre toutes les exécutions
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Compter les succès
        success_count = sum(1 for result in results if result is True)
        total_count = len(results)
        
        self.logger.info(f"Market execution: {success_count}/{total_count} sous-ordres réussis")
        
        return success_count > 0
    
    async def _execute_twap_algorithm(
        self, 
        context: ExecutionContext, 
        routing_decision: RoutingDecision
    ) -> bool:
        """Exécute avec algorithme TWAP (Time-Weighted Average Price)"""
        context.status = OrderStatus.EXECUTING
        
        # Paramètres TWAP
        params = context.request.algorithm_params
        execution_horizon_minutes = params.get('horizon_minutes', 30)
        slice_interval_minutes = params.get('slice_interval', 2)
        
        # Trier les sous-ordres par timing
        scheduled_orders = sorted(
            routing_decision.sub_orders, 
            key=lambda x: x.execution_delay_ms
        )
        
        self.logger.info(
            f"TWAP execution: {len(scheduled_orders)} tranches sur {execution_horizon_minutes} minutes"
        )
        
        # Exécuter les tranches séquentiellement
        for i, sub_order in enumerate(scheduled_orders):
            if context.cancel_requested:
                break
            
            # Attendre le délai programmé
            if sub_order.execution_delay_ms > 0:
                await asyncio.sleep(sub_order.execution_delay_ms / 1000.0)
            
            # Exécuter la tranche
            success = await self._execute_sub_order(context, sub_order)
            
            if success:
                self.logger.debug(f"TWAP tranche {i+1}/{len(scheduled_orders)} exécutée")
            else:
                self.logger.warning(f"TWAP tranche {i+1} échouée")
        
        return len(context.fills) > 0
    
    async def _execute_iceberg_algorithm(
        self, 
        context: ExecutionContext, 
        routing_decision: RoutingDecision
    ) -> bool:
        """Exécute avec algorithme iceberg (masquage de taille)"""
        context.status = OrderStatus.EXECUTING
        
        # Paramètres iceberg
        params = context.request.algorithm_params
        visible_size = params.get('visible_size', context.request.quantity * Decimal('0.1'))
        refresh_interval_ms = params.get('refresh_interval_ms', 5000)
        
        remaining_qty = context.request.quantity
        
        self.logger.info(f"Iceberg execution: taille visible {visible_size}")
        
        while remaining_qty > 0 and not context.cancel_requested:
            # Taille de la tranche actuelle
            current_slice = min(visible_size, remaining_qty)
            
            # Créer un sous-ordre pour cette tranche
            slice_order = SubOrder(
                sub_order_id=f"{context.order_id}_iceberg_{len(context.fills)}",
                parent_correlation_id=context.execution_id,
                venue_id=routing_decision.sub_orders[0].venue_id,  # Première venue
                symbol=context.request.symbol,
                side=context.request.side,
                quantity=current_slice,
                price=context.request.limit_price,
                order_type='LIMIT',
                estimated_cost=Decimal('0'),
                estimated_slippage_bps=0.0,
                priority=0
            )
            
            # Exécuter la tranche
            success = await self._execute_sub_order(context, slice_order)
            
            if success:
                # Mettre à jour la quantité restante
                filled_in_slice = sum(
                    f.quantity for f in context.fills 
                    if f.timestamp >= slice_order.creation_time
                )
                remaining_qty -= filled_in_slice
                
                # Attendre avant la prochaine tranche
                if remaining_qty > 0:
                    await asyncio.sleep(refresh_interval_ms / 1000.0)
            else:
                self.logger.warning("Échec tranche iceberg, arrêt")
                break
        
        return len(context.fills) > 0
    
    async def _execute_sub_order(self, context: ExecutionContext, sub_order: SubOrder) -> bool:
        """Exécute un sous-ordre individuel"""
        try:
            # Simulation d'exécution (à remplacer par vraie exécution venue)
            await asyncio.sleep(0.01)  # Simule latency réseau
            
            # Simuler un fill (à remplacer par vraie logique venue)
            fill = self._simulate_fill(context, sub_order)
            
            if fill:
                context.add_fill(fill)
                
                # Notifier le fill
                await self._notify_fill(context, fill)
                
                # Enregistrer pour le slippage model
                await self._record_execution_for_slippage_model(context, sub_order, fill)
                
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Erreur exécution sous-ordre {sub_order.sub_order_id}: {e}")
            return False
    
    def _simulate_fill(self, context: ExecutionContext, sub_order: SubOrder) -> Optional[Fill]:
        """Simule un fill (à remplacer par vraie intégration venue)"""
        # Simulation basique - à remplacer par vraie exécution
        
        # Prix simulé avec léger slippage
        if sub_order.order_type == 'MARKET':
            # Market order: prix avec slippage
            base_price = Decimal('50000')  # À remplacer par prix réel
            slippage_bps = np.random.normal(2.0, 1.0)  # Slippage simulé
            slippage_factor = 1 + (slippage_bps / 10000)
            
            if context.request.side == OrderSide.BUY:
                fill_price = base_price * Decimal(str(slippage_factor))
            else:
                fill_price = base_price * Decimal(str(1/slippage_factor))
        else:
            # Limit order: prix demandé
            fill_price = sub_order.price or Decimal('50000')
        
        # Commission simulée
        commission_rate = Decimal('0.001')  # 0.1%
        commission = sub_order.quantity * fill_price * commission_rate
        
        fill = Fill(
            fill_id=f"fill_{uuid.uuid4().hex[:8]}",
            execution_id=context.execution_id,
            order_id=context.order_id,
            venue_id=sub_order.venue_id,
            symbol=sub_order.symbol,
            side=FillSide(sub_order.side.value),
            quantity=sub_order.quantity,
            price=fill_price,
            commission=commission,
            timestamp=datetime.now(timezone.utc),
            trade_id=f"trade_{uuid.uuid4().hex[:8]}"
        )
        
        return fill
    
    async def _notify_fill(self, context: ExecutionContext, fill: Fill) -> None:
        """Notifie les systèmes de l'arrivée d'un fill"""
        try:
            # Mettre à jour le portfolio manager
            # await self.portfolio_manager.process_fill(fill)
            
            # Notifier le risk monitor
            await self.risk_monitor.update_position(
                symbol=fill.symbol,
                side=fill.side.value,
                quantity=fill.quantity,
                price=fill.price
            )
            
            # Événement pour monitoring
            await self.performance_tracker.record_fill(fill)
            
            self.logger.debug(f"Fill notifié: {fill.quantity} @ {fill.price}")
            
        except Exception as e:
            self.logger.error(f"Erreur notification fill: {e}")
    
    async def _record_execution_for_slippage_model(
        self, 
        context: ExecutionContext, 
        sub_order: SubOrder, 
        fill: Fill
    ) -> None:
        """Enregistre l'exécution pour l'apprentissage du slippage model"""
        try:
            # Créer les conditions de marché (simplifiées)
            market_conditions = MarketConditions(
                symbol=fill.symbol,
                timestamp=fill.timestamp,
                bid_price=fill.price * Decimal('0.999'),
                ask_price=fill.price * Decimal('1.001'),
                bid_size=Quantity('1000'),
                ask_size=Quantity('1000'),
                mid_price=fill.price,
                spread_bps=10.0,
                volatility_1min=0.01,
                volatility_5min=0.008,
                volume_1min=Quantity('10000'),
                volume_avg_daily=Quantity('1000000'),
                market_regime=MarketRegime.NORMAL,
                time_of_day='session'
            )
            
            # Enregistrer dans le slippage model
            await self.slippage_model.record_execution(
                symbol=fill.symbol,
                order_size=fill.quantity,
                order_side=OrderSide(fill.side.value),
                expected_price=fill.price,  # Simplification
                executed_price=fill.price,
                market_conditions=market_conditions,
                execution_time_ms=50  # Simplification
            )
            
        except Exception as e:
            self.logger.error(f"Erreur enregistrement slippage model: {e}")
    
    async def _complete_execution(self, context: ExecutionContext, reason: str) -> None:
        """Finalise une exécution"""
        try:
            context.completion_time = time.time()
            
            # Générer le rapport d'exécution
            report = self._generate_execution_report(context)
            
            # Déplacer vers les exécutions complétées
            if context.execution_id in self.active_executions:
                del self.active_executions[context.execution_id]
            
            self.completed_executions.append((context, report))
            
            # Enregistrer les métriques
            await self._record_execution_metrics(context, report)
            
            self.logger.info(
                f"Exécution {context.execution_id} terminée: {reason}, "
                f"statut={context.status.value}, "
                f"fills={len(context.fills)}, "
                f"taux_fill={report.fill_rate:.1%}"
            )
            
        except Exception as e:
            self.logger.error(f"Erreur finalisation exécution: {e}")
    
    def _generate_execution_report(self, context: ExecutionContext) -> ExecutionReport:
        """Génère un rapport d'exécution détaillé"""
        
        filled_qty = context.filled_quantity
        remaining_qty = context.remaining_quantity
        avg_price = context.average_fill_price
        total_commission = sum(f.commission for f in context.fills)
        
        # Calculer les métriques de performance
        fill_rate = float(filled_qty / context.request.quantity) if context.request.quantity > 0 else 0
        
        # Timing
        created_at = context.request.created_at
        first_fill_at = context.fills[0].timestamp if context.fills else None
        completed_at = datetime.now(timezone.utc) if context.completion_time else None
        
        execution_time_ms = None
        if context.execution_start_time and context.completion_time:
            execution_time_ms = int((context.completion_time - context.execution_start_time) * 1000)
        
        # Venues utilisées
        venues_used = set(f.venue_id for f in context.fills)
        
        report = ExecutionReport(
            execution_id=context.execution_id,
            request=context.request,
            order_id=context.order_id,
            status=context.status,
            filled_quantity=filled_qty,
            remaining_quantity=remaining_qty,
            average_fill_price=avg_price,
            total_commission=total_commission,
            implementation_shortfall_bps=None,  # À calculer si besoin
            slippage_bps=None,  # À calculer si besoin
            execution_cost_bps=None,  # À calculer si besoin
            fill_rate=fill_rate,
            created_at=created_at,
            first_fill_at=first_fill_at,
            completed_at=completed_at,
            execution_time_ms=execution_time_ms,
            fills=context.fills.copy(),
            routing_decision=context.routing_decision,
            venues_used=venues_used
        )
        
        return report
    
    async def _record_execution_metrics(
        self, 
        context: ExecutionContext, 
        report: ExecutionReport
    ) -> None:
        """Enregistre les métriques d'exécution"""
        try:
            algorithm = context.request.algorithm.value
            
            # Métriques d'exécution
            self.execution_metrics[f'{algorithm}_fill_rate'].append(report.fill_rate)
            
            if report.execution_time_ms:
                self.execution_metrics[f'{algorithm}_execution_time_ms'].append(report.execution_time_ms)
            
            # Métriques par venue
            for venue_id in report.venues_used:
                venue_fills = [f for f in report.fills if f.venue_id == venue_id]
                if venue_fills:
                    avg_fill_price_venue = sum(f.price * f.quantity for f in venue_fills) / sum(f.quantity for f in venue_fills)
                    self.venue_performance[venue_id].append(float(avg_fill_price_venue))
            
            # Envoyer au performance tracker
            await self.performance_tracker.record_execution(report)
            
        except Exception as e:
            self.logger.error(f"Erreur enregistrement métriques: {e}")
    
    async def cancel_execution(self, execution_id: str, reason: str = "User cancellation") -> bool:
        """Annule une exécution en cours"""
        try:
            if execution_id not in self.active_executions:
                self.logger.warning(f"Exécution {execution_id} non trouvée pour annulation")
                return False
            
            context = self.active_executions[execution_id]
            context.cancel_requested = True
            context.status = OrderStatus.CANCELLED
            
            # Annuler les sous-ordres actifs (à implémenter selon venues)
            # for sub_order in context.sub_orders:
            #     await self._cancel_sub_order(sub_order)
            
            await self._complete_execution(context, reason)
            
            self.logger.info(f"Exécution {execution_id} annulée: {reason}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur annulation exécution {execution_id}: {e}")
            return False
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Retourne le statut d'une exécution"""
        
        # Chercher dans les exécutions actives
        if execution_id in self.active_executions:
            context = self.active_executions[execution_id]
            return {
                'execution_id': execution_id,
                'status': context.status.value,
                'filled_quantity': float(context.filled_quantity),
                'remaining_quantity': float(context.remaining_quantity),
                'fill_count': len(context.fills),
                'average_fill_price': float(context.average_fill_price) if context.average_fill_price else None,
                'created_at': context.request.created_at.isoformat(),
                'is_active': True
            }
        
        # Chercher dans les exécutions complétées
        for context, report in self.completed_executions:
            if context.execution_id == execution_id:
                return {
                    'execution_id': execution_id,
                    'status': report.status.value,
                    'filled_quantity': float(report.filled_quantity),
                    'remaining_quantity': float(report.remaining_quantity),
                    'fill_count': len(report.fills),
                    'average_fill_price': float(report.average_fill_price) if report.average_fill_price else None,
                    'fill_rate': report.fill_rate,
                    'execution_time_ms': report.execution_time_ms,
                    'created_at': report.created_at.isoformat(),
                    'completed_at': report.completed_at.isoformat() if report.completed_at else None,
                    'is_active': False
                }
        
        return None
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques d'exécution"""
        
        # Statistiques générales
        total_executions = len(self.completed_executions)
        active_executions = len(self.active_executions)
        
        # Métriques par algorithme
        algorithm_stats = {}
        for algorithm in ExecutionAlgorithm:
            key = algorithm.value
            fill_rates = list(self.execution_metrics.get(f'{key}_fill_rate', []))
            exec_times = list(self.execution_metrics.get(f'{key}_execution_time_ms', []))
            
            if fill_rates or exec_times:
                algorithm_stats[key] = {
                    'avg_fill_rate': np.mean(fill_rates) if fill_rates else 0,
                    'avg_execution_time_ms': np.mean(exec_times) if exec_times else 0,
                    'execution_count': len(fill_rates)
                }
        
        # Performance par venue
        venue_stats = {}
        for venue_id, prices in self.venue_performance.items():
            if prices:
                venue_stats[venue_id] = {
                    'avg_fill_price': np.mean(prices),
                    'fill_count': len(prices),
                    'price_std': np.std(prices)
                }
        
        return {
            'total_completed_executions': total_executions,
            'active_executions': active_executions,
            'algorithm_performance': algorithm_stats,
            'venue_performance': venue_stats,
            'avg_validation_time_ms': np.mean(self.execution_metrics.get('validation_time_ms', [])) if self.execution_metrics.get('validation_time_ms') else 0,
            'is_running': self.is_running,
            'queue_size': self.execution_queue.qsize()
        }
    
    async def _monitor_executions(self) -> None:
        """Monitore les exécutions en cours"""
        while self.is_running:
            try:
                current_time = time.time()
                expired_executions = []
                
                for execution_id, context in self.active_executions.items():
                    # Vérifier les timeouts
                    execution_age = current_time - context.start_time
                    
                    # Timeout configurable par algorithme
                    timeout_seconds = self.config.get('execution_timeout_seconds', 300)  # 5 minutes
                    
                    if execution_age > timeout_seconds:
                        expired_executions.append((execution_id, context))
                
                # Traiter les exécutions expirées
                for execution_id, context in expired_executions:
                    self.logger.warning(f"Exécution {execution_id} expirée après {timeout_seconds}s")
                    context.status = OrderStatus.EXPIRED
                    await self._complete_execution(context, "Timeout")
                
                await asyncio.sleep(10)  # Vérification toutes les 10 secondes
                
            except Exception as e:
                self.logger.error(f"Erreur monitoring exécutions: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_completed_executions(self) -> None:
        """Nettoie les anciennes exécutions complétées"""
        while self.is_running:
            try:
                # Garder seulement les N dernières exécutions
                max_completed = self.config.get('max_completed_executions', 10000)
                
                while len(self.completed_executions) > max_completed:
                    self.completed_executions.popleft()
                
                await asyncio.sleep(3600)  # Nettoyage chaque heure
                
            except Exception as e:
                self.logger.error(f"Erreur nettoyage exécutions: {e}")
                await asyncio.sleep(3600)
    
    async def _update_performance_metrics(self) -> None:
        """Met à jour les métriques de performance"""
        while self.is_running:
            try:
                # Calculer et envoyer les métriques au performance tracker
                stats = self.get_execution_statistics()
                await self.performance_tracker.update_execution_metrics(stats)
                
                await asyncio.sleep(60)  # Mise à jour chaque minute
                
            except Exception as e:
                self.logger.error(f"Erreur mise à jour métriques: {e}")
                await asyncio.sleep(60)


# Exemple d'utilisation
if __name__ == "__main__":
    async def main():
        # Configuration
        config = {
            'max_concurrent_executions': 50,
            'execution_workers': 4,
            'execution_timeout_seconds': 300
        }
        
        # Créer et initialiser le moteur
        engine = ExecutionEngine(config)
        await engine.initialize()
        
        try:
            # Créer une requête d'exécution
            request = ExecutionRequest(
                request_id="req_001",
                strategy_id="arbitrage_btc",
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                quantity=Quantity("1.5"),
                algorithm=ExecutionAlgorithm.TWAP,
                urgency=ExecutionUrgency.NORMAL,
                max_slippage_bps=10.0,
                algorithm_params={
                    'horizon_minutes': 15,
                    'slice_interval': 1
                }
            )
            
            # Exécuter l'ordre
            execution_id = await engine.execute_order(request)
            print(f"Exécution lancée: {execution_id}")
            
            # Suivre le statut
            for i in range(20):  # 20 secondes de monitoring
                status = engine.get_execution_status(execution_id)
                if status:
                    print(f"Statut: {status['status']}, Fill: {status['filled_quantity']}")
                    
                    if not status['is_active']:
                        break
                
                await asyncio.sleep(1)
            
            # Statistiques finales
            stats = engine.get_execution_statistics()
            print(f"Statistiques: {stats}")
            
        finally:
            await engine.shutdown()
    
    # Exécution
    asyncio.run(main())