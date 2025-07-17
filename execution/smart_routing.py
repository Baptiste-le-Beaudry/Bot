"""
Smart Order Routing - Routage Intelligent des Ordres

Ce module implémente un système de routage intelligent (SOR) pour optimiser l'exécution
des ordres across multiple venues d'exécution, minimisant les coûts et maximisant
l'efficacité d'exécution.

Fonctionnalités principales:
- Routage multi-venues (Binance, Coinbase, Kraken, etc.)
- Algorithmes d'optimisation (TWAP, VWAP, POV)
- Machine Learning pour l'optimisation adaptative
- Minimisation du market impact et du slippage
- Splitting intelligent des ordres volumineux
- Monitoring en temps réel des performances

Performance cible: <10ms order-to-market, 95%+ fill rate
"""

import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
from concurrent.futures import ThreadPoolExecutor
import heapq

import numpy as np
import pandas as pd
from decimal import Decimal
import numba
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import asyncio_throttle

# Imports internes
from utils.logger import get_logger
from utils.metrics import calculate_execution_shortfall, calculate_vwap_slippage
from utils.helpers import safe_divide, round_to_tick_size, create_correlation_id
from execution.slippage_model import SlippageModel, MarketConditions
from config.settings import ROUTING_CONFIG, VENUE_CONFIGS
from monitoring.performance_tracker import PerformanceTracker

# Types
Symbol = str
Price = Decimal
Quantity = Decimal
OrderId = str
VenueId = str
Timestamp = datetime


class RoutingAlgorithm(Enum):
    """Algorithmes de routage disponibles"""
    BEST_PRICE = "best_price"              # Meilleur prix disponible
    BEST_LIQUIDITY = "best_liquidity"     # Meilleure liquidité
    LOWEST_COST = "lowest_cost"           # Coût total minimal
    SMART_ADAPTIVE = "smart_adaptive"     # Adaptatif avec ML
    TWAP = "twap"                         # Time-Weighted Average Price
    VWAP = "vwap"                         # Volume-Weighted Average Price
    POV = "pov"                           # Percentage of Volume
    ICEBERG = "iceberg"                   # Ordres iceberg
    IMPLEMENTATION_SHORTFALL = "is"       # Implementation Shortfall


class VenueType(Enum):
    """Types de venues d'exécution"""
    EXCHANGE = "exchange"                  # Exchange public
    DARK_POOL = "dark_pool"               # Dark pool
    ECN = "ecn"                           # Electronic Communication Network
    MARKET_MAKER = "market_maker"         # Market maker
    CROSS_NETWORK = "cross_network"       # Crossing network


class OrderSide(Enum):
    """Côté de l'ordre"""
    BUY = "BUY"
    SELL = "SELL"


class ExecutionUrgency(Enum):
    """Urgence d'exécution"""
    LOW = "low"           # Pas urgent, optimiser les coûts
    NORMAL = "normal"     # Équilibre coût/vitesse
    HIGH = "high"         # Urgent, privilégier la vitesse
    IMMEDIATE = "immediate"  # Exécution immédiate requise


@dataclass
class VenueInfo:
    """Informations sur une venue d'exécution"""
    venue_id: VenueId
    name: str
    venue_type: VenueType
    supported_symbols: List[Symbol]
    min_order_size: Quantity
    max_order_size: Quantity
    tick_size: Decimal
    maker_fee: Decimal  # Frais maker en %
    taker_fee: Decimal  # Frais taker en %
    api_latency_ms: float
    connectivity_score: float  # 0-1
    reliability_score: float   # 0-1
    average_fill_rate: float   # 0-1
    supported_order_types: List[str]
    trading_hours: Dict[str, Any]
    
    @property
    def total_score(self) -> float:
        """Score global de la venue"""
        return (self.connectivity_score * 0.3 + 
                self.reliability_score * 0.4 + 
                self.average_fill_rate * 0.3)


@dataclass
class MarketDepth:
    """Profondeur du marché pour une venue"""
    venue_id: VenueId
    symbol: Symbol
    timestamp: Timestamp
    bids: List[Tuple[Price, Quantity]]  # [(price, quantity), ...]
    asks: List[Tuple[Price, Quantity]]
    last_price: Price
    volume_24h: Quantity
    
    @property
    def best_bid(self) -> Optional[Tuple[Price, Quantity]]:
        """Meilleur bid"""
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[Tuple[Price, Quantity]]:
        """Meilleur ask"""
        return self.asks[0] if self.asks else None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """Spread bid-ask"""
        if self.best_bid and self.best_ask:
            return self.best_ask[0] - self.best_bid[0]
        return None
    
    @property
    def mid_price(self) -> Optional[Price]:
        """Prix mid"""
        if self.best_bid and self.best_ask:
            return (self.best_bid[0] + self.best_ask[0]) / 2
        return None
    
    def get_available_liquidity(self, side: OrderSide, max_levels: int = 5) -> Quantity:
        """Liquidité disponible sur N niveaux"""
        levels = self.asks if side == OrderSide.BUY else self.bids
        return sum(qty for _, qty in levels[:max_levels])


@dataclass
class RoutingDecision:
    """Décision de routage pour un ordre"""
    correlation_id: str
    original_order: 'Order'
    sub_orders: List['SubOrder']
    algorithm_used: RoutingAlgorithm
    estimated_cost_bps: float
    estimated_slippage_bps: float
    estimated_execution_time_ms: int
    reasoning: Dict[str, Any]
    confidence_score: float
    
    @property
    def total_estimated_cost(self) -> Decimal:
        """Coût total estimé"""
        return sum(sub_order.estimated_cost for sub_order in self.sub_orders)


@dataclass
class SubOrder:
    """Sous-ordre routé vers une venue spécifique"""
    sub_order_id: str
    parent_correlation_id: str
    venue_id: VenueId
    symbol: Symbol
    side: OrderSide
    quantity: Quantity
    price: Optional[Price]  # None pour market orders
    order_type: str  # 'MARKET', 'LIMIT', 'ICEBERG', etc.
    estimated_cost: Decimal
    estimated_slippage_bps: float
    priority: int  # Ordre d'exécution (plus petit = plus prioritaire)
    execution_delay_ms: int = 0  # Délai avant exécution
    
    # Métadonnées d'exécution
    creation_time: Timestamp = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_time: Optional[Timestamp] = None
    actual_fill_quantity: Quantity = Quantity('0')
    actual_fill_price: Optional[Price] = None
    actual_cost: Optional[Decimal] = None
    status: str = 'PENDING'  # PENDING, EXECUTING, FILLED, PARTIAL, CANCELLED


@dataclass
class Order:
    """Ordre principal à router"""
    order_id: OrderId
    symbol: Symbol
    side: OrderSide
    quantity: Quantity
    order_type: str
    urgency: ExecutionUrgency
    max_slippage_bps: float
    time_in_force: str  # 'IOC', 'FOK', 'GTC', etc.
    client_id: str
    strategy_id: str
    
    # Contraintes d'exécution
    limit_price: Optional[Price] = None
    min_fill_quantity: Optional[Quantity] = None
    max_participation_rate: float = 0.1  # 10% du volume
    preferred_venues: List[VenueId] = field(default_factory=list)
    excluded_venues: List[VenueId] = field(default_factory=list)
    
    # Timing
    creation_time: Timestamp = field(default_factory=lambda: datetime.now(timezone.utc))
    start_time: Optional[Timestamp] = None
    end_time: Optional[Timestamp] = None
    
    @property
    def is_buy(self) -> bool:
        return self.side == OrderSide.BUY


class VenueConnector:
    """Connecteur pour une venue d'exécution"""
    
    def __init__(self, venue_info: VenueInfo, config: Dict[str, Any]):
        self.venue_info = venue_info
        self.config = config
        self.logger = get_logger(f"{__name__}.{venue_info.venue_id}")
        
        # État de la connexion
        self.is_connected = False
        self.last_heartbeat = None
        self.connection_errors = 0
        
        # Cache des données de marché
        self.market_depth_cache: Dict[Symbol, MarketDepth] = {}
        self.cache_timestamps: Dict[Symbol, float] = {}
        
        # Métriques de performance
        self.execution_latencies = deque(maxlen=1000)
        self.fill_rates = deque(maxlen=1000)
        self.execution_costs = deque(maxlen=1000)
        
        # Rate limiting
        self.rate_limiter = asyncio_throttle.Throttler(
            rate_limit=config.get('rate_limit', 100),
            period=1.0
        )
    
    async def connect(self) -> bool:
        """Établit la connexion avec la venue"""
        try:
            # Implémentation spécifique à chaque venue
            await self._establish_connection()
            self.is_connected = True
            self.connection_errors = 0
            self.logger.info(f"Connecté à {self.venue_info.name}")
            return True
        except Exception as e:
            self.connection_errors += 1
            self.logger.error(f"Erreur connexion {self.venue_info.name}: {e}")
            return False
    
    async def _establish_connection(self):
        """Implémentation spécifique de la connexion"""
        # À implémenter pour chaque venue
        pass
    
    async def get_market_depth(self, symbol: Symbol, force_refresh: bool = False) -> Optional[MarketDepth]:
        """Récupère la profondeur du marché"""
        # Vérifier le cache
        cache_ttl = self.config.get('cache_ttl_ms', 100) / 1000.0
        now = time.time()
        
        if (not force_refresh and 
            symbol in self.market_depth_cache and 
            now - self.cache_timestamps.get(symbol, 0) < cache_ttl):
            return self.market_depth_cache[symbol]
        
        try:
            async with self.rate_limiter:
                depth = await self._fetch_market_depth(symbol)
                if depth:
                    self.market_depth_cache[symbol] = depth
                    self.cache_timestamps[symbol] = now
                return depth
        except Exception as e:
            self.logger.error(f"Erreur récupération depth {symbol}: {e}")
            return None
    
    async def _fetch_market_depth(self, symbol: Symbol) -> Optional[MarketDepth]:
        """Implémentation spécifique de récupération de profondeur"""
        # À implémenter pour chaque venue
        pass
    
    async def submit_order(self, sub_order: SubOrder) -> Dict[str, Any]:
        """Soumet un ordre à la venue"""
        start_time = time.time()
        
        try:
            async with self.rate_limiter:
                result = await self._submit_order_impl(sub_order)
                
                # Enregistrer les métriques
                execution_latency = (time.time() - start_time) * 1000  # ms
                self.execution_latencies.append(execution_latency)
                
                return result
                
        except Exception as e:
            self.logger.error(f"Erreur soumission ordre {sub_order.sub_order_id}: {e}")
            raise
    
    async def _submit_order_impl(self, sub_order: SubOrder) -> Dict[str, Any]:
        """Implémentation spécifique de soumission d'ordre"""
        # À implémenter pour chaque venue
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Retourne les métriques de performance"""
        return {
            'avg_latency_ms': np.mean(self.execution_latencies) if self.execution_latencies else 0,
            'p95_latency_ms': np.percentile(self.execution_latencies, 95) if self.execution_latencies else 0,
            'avg_fill_rate': np.mean(self.fill_rates) if self.fill_rates else 0,
            'connection_uptime': 1.0 - min(self.connection_errors / 100, 1.0),
            'avg_execution_cost_bps': np.mean(self.execution_costs) if self.execution_costs else 0
        }


class SmartRoutingEngine:
    """Moteur principal de routage intelligent"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or ROUTING_CONFIG
        self.logger = get_logger(f"{__name__}.SmartRoutingEngine")
        
        # Venues disponibles
        self.venues: Dict[VenueId, VenueInfo] = {}
        self.connectors: Dict[VenueId, VenueConnector] = {}
        
        # Modèles ML pour l'optimisation
        self.cost_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.latency_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.ml_models_trained = False
        
        # Cache et historique
        self.routing_history = deque(maxlen=10000)
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))
        
        # Slippage model
        self.slippage_model = SlippageModel()
        
        # Performance tracker
        self.performance_tracker = PerformanceTracker()
        
        # État des venues
        self.venue_health_scores: Dict[VenueId, float] = defaultdict(float)
        self.venue_last_update: Dict[VenueId, Timestamp] = {}
        
        self._initialize_venues()
    
    def _initialize_venues(self):
        """Initialise les venues configurées"""
        venue_configs = self.config.get('venues', {})
        
        for venue_id, venue_config in venue_configs.items():
            venue_info = VenueInfo(
                venue_id=venue_id,
                name=venue_config['name'],
                venue_type=VenueType(venue_config['type']),
                supported_symbols=venue_config['symbols'],
                min_order_size=Quantity(str(venue_config['min_order_size'])),
                max_order_size=Quantity(str(venue_config['max_order_size'])),
                tick_size=Decimal(str(venue_config['tick_size'])),
                maker_fee=Decimal(str(venue_config['maker_fee'])),
                taker_fee=Decimal(str(venue_config['taker_fee'])),
                api_latency_ms=venue_config['api_latency_ms'],
                connectivity_score=venue_config.get('connectivity_score', 0.9),
                reliability_score=venue_config.get('reliability_score', 0.95),
                average_fill_rate=venue_config.get('average_fill_rate', 0.98),
                supported_order_types=venue_config['order_types'],
                trading_hours=venue_config.get('trading_hours', {})
            )
            
            self.venues[venue_id] = venue_info
            self.connectors[venue_id] = VenueConnector(venue_info, venue_config)
            self.venue_health_scores[venue_id] = venue_info.total_score
        
        self.logger.info(f"Initialisé {len(self.venues)} venues: {list(self.venues.keys())}")
    
    async def initialize(self):
        """Initialise le moteur de routage"""
        # Connecter aux venues
        connection_tasks = []
        for connector in self.connectors.values():
            connection_tasks.append(connector.connect())
        
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        connected_venues = []
        for venue_id, result in zip(self.connectors.keys(), results):
            if isinstance(result, bool) and result:
                connected_venues.append(venue_id)
            else:
                self.logger.warning(f"Échec connexion {venue_id}: {result}")
        
        self.logger.info(f"Connecté à {len(connected_venues)} venues: {connected_venues}")
        
        # Démarrer les tâches de monitoring
        asyncio.create_task(self._monitor_venue_health())
        asyncio.create_task(self._update_market_data())
    
    async def route_order(self, order: Order) -> RoutingDecision:
        """Route un ordre vers les meilleures venues"""
        start_time = time.time()
        correlation_id = create_correlation_id()
        
        try:
            self.logger.info(f"Routage ordre {order.order_id}: {order.symbol} {order.side.value} {order.quantity}")
            
            # Filtrer les venues applicables
            eligible_venues = await self._filter_eligible_venues(order)
            
            if not eligible_venues:
                raise ValueError("Aucune venue éligible trouvée")
            
            # Récupérer les données de marché
            market_data = await self._gather_market_data(order.symbol, eligible_venues)
            
            # Sélectionner l'algorithme de routage
            algorithm = self._select_routing_algorithm(order, market_data)
            
            # Générer la décision de routage
            routing_decision = await self._generate_routing_decision(
                order, eligible_venues, market_data, algorithm, correlation_id
            )
            
            # Enregistrer pour l'apprentissage
            self.routing_history.append({
                'timestamp': datetime.now(timezone.utc),
                'order': order,
                'decision': routing_decision,
                'market_conditions': market_data,
                'processing_time_ms': (time.time() - start_time) * 1000
            })
            
            self.logger.info(
                f"Ordre routé: {len(routing_decision.sub_orders)} sous-ordres, "
                f"coût estimé: {routing_decision.estimated_cost_bps:.2f} bps, "
                f"algorithme: {algorithm.value}"
            )
            
            return routing_decision
            
        except Exception as e:
            self.logger.error(f"Erreur routage ordre {order.order_id}: {e}")
            raise
    
    async def _filter_eligible_venues(self, order: Order) -> List[VenueId]:
        """Filtre les venues éligibles pour un ordre"""
        eligible = []
        
        for venue_id, venue_info in self.venues.items():
            # Vérifier le symbole supporté
            if order.symbol not in venue_info.supported_symbols:
                continue
            
            # Vérifier la taille d'ordre
            if order.quantity < venue_info.min_order_size or order.quantity > venue_info.max_order_size:
                continue
            
            # Vérifier les venues préférées/exclues
            if order.preferred_venues and venue_id not in order.preferred_venues:
                continue
            
            if venue_id in order.excluded_venues:
                continue
            
            # Vérifier l'état de la connexion
            if not self.connectors[venue_id].is_connected:
                continue
            
            # Vérifier le score de santé
            if self.venue_health_scores[venue_id] < self.config.get('min_health_score', 0.7):
                continue
            
            eligible.append(venue_id)
        
        return eligible
    
    async def _gather_market_data(
        self, 
        symbol: Symbol, 
        venue_ids: List[VenueId]
    ) -> Dict[VenueId, MarketDepth]:
        """Récupère les données de marché de toutes les venues"""
        tasks = []
        for venue_id in venue_ids:
            connector = self.connectors[venue_id]
            tasks.append(connector.get_market_depth(symbol))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        market_data = {}
        for venue_id, result in zip(venue_ids, results):
            if isinstance(result, MarketDepth):
                market_data[venue_id] = result
            else:
                self.logger.warning(f"Pas de données marché pour {venue_id}: {result}")
        
        return market_data
    
    def _select_routing_algorithm(
        self, 
        order: Order, 
        market_data: Dict[VenueId, MarketDepth]
    ) -> RoutingAlgorithm:
        """Sélectionne l'algorithme de routage optimal"""
        
        # Facteurs de décision
        urgency_factor = {
            ExecutionUrgency.IMMEDIATE: 1.0,
            ExecutionUrgency.HIGH: 0.8,
            ExecutionUrgency.NORMAL: 0.5,
            ExecutionUrgency.LOW: 0.2
        }[order.urgency]
        
        # Taille relative de l'ordre
        total_liquidity = sum(
            depth.get_available_liquidity(order.side) 
            for depth in market_data.values()
        )
        
        order_size_ratio = float(order.quantity / total_liquidity) if total_liquidity > 0 else 1.0
        
        # Sélection basée sur les conditions
        if urgency_factor >= 0.8:
            return RoutingAlgorithm.BEST_PRICE
        
        elif order_size_ratio > 0.1:  # Ordre volumineux (>10% liquidité)
            if order.urgency == ExecutionUrgency.LOW:
                return RoutingAlgorithm.TWAP
            else:
                return RoutingAlgorithm.ICEBERG
        
        elif len(market_data) > 3 and self.ml_models_trained:
            return RoutingAlgorithm.SMART_ADAPTIVE
        
        else:
            return RoutingAlgorithm.LOWEST_COST
    
    async def _generate_routing_decision(
        self,
        order: Order,
        eligible_venues: List[VenueId],
        market_data: Dict[VenueId, MarketDepth],
        algorithm: RoutingAlgorithm,
        correlation_id: str
    ) -> RoutingDecision:
        """Génère la décision de routage finale"""
        
        # Déléguer à l'algorithme spécifique
        if algorithm == RoutingAlgorithm.BEST_PRICE:
            return await self._route_best_price(order, market_data, correlation_id)
        
        elif algorithm == RoutingAlgorithm.BEST_LIQUIDITY:
            return await self._route_best_liquidity(order, market_data, correlation_id)
        
        elif algorithm == RoutingAlgorithm.LOWEST_COST:
            return await self._route_lowest_cost(order, market_data, correlation_id)
        
        elif algorithm == RoutingAlgorithm.SMART_ADAPTIVE:
            return await self._route_smart_adaptive(order, market_data, correlation_id)
        
        elif algorithm == RoutingAlgorithm.TWAP:
            return await self._route_twap(order, market_data, correlation_id)
        
        elif algorithm == RoutingAlgorithm.ICEBERG:
            return await self._route_iceberg(order, market_data, correlation_id)
        
        else:
            # Fallback vers best price
            return await self._route_best_price(order, market_data, correlation_id)
    
    async def _route_best_price(
        self,
        order: Order,
        market_data: Dict[VenueId, MarketDepth],
        correlation_id: str
    ) -> RoutingDecision:
        """Routage vers le meilleur prix disponible"""
        
        best_venues = []
        
        for venue_id, depth in market_data.items():
            if order.side == OrderSide.BUY and depth.best_ask:
                price, available_qty = depth.best_ask
                best_venues.append((venue_id, price, available_qty, depth))
            elif order.side == OrderSide.SELL and depth.best_bid:
                price, available_qty = depth.best_bid
                best_venues.append((venue_id, price, available_qty, depth))
        
        # Trier par prix (meilleur d'abord)
        if order.side == OrderSide.BUY:
            best_venues.sort(key=lambda x: x[1])  # Prix croissant pour achats
        else:
            best_venues.sort(key=lambda x: x[1], reverse=True)  # Prix décroissant pour ventes
        
        # Créer les sous-ordres
        sub_orders = []
        remaining_quantity = order.quantity
        priority = 0
        
        for venue_id, price, available_qty, depth in best_venues:
            if remaining_quantity <= 0:
                break
            
            # Quantité à exécuter sur cette venue
            execute_qty = min(remaining_quantity, available_qty)
            execute_qty = min(execute_qty, self.venues[venue_id].max_order_size)
            
            if execute_qty < self.venues[venue_id].min_order_size:
                continue
            
            # Estimer les coûts
            venue_info = self.venues[venue_id]
            fee_rate = venue_info.taker_fee  # Market order = taker
            estimated_cost = execute_qty * price * fee_rate
            
            # Créer le sous-ordre
            sub_order = SubOrder(
                sub_order_id=f"{order.order_id}_{venue_id}_{priority}",
                parent_correlation_id=correlation_id,
                venue_id=venue_id,
                symbol=order.symbol,
                side=order.side,
                quantity=execute_qty,
                price=None,  # Market order
                order_type='MARKET',
                estimated_cost=estimated_cost,
                estimated_slippage_bps=0.0,  # Market order = pas de slippage prix
                priority=priority
            )
            
            sub_orders.append(sub_order)
            remaining_quantity -= execute_qty
            priority += 1
        
        # Calculer les métriques globales
        total_cost = sum(sub.estimated_cost for sub in sub_orders)
        avg_price = safe_divide(total_cost, order.quantity - remaining_quantity, 0)
        
        return RoutingDecision(
            correlation_id=correlation_id,
            original_order=order,
            sub_orders=sub_orders,
            algorithm_used=RoutingAlgorithm.BEST_PRICE,
            estimated_cost_bps=float(total_cost / (order.quantity * avg_price) * 10000) if avg_price > 0 else 0,
            estimated_slippage_bps=0.0,
            estimated_execution_time_ms=sum(self.venues[sub.venue_id].api_latency_ms for sub in sub_orders),
            reasoning={
                'algorithm': 'best_price',
                'venues_considered': len(market_data),
                'venues_selected': len(sub_orders),
                'remaining_quantity': float(remaining_quantity)
            },
            confidence_score=0.9
        )
    
    async def _route_lowest_cost(
        self,
        order: Order,
        market_data: Dict[VenueId, MarketDepth],
        correlation_id: str
    ) -> RoutingDecision:
        """Routage pour minimiser le coût total"""
        
        venue_costs = []
        
        for venue_id, depth in market_data.items():
            venue_info = self.venues[venue_id]
            
            if order.side == OrderSide.BUY and depth.best_ask:
                price, available_qty = depth.best_ask
            elif order.side == OrderSide.SELL and depth.best_bid:
                price, available_qty = depth.best_bid
            else:
                continue
            
            # Calculer le coût total (prix + frais + slippage estimé)
            fee_cost = price * venue_info.taker_fee
            
            # Estimer le slippage avec notre modèle
            market_conditions = await self._create_market_conditions(depth)
            slippage_prediction = await self.slippage_model.predict_slippage(
                order.symbol, 
                min(order.quantity, available_qty),
                order.side,
                market_conditions
            )
            slippage_cost = price * (slippage_prediction.expected_slippage_bps / 10000)
            
            total_cost_per_unit = price + fee_cost + slippage_cost
            
            venue_costs.append((
                venue_id, price, available_qty, total_cost_per_unit, 
                slippage_prediction.expected_slippage_bps, depth
            ))
        
        # Trier par coût total croissant
        venue_costs.sort(key=lambda x: x[3])
        
        # Créer les sous-ordres
        sub_orders = []
        remaining_quantity = order.quantity
        priority = 0
        
        for venue_id, price, available_qty, cost_per_unit, slippage_bps, depth in venue_costs:
            if remaining_quantity <= 0:
                break
            
            execute_qty = min(remaining_quantity, available_qty)
            execute_qty = min(execute_qty, self.venues[venue_id].max_order_size)
            
            if execute_qty < self.venues[venue_id].min_order_size:
                continue
            
            estimated_cost = execute_qty * cost_per_unit
            
            sub_order = SubOrder(
                sub_order_id=f"{order.order_id}_{venue_id}_{priority}",
                parent_correlation_id=correlation_id,
                venue_id=venue_id,
                symbol=order.symbol,
                side=order.side,
                quantity=execute_qty,
                price=None,
                order_type='MARKET',
                estimated_cost=estimated_cost,
                estimated_slippage_bps=slippage_bps,
                priority=priority
            )
            
            sub_orders.append(sub_order)
            remaining_quantity -= execute_qty
            priority += 1
        
        total_cost = sum(sub.estimated_cost for sub in sub_orders)
        avg_slippage = np.mean([sub.estimated_slippage_bps for sub in sub_orders]) if sub_orders else 0
        
        return RoutingDecision(
            correlation_id=correlation_id,
            original_order=order,
            sub_orders=sub_orders,
            algorithm_used=RoutingAlgorithm.LOWEST_COST,
            estimated_cost_bps=float(total_cost / (order.quantity * price) * 10000) if price > 0 else 0,
            estimated_slippage_bps=avg_slippage,
            estimated_execution_time_ms=sum(self.venues[sub.venue_id].api_latency_ms for sub in sub_orders),
            reasoning={
                'algorithm': 'lowest_cost',
                'cost_optimization': True,
                'venues_analyzed': len(venue_costs)
            },
            confidence_score=0.85
        )
    
    async def _route_twap(
        self,
        order: Order,
        market_data: Dict[VenueId, MarketDepth],
        correlation_id: str
    ) -> RoutingDecision:
        """Routage TWAP (Time-Weighted Average Price)"""
        
        # Paramètres TWAP
        execution_horizon_minutes = self.config.get('twap_horizon_minutes', 30)
        slice_interval_minutes = self.config.get('twap_slice_interval', 2)
        num_slices = execution_horizon_minutes // slice_interval_minutes
        
        slice_quantity = order.quantity / num_slices
        
        # Sélectionner les meilleures venues
        best_venues = await self._select_best_venues_for_twap(order, market_data)
        
        # Créer les sous-ordres avec timing
        sub_orders = []
        priority = 0
        
        for slice_num in range(num_slices):
            execution_delay_ms = slice_num * slice_interval_minutes * 60 * 1000
            
            # Distribuer la quantité entre les venues
            venues_for_slice = best_venues[:min(3, len(best_venues))]  # Max 3 venues par slice
            qty_per_venue = slice_quantity / len(venues_for_slice)
            
            for venue_id, _ in venues_for_slice:
                venue_info = self.venues[venue_id]
                
                if qty_per_venue >= venue_info.min_order_size:
                    sub_order = SubOrder(
                        sub_order_id=f"{order.order_id}_{venue_id}_{slice_num}",
                        parent_correlation_id=correlation_id,
                        venue_id=venue_id,
                        symbol=order.symbol,
                        side=order.side,
                        quantity=qty_per_venue,
                        price=None,
                        order_type='MARKET',
                        estimated_cost=qty_per_venue * venue_info.taker_fee,
                        estimated_slippage_bps=2.0,  # Slippage réduit par la dispersion temporelle
                        priority=priority,
                        execution_delay_ms=execution_delay_ms
                    )
                    
                    sub_orders.append(sub_order)
                    priority += 1
        
        return RoutingDecision(
            correlation_id=correlation_id,
            original_order=order,
            sub_orders=sub_orders,
            algorithm_used=RoutingAlgorithm.TWAP,
            estimated_cost_bps=2.0,  # Coût réduit par la dispersion
            estimated_slippage_bps=2.0,
            estimated_execution_time_ms=execution_horizon_minutes * 60 * 1000,
            reasoning={
                'algorithm': 'twap',
                'execution_horizon_minutes': execution_horizon_minutes,
                'num_slices': num_slices,
                'slice_interval_minutes': slice_interval_minutes
            },
            confidence_score=0.8
        )
    
    async def _select_best_venues_for_twap(
        self,
        order: Order,
        market_data: Dict[VenueId, MarketDepth]
    ) -> List[Tuple[VenueId, float]]:
        """Sélectionne les meilleures venues pour TWAP"""
        
        venue_scores = []
        
        for venue_id, depth in market_data.items():
            venue_info = self.venues[venue_id]
            
            # Score basé sur liquidité, frais et fiabilité
            liquidity_score = float(depth.get_available_liquidity(order.side) / order.quantity)
            cost_score = 1.0 - float(venue_info.taker_fee)
            reliability_score = venue_info.reliability_score
            
            total_score = (liquidity_score * 0.4 + 
                          cost_score * 0.3 + 
                          reliability_score * 0.3)
            
            venue_scores.append((venue_id, total_score))
        
        # Trier par score décroissant
        venue_scores.sort(key=lambda x: x[1], reverse=True)
        
        return venue_scores
    
    async def _create_market_conditions(self, depth: MarketDepth) -> MarketConditions:
        """Crée des conditions de marché à partir de la profondeur"""
        # Simplifié - dans un vrai système, utiliser des données plus complètes
        mid_price = depth.mid_price or depth.last_price
        spread = depth.spread or Decimal('0.01')
        
        return MarketConditions(
            symbol=depth.symbol,
            timestamp=depth.timestamp,
            bid_price=depth.best_bid[0] if depth.best_bid else mid_price,
            ask_price=depth.best_ask[0] if depth.best_ask else mid_price,
            bid_size=depth.best_bid[1] if depth.best_bid else Quantity('1000'),
            ask_size=depth.best_ask[1] if depth.best_ask else Quantity('1000'),
            mid_price=mid_price,
            spread_bps=float(spread / mid_price * 10000) if mid_price > 0 else 0,
            volatility_1min=0.01,  # À remplacer par calcul réel
            volatility_5min=0.008,
            volume_1min=Quantity('10000'),
            volume_avg_daily=depth.volume_24h,
            market_regime=MarketRegime.NORMAL,  # À déterminer dynamiquement
            time_of_day='session'
        )
    
    async def _monitor_venue_health(self):
        """Monitore la santé des venues en continu"""
        while True:
            try:
                for venue_id, connector in self.connectors.items():
                    if not connector.is_connected:
                        # Tentative de reconnexion
                        await connector.connect()
                    
                    # Mettre à jour le score de santé
                    metrics = connector.get_performance_metrics()
                    
                    # Score basé sur latence, uptime et fill rate
                    latency_score = max(0, 1 - metrics['avg_latency_ms'] / 1000)  # Pénalité si >1s
                    uptime_score = metrics['connection_uptime']
                    fill_score = metrics['avg_fill_rate']
                    
                    health_score = (latency_score * 0.3 + 
                                   uptime_score * 0.4 + 
                                   fill_score * 0.3)
                    
                    self.venue_health_scores[venue_id] = health_score
                    self.venue_last_update[venue_id] = datetime.now(timezone.utc)
                
                await asyncio.sleep(30)  # Vérification toutes les 30s
                
            except Exception as e:
                self.logger.error(f"Erreur monitoring venues: {e}")
                await asyncio.sleep(60)
    
    async def _update_market_data(self):
        """Met à jour les données de marché en continu"""
        while True:
            try:
                # Mettre à jour pour tous les symboles actifs
                active_symbols = set()
                for venue_info in self.venues.values():
                    active_symbols.update(venue_info.supported_symbols)
                
                # Limiter aux symboles les plus tradés
                priority_symbols = list(active_symbols)[:10]  # Top 10
                
                for symbol in priority_symbols:
                    tasks = []
                    for connector in self.connectors.values():
                        if symbol in connector.venue_info.supported_symbols:
                            tasks.append(connector.get_market_depth(symbol, force_refresh=True))
                    
                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(1)  # Mise à jour chaque seconde
                
            except Exception as e:
                self.logger.error(f"Erreur update market data: {e}")
                await asyncio.sleep(5)
    
    def get_routing_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques de routage"""
        
        if not self.routing_history:
            return {"error": "Pas de données de routage"}
        
        recent_decisions = list(self.routing_history)[-100:]  # 100 dernières
        
        # Statistiques des algorithmes
        algorithm_usage = defaultdict(int)
        avg_costs = defaultdict(list)
        avg_execution_times = defaultdict(list)
        
        for entry in recent_decisions:
            decision = entry['decision']
            algo = decision.algorithm_used.value
            
            algorithm_usage[algo] += 1
            avg_costs[algo].append(decision.estimated_cost_bps)
            avg_execution_times[algo].append(decision.estimated_execution_time_ms)
        
        # Statistiques des venues
        venue_usage = defaultdict(int)
        venue_performance = defaultdict(list)
        
        for entry in recent_decisions:
            for sub_order in entry['decision'].sub_orders:
                venue_id = sub_order.venue_id
                venue_usage[venue_id] += 1
                venue_performance[venue_id].append(sub_order.estimated_cost)
        
        return {
            'total_orders_routed': len(self.routing_history),
            'recent_orders_analyzed': len(recent_decisions),
            'algorithm_usage': dict(algorithm_usage),
            'avg_cost_by_algorithm': {
                algo: np.mean(costs) for algo, costs in avg_costs.items()
            },
            'avg_execution_time_by_algorithm': {
                algo: np.mean(times) for algo, times in avg_execution_times.items()
            },
            'venue_usage': dict(venue_usage),
            'venue_health_scores': dict(self.venue_health_scores),
            'connected_venues': [
                venue_id for venue_id, connector in self.connectors.items() 
                if connector.is_connected
            ]
        }


# Exemple d'utilisation
if __name__ == "__main__":
    async def main():
        # Configuration exemple
        config = {
            'venues': {
                'binance': {
                    'name': 'Binance',
                    'type': 'exchange',
                    'symbols': ['BTCUSDT', 'ETHUSDT'],
                    'min_order_size': 0.001,
                    'max_order_size': 1000,
                    'tick_size': 0.01,
                    'maker_fee': 0.001,
                    'taker_fee': 0.001,
                    'api_latency_ms': 50,
                    'order_types': ['MARKET', 'LIMIT']
                },
                'coinbase': {
                    'name': 'Coinbase Pro',
                    'type': 'exchange', 
                    'symbols': ['BTCUSDT', 'ETHUSDT'],
                    'min_order_size': 0.001,
                    'max_order_size': 500,
                    'tick_size': 0.01,
                    'maker_fee': 0.005,
                    'taker_fee': 0.005,
                    'api_latency_ms': 80,
                    'order_types': ['MARKET', 'LIMIT']
                }
            }
        }
        
        # Créer le moteur de routage
        router = SmartRoutingEngine(config)
        await router.initialize()
        
        # Créer un ordre exemple
        order = Order(
            order_id='test_001',
            symbol='BTCUSDT',
            side=OrderSide.BUY,
            quantity=Quantity('1.5'),
            order_type='MARKET',
            urgency=ExecutionUrgency.NORMAL,
            max_slippage_bps=10.0,
            time_in_force='IOC',
            client_id='client_1',
            strategy_id='arbitrage_1'
        )
        
        # Router l'ordre
        try:
            decision = await router.route_order(order)
            
            print(f"Ordre routé avec succès!")
            print(f"Algorithme utilisé: {decision.algorithm_used.value}")
            print(f"Nombre de sous-ordres: {len(decision.sub_orders)}")
            print(f"Coût estimé: {decision.estimated_cost_bps:.2f} bps")
            print(f"Temps d'exécution estimé: {decision.estimated_execution_time_ms} ms")
            
            for i, sub_order in enumerate(decision.sub_orders):
                print(f"  Sous-ordre {i+1}: {sub_order.venue_id} - {sub_order.quantity} @ {sub_order.order_type}")
            
        except Exception as e:
            print(f"Erreur: {e}")
        
        # Afficher les statistiques
        stats = router.get_routing_statistics()
        print(f"\nStatistiques: {stats}")
    
    # Exécution
    asyncio.run(main())