"""
Order Execution Package
======================

Système d'exécution d'ordres haute performance avec smart routing,
gestion du slippage et optimisation des coûts.

Composants:
    - OrderManager: Gestion du cycle de vie des ordres
    - ExecutionEngine: Moteur d'exécution principal
    - SmartRouter: Routage intelligent multi-venues
    - SlippageModel: Modélisation et prédiction du slippage

Usage:
    from execution import ExecutionEngine, OrderManager, SmartRouter
    
    engine = ExecutionEngine(config)
    order = await engine.execute_order(signal, sizing)
"""

from execution.order_manager import (
    OrderManager,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    OrderUpdate,
    OrderBook,
    OrderTracker
)

from execution.execution_engine import (
    ExecutionEngine,
    ExecutionAlgorithm,
    ExecutionMetrics,
    FillReport,
    ExecutionConfig,
    TWAP,
    VWAP,
    POV,
    ImplementationShortfall
)

from execution.smart_routing import (
    SmartRouter,
    RoutingAlgorithm,
    VenueSelector,
    CostModel,
    LatencyModel,
    LiquidityAggregator,
    RoutingDecision
)

from execution.slippage_model import (
    SlippageModel,
    SlippageEstimator,
    MarketImpactModel,
    TemporaryImpact,
    PermanentImpact,
    SpreadCost,
    SlippageMetrics
)

# Version
__version__ = "1.0.0"

# Exports publics
__all__ = [
    # Order Manager
    "OrderManager",
    "Order",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "TimeInForce",
    "OrderUpdate",
    "OrderBook",
    "OrderTracker",
    
    # Execution Engine
    "ExecutionEngine",
    "ExecutionAlgorithm",
    "ExecutionMetrics",
    "FillReport",
    "ExecutionConfig",
    "TWAP",
    "VWAP",
    "POV",
    "ImplementationShortfall",
    
    # Smart Routing
    "SmartRouter",
    "RoutingAlgorithm",
    "VenueSelector",
    "CostModel",
    "LatencyModel",
    "LiquidityAggregator",
    "RoutingDecision",
    
    # Slippage Model
    "SlippageModel",
    "SlippageEstimator",
    "MarketImpactModel",
    "TemporaryImpact",
    "PermanentImpact",
    "SpreadCost",
    "SlippageMetrics"
]

# Configuration par défaut
DEFAULT_EXECUTION_CONFIG = {
    "engine": {
        "max_order_size": 1000000,  # USD
        "min_order_size": 10,
        "max_orders_per_second": 100,
        "order_timeout": 300,  # seconds
        "retry_attempts": 3,
        "retry_delay": 1,
        "execution_algorithms": ["TWAP", "VWAP", "POV", "IS"]
    },
    "routing": {
        "venues": ["binance", "coinbase", "kraken"],
        "routing_algorithm": "smart",
        "cost_priority": 0.4,
        "speed_priority": 0.3,
        "liquidity_priority": 0.3,
        "max_venues_per_order": 3,
        "min_venue_size": 100  # USD
    },
    "slippage": {
        "model": "linear",
        "temporary_impact_halflife": 300,  # seconds
        "permanent_impact_factor": 0.1,
        "spread_buffer": 0.0001,  # 1 bps
        "urgency_multiplier": {
            "low": 0.5,
            "medium": 1.0,
            "high": 2.0,
            "critical": 5.0
        }
    },
    "fees": {
        "maker_fee": 0.001,  # 0.1%
        "taker_fee": 0.001,  # 0.1%
        "network_fee": 0.0001  # Fixed
    }
}

# Types d'algorithmes d'exécution
class AlgorithmType:
    """Types d'algorithmes d'exécution disponibles"""
    MARKET = "market"              # Ordre au marché immédiat
    LIMIT = "limit"                # Ordre limite
    TWAP = "twap"                  # Time-Weighted Average Price
    VWAP = "vwap"                  # Volume-Weighted Average Price
    POV = "pov"                    # Percentage of Volume
    IS = "implementation_shortfall" # Minimiser l'écart d'implémentation
    ICEBERG = "iceberg"            # Ordre iceberg
    SNIPER = "sniper"              # Snipe les liquidités
    
    @classmethod
    def all(cls):
        return [
            cls.MARKET, cls.LIMIT, cls.TWAP, cls.VWAP,
            cls.POV, cls.IS, cls.ICEBERG, cls.SNIPER
        ]


# Structure d'une instruction d'exécution
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, Any, List


@dataclass
class ExecutionInstruction:
    """Instruction complète pour l'exécution d'un ordre"""
    symbol: str
    side: OrderSide
    quantity: Decimal
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    algorithm: str = AlgorithmType.MARKET
    urgency: str = "medium"  # low, medium, high, critical
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    min_size: Optional[Decimal] = None
    max_show_size: Optional[Decimal] = None
    venues: Optional[List[str]] = None
    avoid_venues: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Valide l'instruction"""
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
            
        if self.order_type == OrderType.LIMIT and not self.limit_price:
            raise ValueError("Limit orders require limit_price")
            
        if self.order_type == OrderType.STOP and not self.stop_price:
            raise ValueError("Stop orders require stop_price")
            
        if self.algorithm in [AlgorithmType.TWAP, AlgorithmType.VWAP]:
            if not self.start_time or not self.end_time:
                raise ValueError(f"{self.algorithm} requires start_time and end_time")
                
        return True


# Gestionnaire d'exécution unifié
class UnifiedExecutionManager:
    """
    Gestionnaire unifié pour toutes les opérations d'exécution.
    Coordonne OrderManager, ExecutionEngine et SmartRouter.
    """
    
    def __init__(self, config: dict = None):
        self.config = {**DEFAULT_EXECUTION_CONFIG, **(config or {})}
        
        # Initialiser les composants
        self.order_manager = OrderManager(self.config)
        self.execution_engine = ExecutionEngine(self.config['engine'])
        self.smart_router = SmartRouter(self.config['routing'])
        self.slippage_model = SlippageModel(self.config['slippage'])
        
        self._running = False
        self._metrics = ExecutionMetrics()
        
    async def start(self):
        """Démarre tous les composants"""
        await self.order_manager.start()
        await self.execution_engine.start()
        await self.smart_router.start()
        self._running = True
        
    async def stop(self):
        """Arrête tous les composants"""
        self._running = False
        await self.execution_engine.stop()
        await self.smart_router.stop()
        await self.order_manager.stop()
        
    async def execute(self, instruction: ExecutionInstruction) -> Order:
        """
        Exécute une instruction de trading.
        
        Args:
            instruction: Instruction d'exécution
            
        Returns:
            Order exécuté
        """
        # Valider l'instruction
        instruction.validate()
        
        # Estimer le slippage
        slippage_estimate = await self.slippage_model.estimate(
            symbol=instruction.symbol,
            side=instruction.side,
            quantity=instruction.quantity,
            urgency=instruction.urgency
        )
        
        # Router l'ordre
        routing_decision = await self.smart_router.route(
            instruction=instruction,
            slippage_estimate=slippage_estimate
        )
        
        # Exécuter via l'engine
        order = await self.execution_engine.execute(
            instruction=instruction,
            routing=routing_decision,
            slippage_estimate=slippage_estimate
        )
        
        # Tracker l'ordre
        await self.order_manager.track_order(order)
        
        return order
        
    def get_metrics(self) -> ExecutionMetrics:
        """Retourne les métriques d'exécution"""
        return self._metrics


# Instance globale optionnelle
_execution_manager = None


def get_execution_manager(config: dict = None) -> UnifiedExecutionManager:
    """Obtient l'instance globale du gestionnaire d'exécution"""
    global _execution_manager
    if _execution_manager is None:
        _execution_manager = UnifiedExecutionManager(config)
    return _execution_manager


# Helpers pour exécution rapide
async def execute_market_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    urgency: str = "medium"
) -> Order:
    """Exécute rapidement un ordre au marché"""
    manager = get_execution_manager()
    
    instruction = ExecutionInstruction(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=OrderType.MARKET,
        algorithm=AlgorithmType.MARKET,
        urgency=urgency
    )
    
    return await manager.execute(instruction)


async def execute_twap_order(
    symbol: str,
    side: OrderSide,
    quantity: Decimal,
    duration_minutes: int,
    urgency: str = "low"
) -> Order:
    """Exécute un ordre TWAP"""
    manager = get_execution_manager()
    
    now = datetime.now(timezone.utc)
    instruction = ExecutionInstruction(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        algorithm=AlgorithmType.TWAP,
        urgency=urgency,
        start_time=now,
        end_time=now + timedelta(minutes=duration_minutes)
    )
    
    return await manager.execute(instruction)


# Calculs de coûts d'exécution
def calculate_implementation_shortfall(
    decision_price: Decimal,
    execution_price: Decimal,
    quantity: Decimal,
    side: OrderSide
) -> Decimal:
    """Calcule l'implementation shortfall"""
    if side == OrderSide.BUY:
        return (execution_price - decision_price) * quantity
    else:
        return (decision_price - execution_price) * quantity


def calculate_effective_spread(
    execution_price: Decimal,
    mid_price: Decimal,
    quantity: Decimal,
    side: OrderSide
) -> Decimal:
    """Calcule le spread effectif"""
    if side == OrderSide.BUY:
        return 2 * (execution_price - mid_price) / mid_price
    else:
        return 2 * (mid_price - execution_price) / mid_price


# Imports temporels
from datetime import timedelta

# Initialisation
import logging
logger = logging.getLogger(__name__)
logger.info(f"Execution package initialized - Version {__version__}")