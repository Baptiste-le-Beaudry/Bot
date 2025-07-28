"""
Core Trading Engine Package
==========================

Module central orchestrant tous les composants du robot de trading.
Fournit le moteur principal, la sélection de stratégies et la gestion
de portefeuille.

Classes principales:
    - TradingEngine: Moteur de trading principal event-driven
    - StrategySelector: Sélecteur dynamique de stratégies avec hot-swapping
    - PortfolioManager: Gestionnaire de portefeuille temps réel

Usage:
    from core import TradingEngine, StrategySelector, PortfolioManager
    
    engine = TradingEngine(config)
    await engine.start()
"""

from core.engine import (
    TradingEngine,
    EngineState,
    EventType,
    Event,
    EventBus,
    SystemComponent,
    EngineMetrics
)

from core.strategy_selector import (
    StrategySelector,
    StrategyRegistry,
    StrategyMetadata,
    SelectionCriteria,
    StrategyPerformance,
    RegimeType
)

from core.portfolio_manager import (
    PortfolioManager,
    Portfolio,
    Position,
    Asset,
    PositionType,
    PortfolioMetrics,
    AllocationStrategy,
    RebalancingTrigger
)

# Version du package
__version__ = "1.0.0"

# Exports publics
__all__ = [
    # Engine
    "TradingEngine",
    "EngineState",
    "EventType",
    "Event",
    "EventBus",
    "SystemComponent",
    "EngineMetrics",
    
    # Strategy Selector
    "StrategySelector",
    "StrategyRegistry",
    "StrategyMetadata",
    "SelectionCriteria",
    "StrategyPerformance",
    "RegimeType",
    
    # Portfolio Manager
    "PortfolioManager",
    "Portfolio",
    "Position",
    "Asset",
    "PositionType",
    "PortfolioMetrics",
    "AllocationStrategy",
    "RebalancingTrigger"
]

# Configuration par défaut
DEFAULT_ENGINE_CONFIG = {
    "event_queue_size": 10000,
    "max_events_per_second": 1000,
    "component_timeout": 30,
    "health_check_interval": 5,
    "metrics_collection_interval": 1
}

DEFAULT_PORTFOLIO_CONFIG = {
    "max_positions": 100,
    "max_position_size": 0.1,  # 10% max par position
    "rebalancing_threshold": 0.05,  # 5% deviation
    "min_trade_size": 10,  # USD
    "settlement_delay": 0  # T+0 pour crypto
}


def create_engine(config: dict = None) -> TradingEngine:
    """
    Factory function pour créer un moteur de trading configuré.
    
    Args:
        config: Configuration optionnelle (utilise les défauts sinon)
        
    Returns:
        Instance configurée de TradingEngine
    """
    merged_config = {**DEFAULT_ENGINE_CONFIG, **(config or {})}
    return TradingEngine(merged_config)


def create_portfolio_manager(config: dict = None) -> PortfolioManager:
    """
    Factory function pour créer un gestionnaire de portefeuille.
    
    Args:
        config: Configuration optionnelle
        
    Returns:
        Instance configurée de PortfolioManager
    """
    merged_config = {**DEFAULT_PORTFOLIO_CONFIG, **(config or {})}
    return PortfolioManager(merged_config)


# Initialisation au chargement du module
def _initialize_core():
    """Initialisation du module core"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Core module initialized - Version {__version__}")


_initialize_core()