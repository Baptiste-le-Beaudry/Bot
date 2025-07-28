"""
Trading Strategies Package
=========================

Collection de stratégies de trading algorithmique optimisées par IA.
Chaque stratégie hérite de BaseStrategy et peut être hot-swappée.

Stratégies disponibles:
    - StatisticalArbitrage: Arbitrage statistique avec pairs trading
    - MarketMaking: Market making dynamique avec gestion d'inventaire
    - Scalping: Scalping haute fréquence sub-seconde
    - EnsembleStrategy: Combinaison intelligente de stratégies

Usage:
    from strategies import StatisticalArbitrage, MarketMaking
    
    strategy = StatisticalArbitrage(config)
    signal = await strategy.generate_signal(market_data)
"""

from strategies.base_strategy import (
    BaseStrategy,
    StrategyState,
    Signal,
    SignalType,
    SignalStrength,
    StrategyConfig,
    StrategyMetrics,
    MarketRegime
)

from strategies.statistical_arbitrage import (
    StatisticalArbitrage,
    PairTradingStrategy,
    CointegrationTest,
    SpreadCalculator,
    ZScoreTracker,
    ArbitrageOpportunity
)

from strategies.market_making import (
    MarketMaking,
    MarketMaker,
    OrderBookAnalyzer,
    SpreadOptimizer,
    InventoryManager,
    MarketMakingConfig
)

from strategies.scalping import (
    Scalping,
    MicrostructureAnalyzer,
    OrderFlowTracker,
    TickAnalyzer,
    ScalpingSignal,
    ExecutionTiming
)

from strategies.ensemble_strategy import (
    EnsembleStrategy,
    VotingMethod,
    WeightingScheme,
    StrategyEnsemble,
    ConsensusBuilder,
    PerformanceWeighter
)

# Version
__version__ = "1.0.0"

# Exports publics
__all__ = [
    # Base
    "BaseStrategy",
    "StrategyState",
    "Signal",
    "SignalType",
    "SignalStrength",
    "StrategyConfig",
    "StrategyMetrics",
    "MarketRegime",
    
    # Statistical Arbitrage
    "StatisticalArbitrage",
    "PairTradingStrategy",
    "CointegrationTest",
    "SpreadCalculator",
    "ZScoreTracker",
    "ArbitrageOpportunity",
    
    # Market Making
    "MarketMaking",
    "MarketMaker",
    "OrderBookAnalyzer",
    "SpreadOptimizer",
    "InventoryManager",
    "MarketMakingConfig",
    
    # Scalping
    "Scalping",
    "MicrostructureAnalyzer",
    "OrderFlowTracker",
    "TickAnalyzer",
    "ScalpingSignal",
    "ExecutionTiming",
    
    # Ensemble
    "EnsembleStrategy",
    "VotingMethod",
    "WeightingScheme",
    "StrategyEnsemble",
    "ConsensusBuilder",
    "PerformanceWeighter"
]

# Registry des stratégies disponibles
STRATEGY_REGISTRY = {
    "statistical_arbitrage": StatisticalArbitrage,
    "market_making": MarketMaking,
    "scalping": Scalping,
    "ensemble": EnsembleStrategy
}

# Configuration par défaut des stratégies
DEFAULT_STRATEGY_CONFIGS = {
    "statistical_arbitrage": {
        "lookback_period": 100,
        "entry_z_score": 2.0,
        "exit_z_score": 0.5,
        "cointegration_pvalue": 0.05,
        "min_half_life": 1,
        "max_half_life": 100
    },
    "market_making": {
        "spread_multiplier": 1.5,
        "inventory_limit": 100000,  # USD
        "skew_factor": 0.1,
        "min_spread_bps": 10,  # basis points
        "max_spread_bps": 100,
        "order_levels": 5
    },
    "scalping": {
        "tick_window": 100,
        "min_profit_ticks": 2,
        "max_holding_seconds": 30,
        "volume_threshold": 0.5,
        "momentum_lookback": 10,
        "microstructure_signals": True
    },
    "ensemble": {
        "voting_method": "weighted",
        "weight_scheme": "performance",
        "min_consensus": 0.6,
        "reweight_frequency": 3600,  # seconds
        "strategy_timeout": 1.0  # seconds
    }
}


def create_strategy(
    strategy_name: str,
    config: dict = None,
    **kwargs
) -> BaseStrategy:
    """
    Factory function pour créer une stratégie.
    
    Args:
        strategy_name: Nom de la stratégie
        config: Configuration optionnelle
        **kwargs: Arguments supplémentaires
        
    Returns:
        Instance de la stratégie
        
    Raises:
        ValueError: Si la stratégie n'existe pas
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    
    # Merger avec config par défaut
    default_config = DEFAULT_STRATEGY_CONFIGS.get(strategy_name, {})
    merged_config = {**default_config, **(config or {}), **kwargs}
    
    # Créer l'instance
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    return strategy_class(merged_config)


def list_strategies() -> list:
    """Retourne la liste des stratégies disponibles"""
    return list(STRATEGY_REGISTRY.keys())


def get_strategy_info(strategy_name: str) -> dict:
    """
    Retourne les informations sur une stratégie.
    
    Args:
        strategy_name: Nom de la stratégie
        
    Returns:
        Dict avec infos de la stratégie
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    
    strategy_class = STRATEGY_REGISTRY[strategy_name]
    default_config = DEFAULT_STRATEGY_CONFIGS.get(strategy_name, {})
    
    return {
        "name": strategy_name,
        "class": strategy_class.__name__,
        "module": strategy_class.__module__,
        "default_config": default_config,
        "docstring": strategy_class.__doc__
    }