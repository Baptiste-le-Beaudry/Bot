"""
ML Environments Sub-package
==========================

Environnements OpenAI Gym pour l'entraînement des agents RL.
"""

from ml.environments.trading_env import (
    TradingEnvironment,
    TradingEnvConfig,
    MarketSimulator,
    OrderExecutor,
    EnvironmentState
)

from ml.environments.multi_asset_env import (
    MultiAssetEnvironment,
    AssetUniverse,
    PortfolioState,
    AllocationAction,
    MultiAssetConfig
)

# Types communs
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import gym
from gym import spaces


class ActionSpace(Enum):
    """Types d'espaces d'action"""
    DISCRETE = "discrete"        # Buy/Hold/Sell
    CONTINUOUS = "continuous"    # Position sizing
    MULTI_DISCRETE = "multi_discrete"  # Multiple discrete actions
    MULTI_BINARY = "multi_binary"      # Multiple binary choices


class ObservationSpace(Enum):
    """Types d'espaces d'observation"""
    PRICE_ONLY = "price_only"
    PRICE_VOLUME = "price_volume"
    FULL_ORDERBOOK = "full_orderbook"
    TECHNICAL_INDICATORS = "technical_indicators"
    CUSTOM = "custom"


class RewardFunction(Enum):
    """Fonctions de récompense disponibles"""
    SIMPLE_PNL = "simple_pnl"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    RISK_ADJUSTED = "risk_adjusted"
    DIFFERENTIAL_SHARPE = "differential_sharpe"
    CUSTOM = "custom"


@dataclass
class EnvironmentConfig:
    """Configuration de base pour les environnements"""
    initial_balance: float = 10000.0
    max_steps: int = 1000
    lookback_window: int = 50
    transaction_cost: float = 0.001
    slippage_model: str = "linear"
    reward_function: RewardFunction = RewardFunction.SHARPE_RATIO
    action_space_type: ActionSpace = ActionSpace.CONTINUOUS
    observation_space_type: ObservationSpace = ObservationSpace.TECHNICAL_INDICATORS


class BaseEnvironment(gym.Env):
    """Classe de base pour les environnements de trading"""
    
    def __init__(self, config: EnvironmentConfig):
        super().__init__()
        self.config = config
        self.current_step = 0
        self.done = False
        
    def seed(self, seed=None):
        """Set random seed"""
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def render(self, mode='human'):
        """Render the environment"""
        pass
        
    def close(self):
        """Clean up resources"""
        pass


__all__ = [
    # Trading Environment
    "TradingEnvironment",
    "TradingEnvConfig",
    "MarketSimulator",
    "OrderExecutor",
    "EnvironmentState",
    
    # Multi-Asset Environment
    "MultiAssetEnvironment",
    "AssetUniverse",
    "PortfolioState",
    "AllocationAction",
    "MultiAssetConfig",
    
    # Common types
    "ActionSpace",
    "ObservationSpace",
    "RewardFunction",
    "EnvironmentConfig",
    "BaseEnvironment"
]