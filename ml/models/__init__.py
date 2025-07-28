"""
Machine Learning Models Sub-package
===================================

Modèles de Deep Reinforcement Learning pour le trading.
"""

from ml.models.dqn import (
    DQNAgent,
    DQNetwork,
    ReplayBuffer,
    PrioritizedReplayBuffer,
    DQNConfig
)

from ml.models.ppo import (
    PPOAgent,
    ActorCriticNetwork,
    PPOBuffer,
    PPOConfig,
    GAECalculator
)

from ml.models.sac import (
    SACAgent,
    SoftActorCritic,
    GaussianPolicy,
    QNetwork,
    SACConfig
)

from ml.models.ensemble_agent import (
    EnsembleAgent,
    EnsembleMethod,
    AgentWeight,
    ConsensusStrategy,
    EnsembleConfig
)

# Base classes communes
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import torch.nn as nn


class AgentState(Enum):
    """États possibles d'un agent"""
    IDLE = "idle"
    TRAINING = "training"
    EVALUATING = "evaluating"
    TRADING = "trading"
    PAUSED = "paused"


@dataclass
class ModelConfig:
    """Configuration de base pour tous les modèles"""
    learning_rate: float = 1e-4
    gamma: float = 0.99
    device: str = "cuda"
    seed: Optional[int] = None
    checkpoint_dir: Optional[str] = "./checkpoints"
    save_frequency: int = 10000
    log_frequency: int = 100


class BaseAgent(ABC):
    """Classe de base pour tous les agents RL"""
    
    def __init__(self, observation_space, action_space, config: ModelConfig):
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.state = AgentState.IDLE
        self.total_steps = 0
        self.episode = 0
        
    @abstractmethod
    def select_action(self, state, deterministic: bool = False):
        """Sélectionne une action"""
        pass
    
    @abstractmethod
    def update(self, batch) -> Dict[str, float]:
        """Met à jour l'agent avec un batch d'expériences"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Sauvegarde le modèle"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Charge le modèle"""
        pass


__all__ = [
    # DQN
    "DQNAgent",
    "DQNetwork",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "DQNConfig",
    
    # PPO
    "PPOAgent",
    "ActorCriticNetwork",
    "PPOBuffer",
    "PPOConfig",
    "GAECalculator",
    
    # SAC
    "SACAgent",
    "SoftActorCritic",
    "GaussianPolicy",
    "QNetwork",
    "SACConfig",
    
    # Ensemble
    "EnsembleAgent",
    "EnsembleMethod",
    "AgentWeight",
    "ConsensusStrategy",
    "EnsembleConfig",
    
    # Base
    "BaseAgent",
    "AgentState",
    "ModelConfig"
]