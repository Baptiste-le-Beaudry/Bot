"""
Machine Learning and Deep Reinforcement Learning Package
=======================================================

Package contenant tous les modèles d'IA et outils d'apprentissage
pour le trading algorithmique avancé.

Composants principaux:
    - Models: DQN, PPO, SAC et agents d'ensemble
    - Training: Entraînement, backtesting et optimisation
    - Features: Ingénierie des caractéristiques et indicateurs
    - Environments: Environnements Gym pour RL

Usage:
    from ml import DQNAgent, TradingEnvironment, FeatureEngineer
    
    env = TradingEnvironment(config)
    agent = DQNAgent(env.observation_space, env.action_space)
    agent.train(env, episodes=1000)
"""

# Models
from ml.models import (
    DQNAgent,
    PPOAgent,
    SACAgent,
    EnsembleAgent,
    BaseAgent,
    AgentState,
    ModelConfig
)

# Training
from ml.training import (
    ModelTrainer,
    Backtester,
    HyperparameterOptimizer,
    TrainingConfig,
    TrainingMetrics,
    ValidationStrategy
)

# Features
from ml.features import (
    FeatureEngineer,
    TechnicalIndicators,
    MarketRegimeDetector,
    FeatureConfig,
    FeatureSet,
    IndicatorLibrary
)

# Environments
from ml.environments import (
    TradingEnvironment,
    MultiAssetEnvironment,
    EnvironmentConfig,
    RewardFunction,
    ActionSpace,
    ObservationSpace
)

# Version
__version__ = "1.0.0"

# Exports publics
__all__ = [
    # Models
    "DQNAgent",
    "PPOAgent", 
    "SACAgent",
    "EnsembleAgent",
    "BaseAgent",
    "AgentState",
    "ModelConfig",
    
    # Training
    "ModelTrainer",
    "Backtester",
    "HyperparameterOptimizer",
    "TrainingConfig",
    "TrainingMetrics",
    "ValidationStrategy",
    
    # Features
    "FeatureEngineer",
    "TechnicalIndicators",
    "MarketRegimeDetector",
    "FeatureConfig",
    "FeatureSet",
    "IndicatorLibrary",
    
    # Environments
    "TradingEnvironment",
    "MultiAssetEnvironment",
    "EnvironmentConfig",
    "RewardFunction",
    "ActionSpace",
    "ObservationSpace"
]

# Registry des modèles disponibles
MODEL_REGISTRY = {
    "dqn": DQNAgent,
    "ppo": PPOAgent,
    "sac": SACAgent,
    "ensemble": EnsembleAgent
}

# Configuration par défaut pour ML
DEFAULT_ML_CONFIG = {
    "device": "cuda",  # ou "cpu"
    "seed": 42,
    "dtype": "float32",
    "num_workers": 4,
    "pin_memory": True,
    "deterministic": False
}

# Hyperparamètres par défaut par modèle
DEFAULT_HYPERPARAMETERS = {
    "dqn": {
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "buffer_size": 100000,
        "batch_size": 64,
        "target_update_freq": 1000,
        "hidden_layers": [256, 256],
        "activation": "relu"
    },
    "ppo": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "n_epochs": 10,
        "batch_size": 64,
        "n_steps": 2048
    },
    "sac": {
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "alpha": 0.2,
        "automatic_entropy_tuning": True,
        "hidden_layers": [256, 256],
        "buffer_size": 1000000,
        "batch_size": 256,
        "reward_scale": 1.0
    }
}


def create_agent(
    agent_type: str,
    observation_space,
    action_space,
    config: dict = None,
    **kwargs
) -> BaseAgent:
    """
    Factory function pour créer un agent RL.
    
    Args:
        agent_type: Type d'agent ('dqn', 'ppo', 'sac', 'ensemble')
        observation_space: Espace d'observation
        action_space: Espace d'action
        config: Configuration optionnelle
        **kwargs: Arguments supplémentaires
        
    Returns:
        Instance de l'agent
    """
    if agent_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )
    
    # Merger configurations
    default_params = DEFAULT_HYPERPARAMETERS.get(agent_type, {})
    merged_config = {**default_params, **(config or {}), **kwargs}
    
    # Créer l'agent
    agent_class = MODEL_REGISTRY[agent_type]
    return agent_class(
        observation_space=observation_space,
        action_space=action_space,
        config=merged_config
    )


def setup_ml_environment(config: dict = None):
    """
    Configure l'environnement ML (GPU, seeds, etc).
    
    Args:
        config: Configuration optionnelle
    """
    import torch
    import numpy as np
    import random
    
    cfg = {**DEFAULT_ML_CONFIG, **(config or {})}
    
    # Set seeds
    if "seed" in cfg:
        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        random.seed(cfg["seed"])
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg["seed"])
            torch.cuda.manual_seed_all(cfg["seed"])
    
    # Configure device
    if cfg["device"] == "cuda" and torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
        # Optimisations CUDA
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = not cfg["deterministic"]
        torch.backends.cudnn.deterministic = cfg["deterministic"]
    
    # Configure precision
    if cfg["dtype"] == "float16":
        torch.set_default_dtype(torch.float16)
    elif cfg["dtype"] == "float64":
        torch.set_default_dtype(torch.float64)
        
    return cfg


# Auto-setup au chargement
try:
    _ml_config = setup_ml_environment()
except Exception as e:
    import warnings
    warnings.warn(f"Failed to setup ML environment: {e}")