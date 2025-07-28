"""
ML Training Sub-package
======================

Outils d'entraînement, backtesting et optimisation pour les modèles ML.
"""

from ml.training.trainer import (
    ModelTrainer,
    TrainingConfig,
    TrainingMetrics,
    TrainingCallback,
    CheckpointManager
)

from ml.training.backtesting import (
    Backtester,
    BacktestConfig,
    BacktestResult,
    WalkForwardAnalysis,
    MonteCarloValidator
)

from ml.training.hyperopt import (
    HyperparameterOptimizer,
    SearchSpace,
    OptimizationObjective,
    BayesianOptimizer,
    GridSearchOptimizer
)

# Types communs
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable
import numpy as np


class ValidationStrategy(Enum):
    """Stratégies de validation"""
    TRAIN_TEST_SPLIT = "train_test_split"
    TIME_SERIES_SPLIT = "time_series_split"
    WALK_FORWARD = "walk_forward"
    PURGED_KFOLD = "purged_kfold"
    COMBINATORIAL = "combinatorial"


class OptimizationMethod(Enum):
    """Méthodes d'optimisation"""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    OPTUNA = "optuna"


@dataclass
class TrainingResult:
    """Résultat d'entraînement"""
    model_id: str
    metrics: Dict[str, float]
    best_epoch: int
    training_time: float
    validation_score: float
    test_score: Optional[float] = None
    hyperparameters: Dict[str, Any] = None


__all__ = [
    # Trainer
    "ModelTrainer",
    "TrainingConfig",
    "TrainingMetrics",
    "TrainingCallback",
    "CheckpointManager",
    
    # Backtesting
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "WalkForwardAnalysis",
    "MonteCarloValidator",
    
    # Hyperopt
    "HyperparameterOptimizer",
    "SearchSpace",
    "OptimizationObjective",
    "BayesianOptimizer",
    "GridSearchOptimizer",
    
    # Common
    "ValidationStrategy",
    "OptimizationMethod",
    "TrainingResult"
]