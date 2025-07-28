"""
Data Processors Sub-package
==========================

Processeurs pour normalisation, validation et stockage des features.
"""

from data.processors.data_normalizer import (
    DataNormalizer,
    NormalizationMethod,
    Scaler,
    OutlierDetector,
    DataTransformer
)

from data.processors.data_validator import (
    DataValidator,
    ValidationRule,
    DataQuality,
    ValidationResult,
    QualityMetrics
)

from data.processors.feature_store import (
    FeatureStore,
    FeatureSet,
    FeatureConfig,
    FeatureVersion,
    FeatureRegistry
)

# Types et enums communs
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd


class NormalizationMethod(Enum):
    """Méthodes de normalisation disponibles"""
    NONE = "none"
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    ROBUST = "robust"
    LOG = "log"
    QUANTILE = "quantile"


class DataQuality(Enum):
    """Niveaux de qualité des données"""
    EXCELLENT = "excellent"  # > 99% complet, pas d'anomalies
    GOOD = "good"           # > 95% complet, peu d'anomalies
    FAIR = "fair"           # > 90% complet, anomalies gérables
    POOR = "poor"           # < 90% complet ou anomalies importantes
    INVALID = "invalid"     # Données inutilisables


@dataclass
class ProcessingConfig:
    """Configuration pour le traitement des données"""
    normalization_method: NormalizationMethod = NormalizationMethod.Z_SCORE
    outlier_threshold: float = 3.0
    missing_threshold: float = 0.05
    validation_rules: List[str] = None
    feature_engineering: bool = True
    cache_features: bool = True


__all__ = [
    # Data Normalizer
    "DataNormalizer",
    "NormalizationMethod",
    "Scaler",
    "OutlierDetector",
    "DataTransformer",
    
    # Data Validator
    "DataValidator",
    "ValidationRule",
    "DataQuality",
    "ValidationResult",
    "QualityMetrics",
    
    # Feature Store
    "FeatureStore",
    "FeatureSet",
    "FeatureConfig",
    "FeatureVersion",
    "FeatureRegistry",
    
    # Common types
    "ProcessingConfig"
]