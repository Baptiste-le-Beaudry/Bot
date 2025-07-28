"""
ML Features Sub-package
======================

Ingénierie des caractéristiques et indicateurs techniques pour ML.
"""

from ml.features.feature_engineering import (
    FeatureEngineer,
    FeatureSet,
    FeatureConfig,
    FeaturePipeline,
    FeatureImportance
)

from ml.features.technical_indicators import (
    TechnicalIndicators,
    IndicatorLibrary,
    MovingAverage,
    RSI,
    MACD,
    BollingerBands,
    ATR,
    Stochastic,
    OBV,
    VWAP
)

from ml.features.market_regime import (
    MarketRegimeDetector,
    RegimeType,
    RegimeFeatures,
    RegimeTransition,
    VolatilityRegime
)

# Types communs
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np


class FeatureType(Enum):
    """Types de features"""
    PRICE = "price"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    MOMENTUM = "momentum"
    TREND = "trend"
    MARKET_MICROSTRUCTURE = "microstructure"
    SENTIMENT = "sentiment"
    FUNDAMENTAL = "fundamental"


class IndicatorType(Enum):
    """Types d'indicateurs techniques"""
    TREND_FOLLOWING = "trend_following"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    SUPPORT_RESISTANCE = "support_resistance"
    CUSTOM = "custom"


@dataclass
class FeatureDefinition:
    """Définition d'une feature"""
    name: str
    type: FeatureType
    lookback_period: int
    parameters: Dict[str, Any]
    dependencies: List[str] = None
    cache_enabled: bool = True


__all__ = [
    # Feature Engineering
    "FeatureEngineer",
    "FeatureSet",
    "FeatureConfig",
    "FeaturePipeline",
    "FeatureImportance",
    
    # Technical Indicators
    "TechnicalIndicators",
    "IndicatorLibrary",
    "MovingAverage",
    "RSI",
    "MACD",
    "BollingerBands",
    "ATR",
    "Stochastic",
    "OBV",
    "VWAP",
    
    # Market Regime
    "MarketRegimeDetector",
    "RegimeType",
    "RegimeFeatures",
    "RegimeTransition",
    "VolatilityRegime",
    
    # Common
    "FeatureType",
    "IndicatorType",
    "FeatureDefinition"
]