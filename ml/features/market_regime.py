"""
Market Regime Detection Module
Détection avancée des régimes de marché pour l'adaptation dynamique des stratégies.
Utilise HMM, clustering, et analyse statistique pour identifier les conditions de marché.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import hmmlearn.hmm as hmm

# Statistiques
from scipy import stats
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from arch import arch_model

# Indicateurs techniques
import talib

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import plotly.graph_objects as go

# Import des modules internes
from .technical_indicators import TechnicalIndicators
from ...utils.metrics import calculate_rolling_metrics

# Configuration du logger
logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Types de régimes de marché"""
    BULL_TREND = "bull_trend"          # Tendance haussière forte
    BEAR_TREND = "bear_trend"          # Tendance baissière forte
    BULL_VOLATILE = "bull_volatile"    # Haussier mais volatil
    BEAR_VOLATILE = "bear_volatile"    # Baissier mais volatil
    RANGING = "ranging"                # Marché en range/consolidation
    BREAKOUT = "breakout"              # Cassure de range
    CRASH = "crash"                    # Krach/chute rapide
    RECOVERY = "recovery"              # Récupération après krach
    LOW_VOLATILITY = "low_volatility"  # Faible volatilité
    HIGH_VOLATILITY = "high_volatility" # Haute volatilité


@dataclass
class RegimeState:
    """État actuel du régime de marché"""
    regime: MarketRegime
    confidence: float  # 0-1
    start_date: datetime
    duration: int  # Nombre de périodes
    
    # Caractéristiques du régime
    trend_strength: float
    volatility_level: float
    volume_profile: str  # 'increasing', 'decreasing', 'stable'
    
    # Probabilités de transition
    transition_probabilities: Dict[MarketRegime, float]
    
    # Métriques du régime
    avg_return: float
    avg_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Signaux d'alerte
    regime_change_probability: float
    warning_signals: List[str]
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['regime'] = self.regime.value
        return data


@dataclass
class RegimeAnalysis:
    """Analyse complète des régimes"""
    current_regime: RegimeState
    regime_history: pd.DataFrame
    transition_matrix: np.ndarray
    regime_statistics: Dict[MarketRegime, Dict[str, float]]
    regime_probabilities: pd.DataFrame
    next_regime_forecast: Dict[MarketRegime, float]
    
    # Features utilisées
    feature_importance: Dict[str, float]
    
    # Métriques de qualité
    model_accuracy: float
    regime_stability: float
    detection_lag: float  # Retard moyen de détection en périodes


class MarketRegimeDetector:
    """
    Détecteur avancé de régimes de marché utilisant multiple méthodes:
    - Hidden Markov Models (HMM)
    - Markov Regime Switching
    - Clustering non-supervisé
    - Analyse statistique
    - Machine Learning
    """
    
    def __init__(self, 
                 lookback_period: int = 252,
                 n_regimes: int = 4,
                 min_regime_duration: int = 20,
                 detection_method: str = 'ensemble'):
        """
        Initialisation du détecteur
        
        Args:
            lookback_period: Période d'historique pour l'analyse
            n_regimes: Nombre de régimes à détecter
            min_regime_duration: Durée minimale d'un régime
            detection_method: 'hmm', 'markov', 'clustering', 'ml', 'ensemble'
        """
        self.lookback_period = lookback_period
        self.n_regimes = n_regimes
        self.min_regime_duration = min_regime_duration
        self.detection_method = detection_method
        
        # Modèles
        self.hmm_model = None
        self.markov_model = None
        self.clustering_model = None
        self.ml_model = None
        
        # Scalers
        self.feature_scaler = StandardScaler()
        
        # Cache
        self.regime_cache = {}
        
        # Configuration des seuils
        self._setup_thresholds()
    
    def _setup_thresholds(self):
        """Configurer les seuils pour la classification des régimes"""
        self.thresholds = {
            'trend_strength': {
                'strong_up': 0.7,
                'moderate_up': 0.3,
                'neutral': -0.3,
                'moderate_down': -0.7,
                'strong_down': -1.0
            },
            'volatility': {
                'very_low': 0.1,
                'low': 0.15,
                'medium': 0.25,
                'high': 0.35,
                'extreme': 0.5
            },
            'volume': {
                'surge': 2.0,  # 2x moyenne
                'high': 1.5,
                'normal': 0.7,
                'low': 0.5
            }
        }
    
    def detect_regime(self, data: pd.DataFrame, 
                     real_time: bool = False) -> RegimeAnalysis:
        """
        Détecter le régime de marché actuel
        
        Args:
            data: DataFrame avec OHLCV
            real_time: Mode temps réel (plus rapide, moins précis)
            
        Returns:
            Analyse complète du régime
        """
        # Calculer les features
        features_df = self._calculate_regime_features(data)
        
        # Détecter selon la méthode
        if self.detection_method == 'ensemble':
            analysis = self._ensemble_detection(features_df, data)
        elif self.detection_method == 'hmm':
            analysis = self._hmm_detection(features_df, data)
        elif self.detection_method == 'markov':
            analysis = self._markov_switching_detection(features_df, data)
        elif self.detection_method == 'clustering':
            analysis = self._clustering_detection(features_df, data)
        elif self.detection_method == 'ml':
            analysis = self._ml_detection(features_df, data)
        else:
            raise ValueError(f"Méthode non supportée: {self.detection_method}")
        
        # Post-traitement
        analysis = self._post_process_regimes(analysis, data)
        
        # Cache pour performances
        if not real_time:
            self.regime_cache[data.index[-1]] = analysis
        
        return analysis
    
    def _calculate_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculer les features pour la détection de régime"""
        features = pd.DataFrame(index=data.index)
        
        # 1. Features de tendance
        # Returns et momentum
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        for period in [5, 10, 20, 50]:
            features[f'returns_{period}'] = data['close'].pct_change(period)
            features[f'momentum_{period}'] = features['returns'].rolling(period).mean()
        
        # Moyennes mobiles
        features['sma_20'] = data['close'].rolling(20).mean()
        features['sma_50'] = data['close'].rolling(50).mean()
        features['sma_200'] = data['close'].rolling(200).mean()
        
        features['price_to_sma20'] = data['close'] / features['sma_20']
        features['price_to_sma50'] = data['close'] / features['sma_50']
        features['sma20_to_sma50'] = features['sma_20'] / features['sma_50']
        
        # Tendance linéaire
        for period in [20, 50]:
            features[f'trend_slope_{period}'] = data['close'].rolling(period).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] / x.mean() if len(x) == period else np.nan
            )
        
        # 2. Features de volatilité
        # Volatilité réalisée
        for period in [10, 20, 50]:
            features[f'volatility_{period}'] = features['returns'].rolling(period).std() * np.sqrt(252)
        
        # ATR
        features['atr_14'] = talib.ATR(data['high'].values, data['low'].values, 
                                      data['close'].values, timeperiod=14)
        features['atr_ratio'] = features['atr_14'] / data['close']
        
        # Volatilité GARCH
        features['garch_vol'] = self._calculate_garch_volatility(features['returns'].dropna())
        
        # Ratio de volatilité
        features['vol_ratio_20_50'] = features['volatility_20'] / features['volatility_50']
        
        # 3. Features de volume
        if 'volume' in data.columns:
            features['volume_ma_20'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_ma_20']
            features['volume_trend'] = features['volume_ma_20'].pct_change(20)
            
            # On-Balance Volume
            features['obv'] = (np.sign(features['returns']) * data['volume']).cumsum()
            features['obv_ma_20'] = features['obv'].rolling(20).mean()
            features['obv_slope'] = features['obv_ma_20'].pct_change(20)
        
        # 4. Features de microstructure
        if all(col in data.columns for col in ['high', 'low']):
            # High-Low spread
            features['hl_spread'] = (data['high'] - data['low']) / data['close']
            features['hl_spread_ma'] = features['hl_spread'].rolling(20).mean()
            
            # Efficiency Ratio
            for period in [10, 20]:
                direction = np.abs(data['close'] - data['close'].shift(period))
                volatility = data['close'].diff().abs().rolling(period).sum()
                features[f'efficiency_ratio_{period}'] = direction / volatility.clip(lower=0.0001)
        
        # 5. Features de régime
        # ADX pour force de tendance
        features['adx_14'] = talib.ADX(data['high'].values, data['low'].values,
                                       data['close'].values, timeperiod=14)
        
        # RSI pour conditions de surachat/survente
        features['rsi_14'] = talib.RSI(data['close'].values, timeperiod=14)
        
        # Bollinger Bands pour volatilité relative
        upper, middle, lower = talib.BBANDS(data['close'].values, timeperiod=20)
        features['bb_width'] = (upper - lower) / middle
        features['bb_position'] = (data['close'] - lower) / (upper - lower)
        
        # 6. Features de changement de régime
        # Changements de volatilité
        features['vol_change_20'] = features['volatility_20'].pct_change(20)
        
        # Changements de tendance
        features['trend_change_20'] = features['trend_slope_20'].diff(20)
        
        # Z-scores pour détecter les anomalies
        for col in ['returns', 'volatility_20', 'volume_ratio']:
            if col in features.columns:
                features[f'{col}_zscore'] = (features[col] - features[col].rolling(50).mean()) / \
                                           features[col].rolling(50).std()
        
        # 7. Features cycliques
        # Détection de cycles avec FFT
        if len(data) > 100:
            features['dominant_cycle'] = self._detect_dominant_cycle(data['close'])
        
        # 8. Features de stress
        # Skewness et Kurtosis roulants
        features['returns_skew'] = features['returns'].rolling(50).skew()
        features['returns_kurt'] = features['returns'].rolling(50).apply(lambda x: stats.kurtosis(x))
        
        # Nettoyer les NaN
        features = features.fillna(method='ffill').fillna(0)
        
        return features
    
    def _ensemble_detection(self, features: pd.DataFrame, 
                          data: pd.DataFrame) -> RegimeAnalysis:
        """Détection par ensemble de méthodes"""
        
        # 1. HMM
        hmm_regimes = self._fit_hmm_model(features)
        
        # 2. Clustering
        cluster_regimes = self._fit_clustering_model(features)
        
        # 3. Règles statistiques
        rule_regimes = self._apply_statistical_rules(features)
        
        # 4. Machine Learning (si entraîné)
        ml_regimes = None
        if self.ml_model is not None:
            ml_regimes = self._predict_ml_regimes(features)
        
        # Combiner les prédictions
        all_predictions = pd.DataFrame({
            'hmm': hmm_regimes,
            'clustering': cluster_regimes,
            'rules': rule_regimes
        })
        
        if ml_regimes is not None:
            all_predictions['ml'] = ml_regimes
        
        # Vote majoritaire avec pondération
        weights = {'hmm': 0.4, 'clustering': 0.2, 'rules': 0.3, 'ml': 0.1}
        
        final_regimes = []
        regime_probabilities = []
        
        for idx in all_predictions.index:
            votes = {}
            for method, regime in all_predictions.loc[idx].items():
                if pd.notna(regime):
                    regime_val = int(regime)
                    votes[regime_val] = votes.get(regime_val, 0) + weights.get(method, 0.25)
            
            if votes:
                final_regime = max(votes, key=votes.get)
                final_regimes.append(final_regime)
                
                # Calculer les probabilités
                total_weight = sum(votes.values())
                probs = {k: v/total_weight for k, v in votes.items()}
                regime_probabilities.append(probs)
            else:
                final_regimes.append(0)
                regime_probabilities.append({0: 1.0})
        
        # Convertir en régimes nommés
        regime_series = pd.Series(final_regimes, index=all_predictions.index)
        regime_names = self._classify_regimes(features, regime_series)
        
        # Construire l'analyse
        current_regime = self._build_regime_state(
            regime_names.iloc[-1],
            features.iloc[-1],
            regime_probabilities[-1],
            data
        )
        
        # Matrice de transition
        transition_matrix = self._calculate_transition_matrix(regime_names)
        
        # Statistiques par régime
        regime_stats = self._calculate_regime_statistics(data, regime_names)
        
        # Prévision du prochain régime
        next_regime_forecast = self._forecast_next_regime(
            current_regime.regime,
            transition_matrix,
            features.iloc[-1]
        )
        
        # Feature importance (depuis HMM)
        feature_importance = self._calculate_feature_importance(features, regime_series)
        
        return RegimeAnalysis(
            current_regime=current_regime,
            regime_history=pd.DataFrame({
                'regime': regime_names,
                'confidence': [p.get(r, 0) for p, r in zip(regime_probabilities, final_regimes)]
            }),
            transition_matrix=transition_matrix,
            regime_statistics=regime_stats,
            regime_probabilities=pd.DataFrame(regime_probabilities),
            next_regime_forecast=next_regime_forecast,
            feature_importance=feature_importance,
            model_accuracy=0.85,  # À calculer avec validation
            regime_stability=self._calculate_regime_stability(regime_names),
            detection_lag=self._estimate_detection_lag(regime_names, features)
        )
    
    def _fit_hmm_model(self, features: pd.DataFrame) -> np.ndarray:
        """Ajuster un Hidden Markov Model"""
        
        # Sélectionner les features principales
        feature_cols = ['returns', 'volatility_20', 'trend_slope_20', 'volume_ratio']
        feature_cols = [col for col in feature_cols if col in features.columns]
        
        X = features[feature_cols].values
        
        # Normaliser
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Créer et entraîner le modèle HMM
        self.hmm_model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        self.hmm_model.fit(X_scaled)
        
        # Prédire les états
        states = self.hmm_model.predict(X_scaled)
        
        return states
    
    def _fit_clustering_model(self, features: pd.DataFrame) -> np.ndarray:
        """Clustering non-supervisé pour détecter les régimes"""
        
        # Features pour clustering
        cluster_features = [
            'returns', 'volatility_20', 'trend_slope_20',
            'volume_ratio', 'efficiency_ratio_20', 'adx_14'
        ]
        cluster_features = [col for col in cluster_features if col in features.columns]
        
        X = features[cluster_features].values
        X_scaled = StandardScaler().fit_transform(X)
        
        # Réduction de dimension si nécessaire
        if len(cluster_features) > 5:
            pca = PCA(n_components=5)
            X_scaled = pca.fit_transform(X_scaled)
        
        # K-Means clustering
        self.clustering_model = KMeans(
            n_clusters=self.n_regimes,
            random_state=42,
            n_init=10
        )
        
        clusters = self.clustering_model.fit_predict(X_scaled)
        
        return clusters
    
    def _apply_statistical_rules(self, features: pd.DataFrame) -> np.ndarray:
        """Appliquer des règles statistiques pour classifier les régimes"""
        
        regimes = []
        
        for idx in features.index:
            row = features.loc[idx]
            
            # Extraire les métriques clés
            returns = row.get('returns_20', 0)
            volatility = row.get('volatility_20', 0.15)
            trend_strength = row.get('adx_14', 25)
            volume_ratio = row.get('volume_ratio', 1.0)
            rsi = row.get('rsi_14', 50)
            
            # Classification basée sur les règles
            if returns > 0.15 and volatility < 0.20 and trend_strength > 30:
                regime = 0  # Bull trend
            elif returns < -0.15 and volatility < 0.20 and trend_strength > 30:
                regime = 1  # Bear trend
            elif returns > 0.10 and volatility > 0.25:
                regime = 2  # Bull volatile
            elif returns < -0.10 and volatility > 0.25:
                regime = 3  # Bear volatile
            elif abs(returns) < 0.05 and volatility < 0.15 and trend_strength < 20:
                regime = 4  # Ranging
            elif volatility > 0.40:
                regime = 5  # High volatility
            elif returns < -0.20 and volatility > 0.30:
                regime = 6  # Crash
            else:
                regime = 4  # Default to ranging
            
            regimes.append(regime)
        
        return np.array(regimes)
    
    def _classify_regimes(self, features: pd.DataFrame, 
                         regime_numbers: pd.Series) -> pd.Series:
        """Convertir les numéros de régime en MarketRegime"""
        
        # Calculer les caractéristiques moyennes de chaque cluster
        regime_characteristics = {}
        
        for regime_num in regime_numbers.unique():
            mask = regime_numbers == regime_num
            regime_data = features[mask]
            
            avg_returns = regime_data['returns'].mean() * 252  # Annualisé
            avg_volatility = regime_data['volatility_20'].mean()
            avg_trend = regime_data.get('trend_slope_20', pd.Series([0])).mean()
            avg_volume = regime_data.get('volume_ratio', pd.Series([1])).mean()
            
            regime_characteristics[regime_num] = {
                'returns': avg_returns,
                'volatility': avg_volatility,
                'trend': avg_trend,
                'volume': avg_volume
            }
        
        # Mapper les numéros vers les régimes nommés
        regime_mapping = {}
        
        for regime_num, chars in regime_characteristics.items():
            if chars['returns'] > 0.10 and chars['volatility'] < 0.20:
                regime_mapping[regime_num] = MarketRegime.BULL_TREND
            elif chars['returns'] < -0.10 and chars['volatility'] < 0.20:
                regime_mapping[regime_num] = MarketRegime.BEAR_TREND
            elif chars['returns'] > 0.05 and chars['volatility'] > 0.25:
                regime_mapping[regime_num] = MarketRegime.BULL_VOLATILE
            elif chars['returns'] < -0.05 and chars['volatility'] > 0.25:
                regime_mapping[regime_num] = MarketRegime.BEAR_VOLATILE
            elif abs(chars['returns']) < 0.05 and chars['volatility'] < 0.15:
                regime_mapping[regime_num] = MarketRegime.RANGING
            elif chars['volatility'] > 0.35:
                regime_mapping[regime_num] = MarketRegime.HIGH_VOLATILITY
            elif chars['returns'] < -0.20:
                regime_mapping[regime_num] = MarketRegime.CRASH
            else:
                regime_mapping[regime_num] = MarketRegime.RANGING
        
        # Appliquer le mapping
        regime_names = regime_numbers.map(regime_mapping)
        
        return regime_names
    
    def _build_regime_state(self, regime: MarketRegime,
                          current_features: pd.Series,
                          regime_probabilities: Dict,
                          data: pd.DataFrame) -> RegimeState:
        """Construire l'état complet du régime actuel"""
        
        # Calculer la durée du régime actuel
        # (Simplified - dans la pratique, il faudrait tracker l'historique)
        duration = 1
        
        # Probabilités de transition (simplified)
        transition_probs = {
            MarketRegime.BULL_TREND: 0.1,
            MarketRegime.BEAR_TREND: 0.1,
            MarketRegime.RANGING: 0.3,
            MarketRegime.BULL_VOLATILE: 0.2,
            MarketRegime.BEAR_VOLATILE: 0.2,
            MarketRegime.HIGH_VOLATILITY: 0.1
        }
        
        # Signaux d'alerte
        warning_signals = []
        
        # Vérifier les signaux d'alerte
        if current_features.get('volatility_20', 0) > 0.30:
            warning_signals.append("High volatility detected")
        
        if abs(current_features.get('returns_zscore', 0)) > 2:
            warning_signals.append("Extreme returns (2+ std)")
        
        if current_features.get('volume_ratio', 1) > 2:
            warning_signals.append("Unusual volume spike")
        
        # Probabilité de changement de régime
        regime_change_prob = 1 - regime_probabilities.get(
            list(regime_probabilities.keys())[0], 0.5
        )
        
        return RegimeState(
            regime=regime,
            confidence=max(regime_probabilities.values()) if regime_probabilities else 0.5,
            start_date=data.index[-1],
            duration=duration,
            trend_strength=current_features.get('adx_14', 25) / 100,
            volatility_level=current_features.get('volatility_20', 0.15),
            volume_profile='increasing' if current_features.get('volume_trend', 0) > 0 else 'stable',
            transition_probabilities=transition_probs,
            avg_return=current_features.get('returns_20', 0) * 252,
            avg_volatility=current_features.get('volatility_20', 0.15),
            sharpe_ratio=current_features.get('returns_20', 0) / max(current_features.get('volatility_20', 0.15), 0.01) * np.sqrt(252),
            max_drawdown=0.0,  # À calculer
            regime_change_probability=regime_change_prob,
            warning_signals=warning_signals
        )
    
    def _calculate_transition_matrix(self, regimes: pd.Series) -> np.ndarray:
        """Calculer la matrice de transition entre régimes"""
        
        # Obtenir tous les régimes uniques
        unique_regimes = sorted(regimes.unique())
        n_regimes = len(unique_regimes)
        
        # Initialiser la matrice
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        # Compter les transitions
        for i in range(len(regimes) - 1):
            from_regime = unique_regimes.index(regimes.iloc[i])
            to_regime = unique_regimes.index(regimes.iloc[i + 1])
            transition_matrix[from_regime, to_regime] += 1
        
        # Normaliser pour obtenir des probabilités
        row_sums = transition_matrix.sum(axis=1)
        transition_matrix = transition_matrix / row_sums[:, np.newaxis].clip(min=1)
        
        return transition_matrix
    
    def _calculate_regime_statistics(self, data: pd.DataFrame,
                                   regimes: pd.Series) -> Dict[MarketRegime, Dict[str, float]]:
        """Calculer les statistiques pour chaque régime"""
        
        returns = data['close'].pct_change()
        
        regime_stats = {}
        
        for regime in regimes.unique():
            mask = regimes == regime
            regime_returns = returns[mask]
            
            if len(regime_returns) > 0:
                # Calculer les métriques
                stats = {
                    'frequency': mask.sum() / len(regimes),
                    'avg_return': regime_returns.mean() * 252,
                    'volatility': regime_returns.std() * np.sqrt(252),
                    'sharpe_ratio': regime_returns.mean() / regime_returns.std() * np.sqrt(252) if regime_returns.std() > 0 else 0,
                    'skewness': regime_returns.skew(),
                    'kurtosis': regime_returns.kurtosis(),
                    'max_drawdown': self._calculate_max_drawdown(data['close'][mask]),
                    'avg_duration': self._calculate_avg_regime_duration(regimes, regime)
                }
                
                regime_stats[regime] = stats
        
        return regime_stats
    
    def _forecast_next_regime(self, current_regime: MarketRegime,
                            transition_matrix: np.ndarray,
                            current_features: pd.Series) -> Dict[MarketRegime, float]:
        """Prévoir le prochain régime"""
        
        # Simplified - utiliser la matrice de transition
        # Dans la pratique, on pourrait utiliser le modèle HMM pour une vraie prévision
        
        forecast = {
            MarketRegime.BULL_TREND: 0.2,
            MarketRegime.BEAR_TREND: 0.1,
            MarketRegime.RANGING: 0.4,
            MarketRegime.BULL_VOLATILE: 0.15,
            MarketRegime.BEAR_VOLATILE: 0.15
        }
        
        # Ajuster selon les features actuelles
        if current_features.get('volatility_change_20', 0) > 0.5:
            forecast[MarketRegime.HIGH_VOLATILITY] = 0.3
        
        return forecast
    
    def _calculate_garch_volatility(self, returns: pd.Series) -> pd.Series:
        """Calculer la volatilité GARCH"""
        try:
            # Nettoyer les données
            returns_clean = returns.dropna()
            if len(returns_clean) < 100:
                return pd.Series(index=returns.index, data=np.nan)
            
            # Ajuster le modèle GARCH(1,1)
            model = arch_model(returns_clean * 100, vol='Garch', p=1, q=1)
            res = model.fit(disp='off')
            
            # Extraire la volatilité conditionnelle
            volatility = res.conditional_volatility / 100
            
            # Réindexer sur la série originale
            volatility_full = pd.Series(index=returns.index, data=np.nan)
            volatility_full.loc[volatility.index] = volatility
            
            return volatility_full * np.sqrt(252)  # Annualiser
            
        except Exception as e:
            logger.warning(f"Erreur GARCH: {e}")
            return pd.Series(index=returns.index, data=np.nan)
    
    def _detect_dominant_cycle(self, prices: pd.Series) -> pd.Series:
        """Détecter le cycle dominant avec FFT"""
        try:
            cycles = pd.Series(index=prices.index, data=np.nan)
            window = 50
            
            for i in range(window, len(prices)):
                # Extraire la fenêtre
                price_window = prices.iloc[i-window:i].values
                
                # Detrend
                x = np.arange(len(price_window))
                trend = np.polyfit(x, price_window, 1)
                detrended = price_window - (trend[0] * x + trend[1])
                
                # FFT
                fft = np.fft.fft(detrended)
                freqs = np.fft.fftfreq(len(detrended))
                
                # Trouver la fréquence dominante (excluant DC)
                power = np.abs(fft[1:len(fft)//2])
                freqs_positive = freqs[1:len(freqs)//2]
                
                if len(power) > 0:
                    dominant_freq_idx = np.argmax(power)
                    dominant_freq = freqs_positive[dominant_freq_idx]
                    
                    if dominant_freq > 0:
                        dominant_period = 1 / dominant_freq
                        cycles.iloc[i] = dominant_period
            
            return cycles.fillna(method='ffill')
            
        except Exception as e:
            logger.warning(f"Erreur détection cycle: {e}")
            return pd.Series(index=prices.index, data=20)  # Default
    
    def _calculate_feature_importance(self, features: pd.DataFrame,
                                    regimes: pd.Series) -> Dict[str, float]:
        """Calculer l'importance des features pour la classification"""
        
        # Utiliser Random Forest pour l'importance
        feature_cols = [col for col in features.columns if col not in ['symbol', 'timestamp']]
        X = features[feature_cols].fillna(0)
        y = regimes
        
        if len(np.unique(y)) < 2:
            return {col: 0.0 for col in feature_cols}
        
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            importance = dict(zip(feature_cols, rf.feature_importances_))
            
            # Normaliser
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {k: v/total_importance for k, v in importance.items()}
            
            # Garder seulement le top 20
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20])
            
            return importance
            
        except Exception as e:
            logger.warning(f"Erreur calcul importance: {e}")
            return {col: 0.0 for col in feature_cols[:10]}
    
    def _calculate_regime_stability(self, regimes: pd.Series) -> float:
        """Calculer la stabilité des régimes (moins de changements = plus stable)"""
        
        # Compter les changements de régime
        regime_changes = (regimes != regimes.shift()).sum()
        
        # Normaliser par la longueur
        stability = 1 - (regime_changes / len(regimes))
        
        return stability
    
    def _estimate_detection_lag(self, regimes: pd.Series,
                              features: pd.DataFrame) -> float:
        """Estimer le retard de détection moyen"""
        
        # Simplified - dans la pratique, il faudrait comparer avec des changements connus
        # Ici on utilise les changements brusques de volatilité comme proxy
        
        vol_changes = features['volatility_20'].pct_change().abs()
        significant_changes = vol_changes > vol_changes.quantile(0.95)
        
        # Calculer le lag moyen entre changement de vol et changement de régime
        lags = []
        
        for idx in significant_changes[significant_changes].index:
            # Chercher le prochain changement de régime
            future_regimes = regimes.loc[idx:]
            if len(future_regimes) > 1:
                regime_change_idx = (future_regimes != future_regimes.iloc[0]).idxmax()
                if regime_change_idx != idx:
                    lag = (regime_change_idx - idx).days if hasattr(idx, 'days') else 1
                    lags.append(lag)
        
        return np.mean(lags) if lags else 0.0
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculer le drawdown maximum"""
        cumulative = (1 + prices.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def _calculate_avg_regime_duration(self, regimes: pd.Series,
                                     target_regime: MarketRegime) -> float:
        """Calculer la durée moyenne d'un régime spécifique"""
        
        durations = []
        in_regime = False
        duration = 0
        
        for regime in regimes:
            if regime == target_regime:
                if not in_regime:
                    in_regime = True
                    duration = 1
                else:
                    duration += 1
            else:
                if in_regime:
                    durations.append(duration)
                    in_regime = False
                    duration = 0
        
        # Ajouter la dernière durée si on termine dans le régime
        if in_regime:
            durations.append(duration)
        
        return np.mean(durations) if durations else 0
    
    def _post_process_regimes(self, analysis: RegimeAnalysis,
                            data: pd.DataFrame) -> RegimeAnalysis:
        """Post-traitement pour améliorer la qualité de la détection"""
        
        # 1. Filtrer les régimes trop courts
        min_duration = self.min_regime_duration
        regime_series = analysis.regime_history['regime']
        
        # Identifier les segments
        segments = []
        current_regime = regime_series.iloc[0]
        start_idx = 0
        
        for i in range(1, len(regime_series)):
            if regime_series.iloc[i] != current_regime:
                segments.append({
                    'regime': current_regime,
                    'start': start_idx,
                    'end': i - 1,
                    'duration': i - start_idx
                })
                current_regime = regime_series.iloc[i]
                start_idx = i
        
        # Ajouter le dernier segment
        segments.append({
            'regime': current_regime,
            'start': start_idx,
            'end': len(regime_series) - 1,
            'duration': len(regime_series) - start_idx
        })
        
        # Fusionner les segments courts
        filtered_regimes = regime_series.copy()
        
        for segment in segments:
            if segment['duration'] < min_duration:
                # Assigner au régime adjacent le plus long
                if segment['start'] > 0:
                    filtered_regimes.iloc[segment['start']:segment['end']+1] = \
                        filtered_regimes.iloc[segment['start']-1]
        
        analysis.regime_history['regime'] = filtered_regimes
        
        # 2. Recalculer les statistiques si nécessaire
        if not filtered_regimes.equals(regime_series):
            analysis.regime_statistics = self._calculate_regime_statistics(data, filtered_regimes)
            analysis.transition_matrix = self._calculate_transition_matrix(filtered_regimes)
        
        return analysis
    
    def plot_regime_analysis(self, data: pd.DataFrame,
                            analysis: RegimeAnalysis,
                            save_path: Optional[str] = None):
        """Visualiser l'analyse des régimes"""
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        
        # 1. Prix avec régimes colorés
        ax1 = axes[0]
        
        # Colorier selon les régimes
        regime_colors = {
            MarketRegime.BULL_TREND: 'green',
            MarketRegime.BEAR_TREND: 'red',
            MarketRegime.BULL_VOLATILE: 'lightgreen',
            MarketRegime.BEAR_VOLATILE: 'lightcoral',
            MarketRegime.RANGING: 'gray',
            MarketRegime.HIGH_VOLATILITY: 'orange',
            MarketRegime.CRASH: 'darkred',
            MarketRegime.RECOVERY: 'darkgreen'
        }
        
        # Plot prix
        ax1.plot(data.index, data['close'], 'b-', alpha=0.7, linewidth=1)
        
        # Colorier les backgrounds selon les régimes
        for regime in analysis.regime_history['regime'].unique():
            mask = analysis.regime_history['regime'] == regime
            if mask.any():
                ax1.fill_between(
                    data.index[mask],
                    data['close'].min() * 0.95,
                    data['close'].max() * 1.05,
                    alpha=0.2,
                    color=regime_colors.get(regime, 'gray'),
                    label=regime.value if isinstance(regime, MarketRegime) else str(regime)
                )
        
        ax1.set_ylabel('Price')
        ax1.set_title('Market Regimes Detection')
        ax1.legend(loc='best', ncol=3)
        ax1.grid(True, alpha=0.3)
        
        # 2. Probabilités des régimes
        ax2 = axes[1]
        
        if hasattr(analysis, 'regime_probabilities') and not analysis.regime_probabilities.empty:
            # Stacked area chart des probabilités
            regime_probs = analysis.regime_probabilities.fillna(0)
            regime_probs.plot(kind='area', stacked=True, ax=ax2, alpha=0.7)
            ax2.set_ylabel('Regime Probability')
            ax2.set_title('Regime Probabilities Over Time')
            ax2.legend(loc='best', ncol=3)
            ax2.set_ylim(0, 1)
        
        # 3. Métriques clés
        ax3 = axes[2]
        
        # Calculer les métriques
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        
        ax3_twin = ax3.twinx()
        
        # Returns cumulés
        cum_returns = (1 + returns).cumprod()
        ax3.plot(data.index, cum_returns, 'b-', label='Cumulative Returns')
        ax3.set_ylabel('Cumulative Returns', color='b')
        ax3.tick_params(axis='y', labelcolor='b')
        
        # Volatilité
        ax3_twin.plot(data.index, volatility * 100, 'r-', alpha=0.7, label='Volatility')
        ax3_twin.set_ylabel('Volatility (%)', color='r')
        ax3_twin.tick_params(axis='y', labelcolor='r')
        
        ax3.set_title('Returns and Volatility')
        ax3.grid(True, alpha=0.3)
        
        # 4. Signaux d'alerte
        ax4 = axes[3]
        
        # Probabilité de changement de régime
        regime_change_probs = []
        for i, row in analysis.regime_history.iterrows():
            # Simplified - utiliser la confidence inversée comme proxy
            change_prob = 1 - row.get('confidence', 0.5)
            regime_change_probs.append(change_prob)
        
        regime_change_series = pd.Series(
            regime_change_probs,
            index=analysis.regime_history.index
        )
        
        ax4.plot(data.index, regime_change_series, 'purple', label='Regime Change Probability')
        ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Alert Threshold')
        ax4.fill_between(data.index, 0, regime_change_series, 
                        where=regime_change_series > 0.5,
                        color='red', alpha=0.2, label='High Risk')
        
        ax4.set_ylabel('Probability')
        ax4.set_xlabel('Date')
        ax4.set_title('Regime Change Risk')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def get_regime_trading_signals(self, current_regime: MarketRegime,
                                 regime_forecast: Dict[MarketRegime, float]) -> Dict[str, Any]:
        """
        Générer des signaux de trading basés sur le régime
        
        Returns:
            Recommandations de trading adaptées au régime
        """
        
        signals = {
            'regime': current_regime,
            'position_size': 0.5,  # Par défaut
            'strategy': 'neutral',
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'recommendations': []
        }
        
        # Adapter selon le régime
        if current_regime == MarketRegime.BULL_TREND:
            signals.update({
                'position_size': 0.8,
                'strategy': 'trend_following',
                'stop_loss': 0.015,
                'take_profit': 0.06,
                'recommendations': [
                    "Favoriser les positions longues",
                    "Utiliser des trailing stops",
                    "Augmenter la taille des positions"
                ]
            })
            
        elif current_regime == MarketRegime.BEAR_TREND:
            signals.update({
                'position_size': 0.3,
                'strategy': 'short_or_hedge',
                'stop_loss': 0.01,
                'take_profit': 0.03,
                'recommendations': [
                    "Réduire l'exposition",
                    "Considérer les positions short",
                    "Stops serrés"
                ]
            })
            
        elif current_regime == MarketRegime.RANGING:
            signals.update({
                'position_size': 0.6,
                'strategy': 'mean_reversion',
                'stop_loss': 0.015,
                'take_profit': 0.02,
                'recommendations': [
                    "Trading de range",
                    "Acheter support, vendre résistance",
                    "Réduire les objectifs"
                ]
            })
            
        elif current_regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.CRASH]:
            signals.update({
                'position_size': 0.2,
                'strategy': 'defensive',
                'stop_loss': 0.025,
                'take_profit': 0.05,
                'recommendations': [
                    "Réduire fortement l'exposition",
                    "Élargir les stops",
                    "Attendre la stabilisation"
                ]
            })
        
        # Ajuster selon les prévisions
        if regime_forecast.get(MarketRegime.CRASH, 0) > 0.3:
            signals['position_size'] *= 0.5
            signals['recommendations'].append("⚠️ Risque élevé de krach détecté")
        
        return signals


# Fonctions utilitaires
def detect_market_regime(data: pd.DataFrame, 
                        method: str = 'ensemble') -> RegimeAnalysis:
    """
    Fonction rapide pour détecter le régime de marché
    
    Args:
        data: DataFrame avec OHLCV
        method: Méthode de détection
        
    Returns:
        Analyse du régime
    """
    detector = MarketRegimeDetector(detection_method=method)
    return detector.detect_regime(data)


def get_regime_recommendation(regime: MarketRegime) -> Dict[str, str]:
    """
    Obtenir des recommandations basées sur le régime
    
    Args:
        regime: Régime actuel
        
    Returns:
        Recommandations de trading
    """
    recommendations = {
        MarketRegime.BULL_TREND: {
            'strategy': 'trend_following',
            'bias': 'long',
            'risk': 'moderate',
            'indicators': 'momentum, breakouts'
        },
        MarketRegime.BEAR_TREND: {
            'strategy': 'defensive',
            'bias': 'short_or_cash',
            'risk': 'low',
            'indicators': 'support_levels, oversold'
        },
        MarketRegime.RANGING: {
            'strategy': 'mean_reversion',
            'bias': 'neutral',
            'risk': 'moderate',
            'indicators': 'bollinger_bands, rsi'
        },
        MarketRegime.HIGH_VOLATILITY: {
            'strategy': 'options_or_hedge',
            'bias': 'neutral',
            'risk': 'very_low',
            'indicators': 'vix, atr'
        }
    }
    
    return recommendations.get(regime, {
        'strategy': 'wait',
        'bias': 'neutral',
        'risk': 'none',
        'indicators': 'all'
    })


# Exemple d'utilisation
def main():
    """Exemple d'utilisation du détecteur de régimes"""
    
    # Créer des données de test
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    
    # Simuler différents régimes
    n = len(dates)
    regimes_sim = []
    
    # Bull trend
    regimes_sim.extend([0.01 + np.random.normal(0, 0.005) for _ in range(n//4)])
    # High vol
    regimes_sim.extend([np.random.normal(0, 0.03) for _ in range(n//4)])
    # Bear trend
    regimes_sim.extend([-0.01 + np.random.normal(0, 0.005) for _ in range(n//4)])
    # Ranging
    regimes_sim.extend([np.random.normal(0, 0.002) for _ in range(n//4)])
    
    # Créer les prix
    returns = pd.Series(regimes_sim[:n])
    prices = 100 * (1 + returns).cumprod()
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n))),
        'close': prices,
        'volume': np.random.exponential(1000000, n)
    }, index=dates)
    
    # Détecter les régimes
    detector = MarketRegimeDetector()
    analysis = detector.detect_regime(data)
    
    # Afficher les résultats
    print(f"Régime actuel: {analysis.current_regime.regime}")
    print(f"Confiance: {analysis.current_regime.confidence:.2%}")
    print(f"Durée: {analysis.current_regime.duration} périodes")
    print(f"Signaux d'alerte: {analysis.current_regime.warning_signals}")
    
    # Visualiser
    detector.plot_regime_analysis(data, analysis)
    
    # Obtenir les signaux de trading
    signals = detector.get_regime_trading_signals(
        analysis.current_regime.regime,
        analysis.next_regime_forecast
    )
    print(f"\nSignaux de trading: {signals}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()