"""
Feature Engineering Module
Ingénierie des caractéristiques avancées pour les modèles de Deep Reinforcement Learning.
Génère des features de microstructure, patterns, et signaux pour maximiser la rentabilité.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import talib
from scipy import stats, signal
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# Import des modules internes
from ..processors.data_normalizer import NormalizedData, TimeFrame
from ..collectors.multi_exchange import Exchange, UnifiedMarketData

# Configuration du logger
logger = logging.getLogger(__name__)


class FeatureCategory(Enum):
    """Catégories de features"""
    PRICE = "price"
    VOLUME = "volume"
    VOLATILITY = "volatility"
    MICROSTRUCTURE = "microstructure"
    PATTERN = "pattern"
    SENTIMENT = "sentiment"
    ARBITRAGE = "arbitrage"
    REGIME = "regime"


@dataclass
class FeatureSet:
    """Ensemble de features pour ML/DRL"""
    timestamp: int
    symbol: str
    
    # Features organisées par catégorie
    price_features: Dict[str, float] = field(default_factory=dict)
    volume_features: Dict[str, float] = field(default_factory=dict)
    volatility_features: Dict[str, float] = field(default_factory=dict)
    microstructure_features: Dict[str, float] = field(default_factory=dict)
    pattern_features: Dict[str, float] = field(default_factory=dict)
    arbitrage_features: Dict[str, float] = field(default_factory=dict)
    regime_features: Dict[str, float] = field(default_factory=dict)
    
    # Métadonnées
    quality_score: float = 1.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    def to_numpy(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Convertir en array numpy pour ML"""
        all_features = {}
        all_features.update(self.price_features)
        all_features.update(self.volume_features)
        all_features.update(self.volatility_features)
        all_features.update(self.microstructure_features)
        all_features.update(self.pattern_features)
        all_features.update(self.arbitrage_features)
        all_features.update(self.regime_features)
        
        if feature_names:
            return np.array([all_features.get(name, np.nan) for name in feature_names])
        else:
            return np.array(list(all_features.values()))
    
    def get_feature_names(self) -> List[str]:
        """Obtenir tous les noms de features"""
        all_features = {}
        for attr_name in ['price_features', 'volume_features', 'volatility_features',
                         'microstructure_features', 'pattern_features', 
                         'arbitrage_features', 'regime_features']:
            features_dict = getattr(self, attr_name)
            all_features.update(features_dict)
        return list(all_features.keys())


class FeatureEngineer:
    """
    Générateur de features avancées pour le trading algorithmique.
    Optimisé pour les stratégies DRL, arbitrage statistique et market making.
    """
    
    def __init__(self, 
                 lookback_periods: List[int] = [5, 10, 20, 50, 100],
                 feature_selection: bool = True,
                 max_features: int = 100):
        """
        Initialisation du feature engineer
        
        Args:
            lookback_periods: Périodes pour les calculs roulants
            feature_selection: Activer la sélection automatique de features
            max_features: Nombre max de features après sélection
        """
        self.lookback_periods = lookback_periods
        self.feature_selection = feature_selection
        self.max_features = max_features
        
        # Cache pour les calculs coûteux
        self.cache = {}
        
        # Importance des features (mise à jour dynamiquement)
        self.feature_importance = {}
        
        # Sélecteur de features
        self.feature_selector = None
        if feature_selection:
            self.feature_selector = SelectKBest(f_classif, k=max_features)
    
    def engineer_features(self, df: pd.DataFrame, 
                         target_col: Optional[str] = None,
                         include_categories: Optional[List[FeatureCategory]] = None) -> pd.DataFrame:
        """
        Générer toutes les features pour un DataFrame
        
        Args:
            df: DataFrame avec données OHLCV normalisées
            target_col: Colonne cible pour la sélection de features
            include_categories: Catégories de features à inclure
            
        Returns:
            DataFrame avec toutes les features
        """
        if include_categories is None:
            include_categories = list(FeatureCategory)
        
        # Copier pour éviter les modifications
        df_features = df.copy()
        
        # Générer les features par catégorie
        if FeatureCategory.PRICE in include_categories:
            df_features = self._generate_price_features(df_features)
        
        if FeatureCategory.VOLUME in include_categories:
            df_features = self._generate_volume_features(df_features)
        
        if FeatureCategory.VOLATILITY in include_categories:
            df_features = self._generate_volatility_features(df_features)
        
        if FeatureCategory.MICROSTRUCTURE in include_categories:
            df_features = self._generate_microstructure_features(df_features)
        
        if FeatureCategory.PATTERN in include_categories:
            df_features = self._generate_pattern_features(df_features)
        
        if FeatureCategory.ARBITRAGE in include_categories:
            df_features = self._generate_arbitrage_features(df_features)
        
        if FeatureCategory.REGIME in include_categories:
            df_features = self._generate_regime_features(df_features)
        
        # Nettoyer les NaN
        df_features = self._handle_missing_features(df_features)
        
        # Sélection de features si demandée
        if self.feature_selection and target_col and target_col in df_features.columns:
            df_features = self._select_best_features(df_features, target_col)
        
        return df_features
    
    def engineer_state_for_rl(self, df: pd.DataFrame, 
                            lookback: int = 50,
                            step_size: int = 1) -> np.ndarray:
        """
        Créer les états pour l'environnement RL
        
        Args:
            df: DataFrame avec features
            lookback: Nombre de pas de temps dans l'état
            step_size: Pas entre les échantillons
            
        Returns:
            Array 3D (samples, timesteps, features) pour LSTM/CNN
        """
        # Sélectionner les features importantes pour RL
        rl_features = [
            'close', 'volume', 'returns', 'volatility_20',
            'rsi_14', 'macd_signal', 'bb_position',
            'bid_ask_spread_pct', 'order_flow_imbalance',
            'vpin', 'regime_probability'
        ]
        
        # Filtrer les features disponibles
        available_features = [f for f in rl_features if f in df.columns]
        
        # Créer les séquences
        sequences = []
        for i in range(lookback, len(df), step_size):
            sequence = df.iloc[i-lookback:i][available_features].values
            sequences.append(sequence)
        
        return np.array(sequences)
    
    def _generate_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Générer les features de prix avancées"""
        
        # Moyennes mobiles adaptatives
        for period in self.lookback_periods:
            # KAMA (Kaufman Adaptive Moving Average)
            df[f'kama_{period}'] = talib.KAMA(df['close'].values, timeperiod=period)
            
            # Distance à KAMA
            df[f'close_to_kama_{period}'] = (df['close'] - df[f'kama_{period}']) / df[f'kama_{period}']
        
        # MACD avec paramètres multiples
        for fast, slow in [(12, 26), (5, 35)]:
            macd, signal, hist = talib.MACD(df['close'].values, 
                                           fastperiod=fast, 
                                           slowperiod=slow, 
                                           signalperiod=9)
            df[f'macd_{fast}_{slow}'] = macd
            df[f'macd_signal_{fast}_{slow}'] = signal
            df[f'macd_hist_{fast}_{slow}'] = hist
            df[f'macd_cross_{fast}_{slow}'] = np.where(macd > signal, 1, -1)
        
        # Ichimoku Cloud
        high_9 = df['high'].rolling(9).max()
        low_9 = df['low'].rolling(9).min()
        high_26 = df['high'].rolling(26).max()
        low_26 = df['low'].rolling(26).min()
        high_52 = df['high'].rolling(52).max()
        low_52 = df['low'].rolling(52).min()
        
        df['ichimoku_tenkan'] = (high_9 + low_9) / 2
        df['ichimoku_kijun'] = (high_26 + low_26) / 2
        df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
        df['ichimoku_senkou_b'] = ((high_52 + low_52) / 2).shift(26)
        df['ichimoku_chikou'] = df['close'].shift(-26)
        
        # Position relative dans le nuage
        df['ichimoku_cloud_distance'] = df['close'] - (df['ichimoku_senkou_a'] + df['ichimoku_senkou_b']) / 2
        
        # Fibonacci Retracements dynamiques
        for period in [20, 50]:
            high_p = df['high'].rolling(period).max()
            low_p = df['low'].rolling(period).min()
            diff = high_p - low_p
            
            df[f'fib_0.236_{period}'] = high_p - 0.236 * diff
            df[f'fib_0.382_{period}'] = high_p - 0.382 * diff
            df[f'fib_0.5_{period}'] = high_p - 0.5 * diff
            df[f'fib_0.618_{period}'] = high_p - 0.618 * diff
            
            # Distance au niveau Fibonacci le plus proche
            fib_levels = df[[f'fib_0.236_{period}', f'fib_0.382_{period}', 
                            f'fib_0.5_{period}', f'fib_0.618_{period}']].values
            df[f'fib_distance_{period}'] = np.min(np.abs(df['close'].values[:, np.newaxis] - fib_levels), axis=1)
        
        # Pivot Points
        df['pivot'] = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        df['pivot_r1'] = 2 * df['pivot'] - df['low'].shift(1)
        df['pivot_s1'] = 2 * df['pivot'] - df['high'].shift(1)
        df['pivot_r2'] = df['pivot'] + (df['high'].shift(1) - df['low'].shift(1))
        df['pivot_s2'] = df['pivot'] - (df['high'].shift(1) - df['low'].shift(1))
        
        # Heikin-Ashi
        ha_close = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        ha_open = (df['open'].shift(1) + df['close'].shift(1)) / 2
        ha_high = df[['high', 'open', 'close']].max(axis=1)
        ha_low = df[['low', 'open', 'close']].min(axis=1)
        
        df['ha_trend'] = np.where(ha_close > ha_open, 1, -1)
        df['ha_body_size'] = np.abs(ha_close - ha_open) / df['close']
        
        return df
    
    def _generate_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Générer les features de volume avancées"""
        
        # Volume Price Trend (VPT)
        df['vpt'] = (df['volume'] * df['returns']).cumsum()
        
        # On Balance Volume (OBV)
        df['obv'] = (np.sign(df['returns']) * df['volume']).cumsum()
        
        # Accumulation/Distribution Line
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        df['adl'] = (clv * df['volume']).cumsum()
        
        # Money Flow Index (MFI)
        for period in [14, 28]:
            df[f'mfi_{period}'] = talib.MFI(df['high'].values, df['low'].values, 
                                           df['close'].values, df['volume'].values, 
                                           timeperiod=period)
        
        # Chaikin Money Flow (CMF)
        for period in [20]:
            mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            mfv = mfv.fillna(0) * df['volume']
            df[f'cmf_{period}'] = mfv.rolling(period).sum() / df['volume'].rolling(period).sum()
        
        # Volume Rate of Change
        for period in self.lookback_periods:
            df[f'volume_roc_{period}'] = df['volume'].pct_change(period)
        
        # Volume Weighted Average Price (VWAP) avec bandes
        df['vwap'] = (df['volume'] * df['close']).cumsum() / df['volume'].cumsum()
        vwap_std = ((df['close'] - df['vwap']) ** 2).rolling(20).mean() ** 0.5
        df['vwap_upper'] = df['vwap'] + 2 * vwap_std
        df['vwap_lower'] = df['vwap'] - 2 * vwap_std
        df['close_to_vwap'] = (df['close'] - df['vwap']) / df['vwap']
        
        # Force Index
        df['force_index'] = df['returns'] * df['volume']
        for period in [13, 20]:
            df[f'force_index_ema_{period}'] = df['force_index'].ewm(span=period).mean()
        
        # Volume Profile (simplified)
        for period in [20, 50]:
            # Calculer les niveaux de prix importants basés sur le volume
            price_bins = pd.qcut(df['close'].rolling(period).mean(), q=10, duplicates='drop')
            volume_profile = df.groupby(price_bins)['volume'].rolling(period).sum()
            df[f'volume_profile_poc_{period}'] = volume_profile.transform('max')  # Point of Control
        
        return df
    
    def _generate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Générer les features de volatilité avancées"""
        
        # Volatilité réalisée avec différentes méthodes
        for period in self.lookback_periods:
            # Parkinson volatility (utilise high-low)
            hl_ratio = np.log(df['high'] / df['low'])
            df[f'parkinson_vol_{period}'] = np.sqrt(
                (1 / (4 * np.log(2))) * (hl_ratio ** 2).rolling(period).mean()
            ) * np.sqrt(252)
            
            # Garman-Klass volatility
            cc = np.log(df['close'] / df['close'].shift(1)) ** 2
            hl = (np.log(df['high'] / df['low'])) ** 2
            df[f'garman_klass_vol_{period}'] = np.sqrt(
                ((0.5 * hl - 0.39 * cc).rolling(period).mean())
            ) * np.sqrt(252)
            
            # Rogers-Satchell volatility
            rs = np.log(df['high'] / df['close']) * np.log(df['high'] / df['open']) + \
                 np.log(df['low'] / df['close']) * np.log(df['low'] / df['open'])
            df[f'rogers_satchell_vol_{period}'] = np.sqrt(rs.rolling(period).mean()) * np.sqrt(252)
        
        # Average True Range (ATR) et dérivés
        for period in [14, 20]:
            df[f'atr_{period}'] = talib.ATR(df['high'].values, df['low'].values, 
                                           df['close'].values, timeperiod=period)
            df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['close']
            
            # Chandelier Exit
            df[f'chandelier_long_{period}'] = df['high'].rolling(period).max() - df[f'atr_{period}'] * 3
            df[f'chandelier_short_{period}'] = df['low'].rolling(period).min() + df[f'atr_{period}'] * 3
        
        # Volatility Ratio
        for period in [10, 20]:
            true_range = np.maximum(df['high'] - df['low'], 
                                  np.abs(df['high'] - df['close'].shift(1)),
                                  np.abs(df['low'] - df['close'].shift(1)))
            df[f'volatility_ratio_{period}'] = true_range / true_range.rolling(period).mean()
        
        # Keltner Channels
        for period in [20]:
            ma = df['close'].rolling(period).mean()
            atr = talib.ATR(df['high'].values, df['low'].values, 
                           df['close'].values, timeperiod=period)
            df[f'keltner_upper_{period}'] = ma + 2 * atr
            df[f'keltner_lower_{period}'] = ma - 2 * atr
            df[f'keltner_position_{period}'] = (df['close'] - df[f'keltner_lower_{period}']) / \
                                              (df[f'keltner_upper_{period}'] - df[f'keltner_lower_{period}'])
        
        # Volatility Smile (pour options si disponible)
        # Simplified version using ATM implied vol proxy
        df['volatility_smile_skew'] = (df['volatility_20'] - df['volatility_20'].rolling(50).mean()) / \
                                     df['volatility_20'].rolling(50).std()
        
        return df
    
    def _generate_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Générer les features de microstructure de marché"""
        
        # Order Flow Imbalance (OFI)
        if all(col in df.columns for col in ['bid_volume', 'ask_volume']):
            df['order_flow_imbalance'] = (df['bid_volume'] - df['ask_volume']) / \
                                        (df['bid_volume'] + df['ask_volume'] + 1)
            
            # OFI cumulé
            df['ofi_cumulative'] = df['order_flow_imbalance'].cumsum()
            
            # OFI momentum
            for period in [5, 10, 20]:
                df[f'ofi_momentum_{period}'] = df['order_flow_imbalance'].rolling(period).mean()
        
        # Kyle's Lambda (impact de prix)
        if 'volume' in df.columns and 'returns' in df.columns:
            for period in [20, 50]:
                # Régression rolling pour estimer lambda
                def kyle_lambda(window):
                    if len(window) < period:
                        return np.nan
                    X = window['volume'].values.reshape(-1, 1)
                    y = np.abs(window['returns'].values)
                    if np.std(X) == 0 or np.std(y) == 0:
                        return 0
                    return np.corrcoef(X.flatten(), y)[0, 1] * np.std(y) / np.std(X)
                
                df[f'kyle_lambda_{period}'] = df.rolling(period).apply(
                    lambda x: kyle_lambda(df.loc[x.index]), raw=False
                )
        
        # VPIN (Volume-synchronized Probability of Informed Trading)
        if 'volume' in df.columns:
            # Simplified VPIN
            bucket_size = df['volume'].rolling(50).mean()
            buy_volume = df['volume'] * (df['returns'] > 0).astype(float)
            sell_volume = df['volume'] * (df['returns'] < 0).astype(float)
            
            df['vpin'] = np.abs(buy_volume - sell_volume).rolling(50).sum() / \
                        df['volume'].rolling(50).sum().clip(lower=1)
        
        # Effective Spread
        if all(col in df.columns for col in ['bid', 'ask']):
            mid_price = (df['bid'] + df['ask']) / 2
            df['effective_spread'] = 2 * np.abs(df['close'] - mid_price) / mid_price
            df['realized_spread'] = 2 * df['returns'].abs() - df['effective_spread']
        
        # Quote Intensity (changements de bid/ask)
        if all(col in df.columns for col in ['bid', 'ask']):
            df['quote_intensity'] = (df['bid'].diff() != 0).astype(int) + \
                                   (df['ask'].diff() != 0).astype(int)
            df['quote_intensity_ma'] = df['quote_intensity'].rolling(20).mean()
        
        # Microstructure Noise
        for period in [5, 10]:
            # Variance ratio test pour le bruit
            returns_1 = df['returns']
            returns_n = df['close'].pct_change(period)
            variance_ratio = (returns_n.var() / period) / returns_1.var()
            df[f'microstructure_noise_{period}'] = np.abs(1 - variance_ratio)
        
        # Trades per minute (si disponible)
        if 'trade_count' in df.columns:
            df['trade_intensity'] = df['trade_count'].rolling(10).mean()
            df['trade_intensity_zscore'] = (df['trade_count'] - df['trade_intensity']) / \
                                          df['trade_count'].rolling(50).std()
        
        return df
    
    def _generate_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Générer les features de patterns et formations"""
        
        # Candlestick patterns avec TA-Lib
        pattern_functions = {
            'doji': talib.CDLDOJI,
            'hammer': talib.CDLHAMMER,
            'shooting_star': talib.CDLSHOOTINGSTAR,
            'engulfing': talib.CDLENGULFING,
            'morning_star': talib.CDLMORNINGSTAR,
            'evening_star': talib.CDLEVENINGSTAR,
            'three_white_soldiers': talib.CDL3WHITESOLDIERS,
            'three_black_crows': talib.CDL3BLACKCROWS,
            'spinning_top': talib.CDLSPINNINGTOP,
            'marubozu': talib.CDLMARUBOZU
        }
        
        for name, func in pattern_functions.items():
            df[f'cdl_{name}'] = func(df['open'].values, df['high'].values, 
                                    df['low'].values, df['close'].values)
        
        # Support et Résistance dynamiques
        for period in [20, 50, 100]:
            # Utiliser les fractales pour identifier S/R
            df[f'resistance_{period}'] = df['high'].rolling(period).max()
            df[f'support_{period}'] = df['low'].rolling(period).min()
            
            # Distance aux niveaux S/R
            df[f'dist_to_resistance_{period}'] = (df[f'resistance_{period}'] - df['close']) / df['close']
            df[f'dist_to_support_{period}'] = (df['close'] - df[f'support_{period}']) / df['close']
            
            # Nombre de touches des niveaux S/R
            tolerance = 0.001  # 0.1%
            df[f'resistance_touches_{period}'] = (
                (np.abs(df['high'] - df[f'resistance_{period}']) / df[f'resistance_{period}']) < tolerance
            ).rolling(period).sum()
            df[f'support_touches_{period}'] = (
                (np.abs(df['low'] - df[f'support_{period}']) / df[f'support_{period}']) < tolerance
            ).rolling(period).sum()
        
        # Chart Patterns détection simplifiée
        # Head and Shoulders
        for period in [20, 40]:
            window = period
            # Identifier les pics locaux
            highs = df['high'].rolling(window).max()
            
            # Pattern detection logic (simplified)
            left_shoulder = highs.shift(window)
            head = highs.shift(window // 2)
            right_shoulder = highs
            
            df[f'head_shoulders_score_{period}'] = np.where(
                (head > left_shoulder * 1.02) & (head > right_shoulder * 1.02) &
                (np.abs(left_shoulder - right_shoulder) / left_shoulder < 0.02),
                1, 0
            )
        
        # Triangle patterns
        for period in [30, 60]:
            high_slope = df['high'].rolling(period).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else np.nan
            )
            low_slope = df['low'].rolling(period).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else np.nan
            )
            
            # Converging triangle
            df[f'triangle_converging_{period}'] = np.where(
                (high_slope < 0) & (low_slope > 0), 1, 0
            )
            
            # Ascending triangle
            df[f'triangle_ascending_{period}'] = np.where(
                (np.abs(high_slope) < 0.0001) & (low_slope > 0), 1, 0
            )
        
        # Wave patterns (Elliott Wave proxy)
        for period in [5, 8, 13]:  # Fibonacci numbers
            df[f'wave_count_{period}'] = (
                (df['close'].diff() > 0).astype(int).diff().abs()
            ).rolling(period).sum()
        
        return df
    
    def _generate_arbitrage_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Générer les features spécifiques à l'arbitrage"""
        
        # Si nous avons des données multi-exchanges
        exchange_cols = [col for col in df.columns if any(
            ex in col for ex in ['binance', 'ib', 'kraken', 'coinbase']
        )]
        
        if len(exchange_cols) >= 2:
            # Spreads inter-exchanges
            close_cols = [col for col in exchange_cols if 'close' in col]
            if len(close_cols) >= 2:
                # Calculer tous les spreads possibles
                for i, col1 in enumerate(close_cols):
                    for col2 in close_cols[i+1:]:
                        spread_name = f'spread_{col1}_{col2}'
                        df[spread_name] = df[col1] - df[col2]
                        df[f'{spread_name}_pct'] = (df[spread_name] / df[col2]) * 100
                        
                        # Z-score du spread pour mean reversion
                        df[f'{spread_name}_zscore'] = (
                            df[spread_name] - df[spread_name].rolling(50).mean()
                        ) / df[spread_name].rolling(50).std()
                
                # Opportunités d'arbitrage
                df['max_spread_pct'] = df[[col for col in df.columns if 'spread_' in col and '_pct' in col]].max(axis=1)
                df['arbitrage_signal'] = (df['max_spread_pct'] > 0.1).astype(int)  # 0.1% threshold
        
        # Co-intégration features (pour pairs trading)
        if 'returns' in df.columns:
            # Autocorrélation pour détecter mean reversion
            for lag in [1, 5, 10]:
                df[f'returns_autocorr_{lag}'] = df['returns'].rolling(50).apply(
                    lambda x: x.autocorr(lag=lag) if len(x) >= lag + 1 else np.nan
                )
        
        # Spread vs volatilité (opportunités ajustées au risque)
        if 'volatility_20' in df.columns and 'max_spread_pct' in df.columns:
            df['risk_adjusted_spread'] = df['max_spread_pct'] / (df['volatility_20'] + 0.001)
            df['sharpe_spread'] = df['risk_adjusted_spread'].rolling(20).mean() / \
                                 df['risk_adjusted_spread'].rolling(20).std()
        
        # Lead-lag relationships
        if len(close_cols) >= 2:
            for col in close_cols:
                # Corrélation avec les autres exchanges à différents lags
                for lag in [1, 2, 5]:
                    other_cols = [c for c in close_cols if c != col]
                    if other_cols:
                        df[f'{col}_lead_{lag}'] = df[col].shift(-lag).rolling(20).corr(df[other_cols[0]])
        
        # Liquidité relative entre exchanges
        volume_cols = [col for col in exchange_cols if 'volume' in col]
        if len(volume_cols) >= 2:
            total_volume = df[volume_cols].sum(axis=1)
            for col in volume_cols:
                df[f'{col}_share'] = df[col] / (total_volume + 1)
            
            # Concentration de liquidité (Herfindahl index)
            volume_shares = df[[col for col in df.columns if '_share' in col]]
            df['liquidity_concentration'] = (volume_shares ** 2).sum(axis=1)
        
        return df
    
    def _generate_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Générer les features de régime de marché"""
        
        # Détection de tendance
        for period in [20, 50, 100]:
            # ADX (Average Directional Index)
            df[f'adx_{period}'] = talib.ADX(df['high'].values, df['low'].values, 
                                           df['close'].values, timeperiod=period)
            
            # Régression linéaire pour la force de tendance
            df[f'trend_strength_{period}'] = df['close'].rolling(period).apply(
                lambda x: np.abs(np.polyfit(range(len(x)), x, 1)[0]) / np.std(x) 
                if len(x) == period and np.std(x) > 0 else 0
            )
        
        # Régime de volatilité
        vol_regimes = pd.qcut(df['volatility_20'].dropna(), q=3, labels=['low', 'medium', 'high'])
        df['volatility_regime'] = vol_regimes.cat.codes.reindex(df.index)
        
        # Changements de régime
        df['volatility_regime_change'] = df['volatility_regime'].diff().abs()
        
        # Markov Regime Switching proxy
        # Utiliser les returns pour identifier 2 régimes (bull/bear)
        returns_ma = df['returns'].rolling(20).mean()
        returns_std = df['returns'].rolling(20).std()
        
        # Régime haussier si returns > 0 et volatilité faible
        df['bull_regime_score'] = np.where(
            (returns_ma > 0) & (returns_std < df['volatility_20'].median()),
            1, 0
        )
        
        # Probabilité de régime (lissée)
        df['regime_probability'] = df['bull_regime_score'].rolling(50).mean()
        
        # Durée du régime actuel
        regime_changes = (df['bull_regime_score'].diff() != 0).cumsum()
        df['regime_duration'] = df.groupby(regime_changes).cumcount()
        
        # Market state features
        # Trending vs Ranging
        for period in [20, 50]:
            # Efficiency Ratio (Kaufman)
            direction = np.abs(df['close'] - df['close'].shift(period))
            volatility = df['close'].diff().abs().rolling(period).sum()
            df[f'efficiency_ratio_{period}'] = direction / (volatility + 1e-10)
            
            # Market is trending if ER > threshold
            df[f'is_trending_{period}'] = (df[f'efficiency_ratio_{period}'] > 0.3).astype(int)
        
        # Fractal Dimension (mesure de complexité)
        for period in [30]:
            def hurst_exponent(ts):
                """Calcul simplifié de l'exposant de Hurst"""
                if len(ts) < period:
                    return 0.5
                
                lags = range(2, min(20, len(ts) // 2))
                tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
                
                if not tau or np.std(np.log(lags)) == 0:
                    return 0.5
                
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                return poly[0] * 2.0
            
            df[f'hurst_exponent_{period}'] = df['close'].rolling(period).apply(
                hurst_exponent, raw=False
            )
            
            # Fractal dimension = 2 - H
            df[f'fractal_dimension_{period}'] = 2 - df[f'hurst_exponent_{period}']
        
        # Cycle detection (utiliser DFT)
        for period in [50, 100]:
            def dominant_cycle(prices):
                if len(prices) < period:
                    return np.nan
                
                # Detrend
                detrended = prices - np.linspace(prices.iloc[0], prices.iloc[-1], len(prices))
                
                # FFT
                fft = np.fft.fft(detrended)
                freqs = np.fft.fftfreq(len(detrended))
                
                # Trouver la fréquence dominante (exclure DC)
                idx = np.argmax(np.abs(fft[1:len(fft)//2])) + 1
                
                if freqs[idx] > 0:
                    return 1 / freqs[idx]
                return np.nan
            
            df[f'dominant_cycle_{period}'] = df['close'].rolling(period).apply(
                dominant_cycle, raw=False
            )
        
        return df
    
    def _handle_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gérer les valeurs manquantes dans les features"""
        
        # Forward fill pour la plupart des features
        df = df.fillna(method='ffill', limit=5)
        
        # Pour les features de pourcentage, remplacer par 0
        percentage_cols = [col for col in df.columns if any(
            x in col for x in ['_pct', '_ratio', 'returns', 'zscore']
        )]
        df[percentage_cols] = df[percentage_cols].fillna(0)
        
        # Pour les indicateurs, utiliser la valeur neutre
        indicator_cols = [col for col in df.columns if any(
            x in col for x in ['rsi', 'mfi', 'adx']
        )]
        for col in indicator_cols:
            if 'rsi' in col or 'mfi' in col:
                df[col] = df[col].fillna(50)  # Valeur neutre pour RSI/MFI
            elif 'adx' in col:
                df[col] = df[col].fillna(25)  # Pas de tendance forte
        
        # Remplacer les infinis
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Dernière passe: remplacer les NaN restants par 0
        df = df.fillna(0)
        
        return df
    
    def _select_best_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Sélectionner les meilleures features pour la prédiction"""
        
        # Séparer features et target
        feature_cols = [col for col in df.columns if col not in [
            target_col, 'symbol', 'timestamp', 'exchange'
        ]]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Enlever les lignes avec target NaN
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
        
        if len(X) < 100:  # Pas assez de données
            return df
        
        try:
            # Sélection des K meilleures features
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Obtenir les noms des features sélectionnées
            selected_features = [feature_cols[i] for i in self.feature_selector.get_support(indices=True)]
            
            # Mettre à jour l'importance des features
            scores = self.feature_selector.scores_
            for i, col in enumerate(feature_cols):
                self.feature_importance[col] = scores[i]
            
            # Retourner seulement les colonnes sélectionnées + métadonnées
            keep_cols = selected_features + [target_col, 'symbol', 'timestamp']
            if 'exchange' in df.columns:
                keep_cols.append('exchange')
            
            logger.info(f"Features sélectionnées: {len(selected_features)}/{len(feature_cols)}")
            
            return df[keep_cols]
            
        except Exception as e:
            logger.warning(f"Erreur sélection features: {e}")
            return df
    
    def get_feature_importance_report(self) -> pd.DataFrame:
        """Obtenir un rapport sur l'importance des features"""
        if not self.feature_importance:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame(
            list(self.feature_importance.items()),
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False)
        
        # Ajouter la catégorie
        def get_category(feature_name):
            if any(x in feature_name for x in ['price', 'close', 'open', 'high', 'low', 'ma', 'ema']):
                return 'price'
            elif any(x in feature_name for x in ['volume', 'obv', 'vpt', 'mfi']):
                return 'volume'
            elif any(x in feature_name for x in ['volatility', 'atr', 'std']):
                return 'volatility'
            elif any(x in feature_name for x in ['spread', 'arbitrage', 'cross_exchange']):
                return 'arbitrage'
            elif any(x in feature_name for x in ['regime', 'trend', 'cycle']):
                return 'regime'
            elif any(x in feature_name for x in ['bid', 'ask', 'microstructure', 'ofi', 'vpin']):
                return 'microstructure'
            else:
                return 'other'
        
        importance_df['category'] = importance_df['feature'].apply(get_category)
        
        return importance_df


# Exemple d'utilisation
async def main():
    """Exemple d'utilisation du feature engineer"""
    # Créer des données de test
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='1h')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 101,
        'low': np.random.randn(len(dates)).cumsum() + 99,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.exponential(1000, len(dates)),
        'bid': np.random.randn(len(dates)).cumsum() + 99.9,
        'ask': np.random.randn(len(dates)).cumsum() + 100.1,
        'bid_volume': np.random.exponential(100, len(dates)),
        'ask_volume': np.random.exponential(100, len(dates))
    })
    df.set_index('timestamp', inplace=True)
    
    # Ajouter des colonnes multi-exchange
    df['binance_close'] = df['close'] + np.random.normal(0, 0.1, len(df))
    df['ib_close'] = df['close'] + np.random.normal(0.05, 0.1, len(df))
    
    # Créer le feature engineer
    engineer = FeatureEngineer(
        lookback_periods=[5, 10, 20, 50],
        feature_selection=False  # Désactivé pour l'exemple
    )
    
    # Générer toutes les features
    df_features = engineer.engineer_features(df)
    
    logger.info(f"Features générées: {df_features.shape[1]} colonnes")
    logger.info(f"Catégories de features: {set([col.split('_')[0] for col in df_features.columns])}")
    
    # Créer les états pour RL
    states = engineer.engineer_state_for_rl(df_features, lookback=20)
    logger.info(f"États RL créés: {states.shape}")
    
    # Afficher quelques features importantes
    important_features = [
        'rsi_14', 'macd_signal_12_26', 'volatility_20',
        'order_flow_imbalance', 'vpin', 'regime_probability',
        'arbitrage_signal', 'max_spread_pct'
    ]
    
    for feature in important_features:
        if feature in df_features.columns:
            logger.info(f"{feature}: {df_features[feature].iloc[-1]:.4f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())