"""
Multi-Asset Trading Environment - Environnement RL pour Trading Multi-Actifs
===========================================================================

Ce module implémente un environnement de trading multi-actifs compatible
OpenAI Gym pour l'entraînement d'agents de reinforcement learning. Supporte
le trading simultané de multiples actifs avec gestion de portefeuille,
corrélations et contraintes réalistes.

Caractéristiques:
- Actions continues : allocation de portefeuille par actif
- Observations riches : prix, volumes, indicateurs, corrélations
- Rewards sophistiqués : Sharpe ratio, drawdown penalties
- Frais de transaction et slippage réalistes
- Support backtesting et trading live
- Métriques de performance complètes
- Gestion des corrélations inter-actifs

Architecture:
- Compatible Stable Baselines3 et RLlib
- Vectorisé pour entraînement parallèle
- State normalization automatique
- Action masking pour contraintes
- Render avec visualisations avancées

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

# Imports internes
from utils.logger import get_structured_logger
from utils.metrics import MetricsCalculator, PerformanceMetrics
from ml.features.technical_indicators import TechnicalIndicators
from data.storage.timeseries_db import TimescaleDBStorage

warnings.filterwarnings('ignore')


class ActionType(Enum):
    """Types d'actions supportées"""
    DISCRETE = "discrete"           # Buy/Hold/Sell par actif
    CONTINUOUS = "continuous"       # Allocation continue [0, 1]
    MIXED = "mixed"                # Discrete + continuous


class RewardType(Enum):
    """Types de fonctions de récompense"""
    SIMPLE_RETURN = "simple_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    RISK_ADJUSTED = "risk_adjusted"
    DIFFERENTIAL_SHARPE = "differential_sharpe"


@dataclass
class TradingFees:
    """Structure des frais de trading"""
    maker_fee: float = 0.001       # 0.1%
    taker_fee: float = 0.001       # 0.1%
    slippage: float = 0.0005       # 0.05%
    min_trade_size: float = 10.0   # Taille minimum en USD
    
    def calculate_cost(self, trade_value: float, is_maker: bool = False) -> float:
        """Calcule le coût total d'un trade"""
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        fee_cost = abs(trade_value) * fee_rate
        slippage_cost = abs(trade_value) * self.slippage
        return fee_cost + slippage_cost


@dataclass
class PortfolioState:
    """État du portefeuille"""
    cash: float                              # Cash disponible
    positions: Dict[str, float]              # Quantités par actif
    values: Dict[str, float]                 # Valeurs par actif
    total_value: float                       # Valeur totale
    weights: Dict[str, float]                # Poids du portefeuille
    returns: float = 0.0                     # Rendement de la période
    cumulative_returns: float = 0.0          # Rendement cumulé
    
    def update_values(self, prices: Dict[str, float]) -> None:
        """Met à jour les valeurs du portefeuille"""
        self.values = {
            asset: self.positions.get(asset, 0) * prices.get(asset, 0)
            for asset in prices
        }
        position_value = sum(self.values.values())
        self.total_value = self.cash + position_value
        
        # Calculer les poids
        if self.total_value > 0:
            self.weights = {
                asset: value / self.total_value
                for asset, value in self.values.items()
            }
        else:
            self.weights = {asset: 0.0 for asset in self.values}


class MultiAssetTradingEnv(gym.Env):
    """
    Environnement de trading multi-actifs pour Reinforcement Learning
    Compatible avec Stable Baselines3 et autres frameworks RL
    """
    
    metadata = {'render.modes': ['human', 'rgb_array', 'terminal']}
    
    def __init__(
        self,
        symbols: List[str],
        data_source: Optional[Union[pd.DataFrame, TimescaleDBStorage]] = None,
        initial_capital: float = 100000.0,
        max_positions: int = 10,
        action_type: ActionType = ActionType.CONTINUOUS,
        reward_type: RewardType = RewardType.SHARPE_RATIO,
        trading_fees: Optional[TradingFees] = None,
        lookback_window: int = 50,
        technical_indicators: Optional[List[str]] = None,
        use_portfolio_features: bool = True,
        use_correlation_features: bool = True,
        normalize_observations: bool = True,
        normalize_rewards: bool = True,
        max_episode_steps: int = 252,  # ~1 année de trading
        random_start: bool = True,
        validation_split: float = 0.2,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        # Configuration de base
        self.symbols = symbols
        self.n_assets = len(symbols)
        self.data_source = data_source
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.action_type = action_type
        self.reward_type = reward_type
        self.trading_fees = trading_fees or TradingFees()
        self.lookback_window = lookback_window
        self.use_portfolio_features = use_portfolio_features
        self.use_correlation_features = use_correlation_features
        self.normalize_observations = normalize_observations
        self.normalize_rewards = normalize_rewards
        self.max_episode_steps = max_episode_steps
        self.random_start = random_start
        self.validation_split = validation_split
        self.config = config or {}
        
        # Logger
        self.logger = get_structured_logger(
            "multi_asset_env",
            module="ml.environments"
        )
        
        # Indicateurs techniques
        self.technical_indicators = technical_indicators or [
            'sma_20', 'sma_50', 'rsi_14', 'macd_signal',
            'bb_upper', 'bb_lower', 'atr_14', 'volume_ratio'
        ]
        self.indicator_calculator = TechnicalIndicators()
        
        # Métriques
        self.metrics_calculator = MetricsCalculator()
        
        # État
        self.current_step = 0
        self.episode_count = 0
        self.portfolio = None
        self.done = False
        
        # Historique
        self.price_history = defaultdict(deque)
        self.portfolio_history = []
        self.action_history = []
        self.reward_history = []
        self.trade_history = []
        
        # Normalisation
        self.obs_mean = None
        self.obs_std = None
        self.reward_mean = 0.0
        self.reward_std = 1.0
        
        # Données
        self._load_and_prepare_data()
        
        # Espaces d'action et d'observation
        self._setup_spaces()
        
        # Random seed
        self.seed()
        
        self.logger.info(
            "multi_asset_env_initialized",
            symbols=symbols,
            action_type=action_type.value,
            reward_type=reward_type.value
        )
    
    def _load_and_prepare_data(self) -> None:
        """Charge et prépare les données de marché"""
        if isinstance(self.data_source, pd.DataFrame):
            self.data = self.data_source
        elif isinstance(self.data_source, TimescaleDBStorage):
            # Charger depuis TimescaleDB
            self._load_from_timescale()
        else:
            # Générer des données synthétiques pour tests
            self._generate_synthetic_data()
        
        # Calculer les indicateurs techniques
        self._calculate_technical_indicators()
        
        # Diviser train/validation
        split_idx = int(len(self.data) * (1 - self.validation_split))
        self.train_data = self.data.iloc[:split_idx]
        self.val_data = self.data.iloc[split_idx:]
        self.current_data = self.train_data  # Par défaut en train
        
        # Calculer les statistiques de normalisation sur train
        if self.normalize_observations:
            self._calculate_normalization_stats()
    
    def _generate_synthetic_data(self) -> None:
        """Génère des données synthétiques pour tests"""
        n_days = 1000
        dates = pd.date_range(end=datetime.now(), periods=n_days, freq='D')
        
        # Générer des prix corrélés
        correlation_matrix = np.random.rand(self.n_assets, self.n_assets)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2
        np.fill_diagonal(correlation_matrix, 1)
        
        # Processus de prix
        returns = np.random.multivariate_normal(
            mean=np.zeros(self.n_assets),
            cov=correlation_matrix * 0.02,  # Volatilité ~14% annuelle
            size=n_days
        )
        
        prices = 100 * np.exp(np.cumsum(returns, axis=0))
        volumes = np.random.lognormal(10, 1, (n_days, self.n_assets))
        
        # Créer DataFrame
        data_dict = {}
        for i, symbol in enumerate(self.symbols):
            data_dict[f"{symbol}_close"] = prices[:, i]
            data_dict[f"{symbol}_volume"] = volumes[:, i]
            data_dict[f"{symbol}_high"] = prices[:, i] * (1 + np.random.uniform(0, 0.02, n_days))
            data_dict[f"{symbol}_low"] = prices[:, i] * (1 - np.random.uniform(0, 0.02, n_days))
        
        self.data = pd.DataFrame(data_dict, index=dates)
        
        self.logger.info("synthetic_data_generated", shape=self.data.shape)
    
    def _calculate_technical_indicators(self) -> None:
        """Calcule les indicateurs techniques pour tous les actifs"""
        for symbol in self.symbols:
            price_col = f"{symbol}_close"
            volume_col = f"{symbol}_volume"
            high_col = f"{symbol}_high"
            low_col = f"{symbol}_low"
            
            if price_col in self.data.columns:
                # Prix et volume
                prices = self.data[price_col]
                volumes = self.data[volume_col] if volume_col in self.data.columns else None
                highs = self.data[high_col] if high_col in self.data.columns else prices
                lows = self.data[low_col] if low_col in self.data.columns else prices
                
                # Calculer chaque indicateur
                for indicator in self.technical_indicators:
                    if indicator.startswith('sma_'):
                        period = int(indicator.split('_')[1])
                        self.data[f"{symbol}_{indicator}"] = prices.rolling(period).mean()
                    
                    elif indicator.startswith('rsi_'):
                        period = int(indicator.split('_')[1])
                        self.data[f"{symbol}_{indicator}"] = self._calculate_rsi(prices, period)
                    
                    elif indicator == 'macd_signal':
                        macd, signal = self._calculate_macd(prices)
                        self.data[f"{symbol}_macd"] = macd
                        self.data[f"{symbol}_macd_signal"] = signal
                    
                    elif indicator == 'bb_upper' or indicator == 'bb_lower':
                        upper, lower = self._calculate_bollinger_bands(prices)
                        self.data[f"{symbol}_bb_upper"] = upper
                        self.data[f"{symbol}_bb_lower"] = lower
                    
                    elif indicator.startswith('atr_'):
                        period = int(indicator.split('_')[1])
                        self.data[f"{symbol}_{indicator}"] = self._calculate_atr(
                            highs, lows, prices, period
                        )
                    
                    elif indicator == 'volume_ratio' and volumes is not None:
                        self.data[f"{symbol}_{indicator}"] = volumes / volumes.rolling(20).mean()
        
        # Calculer les corrélations rolling
        if self.use_correlation_features:
            self._calculate_correlation_features()
        
        # Forward fill NaN values
        self.data.fillna(method='ffill', inplace=True)
        self.data.fillna(0, inplace=True)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcule le RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(
        self, 
        prices: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series]:
        """Calcule MACD et signal"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def _calculate_bollinger_bands(
        self, 
        prices: pd.Series, 
        period: int = 20, 
        std_dev: int = 2
    ) -> Tuple[pd.Series, pd.Series]:
        """Calcule les bandes de Bollinger"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def _calculate_atr(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series, 
        period: int = 14
    ) -> pd.Series:
        """Calcule l'Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def _calculate_correlation_features(self, window: int = 20) -> None:
        """Calcule les features de corrélation entre actifs"""
        # Matrice de corrélation rolling
        price_columns = [f"{symbol}_close" for symbol in self.symbols]
        
        for i in range(len(self.symbols)):
            for j in range(i+1, len(self.symbols)):
                symbol1, symbol2 = self.symbols[i], self.symbols[j]
                col1, col2 = price_columns[i], price_columns[j]
                
                # Corrélation rolling
                corr = self.data[col1].rolling(window).corr(self.data[col2])
                self.data[f"corr_{symbol1}_{symbol2}"] = corr
    
    def _calculate_normalization_stats(self) -> None:
        """Calcule les statistiques de normalisation"""
        # Obtenir toutes les colonnes de features
        feature_cols = []
        for col in self.data.columns:
            if any(col.endswith(f"_{ind}") for ind in self.technical_indicators):
                feature_cols.append(col)
            elif col.startswith("corr_"):
                feature_cols.append(col)
        
        # Calculer mean et std sur les données d'entraînement
        train_features = self.train_data[feature_cols]
        self.obs_mean = train_features.mean().values
        self.obs_std = train_features.std().values
        self.obs_std[self.obs_std == 0] = 1  # Éviter division par zéro
    
    def _setup_spaces(self) -> None:
        """Configure les espaces d'action et d'observation"""
        # Dimension de l'observation
        n_price_features = 4  # open, high, low, close par actif
        n_tech_indicators = len(self.technical_indicators) * self.n_assets
        n_portfolio_features = self.n_assets + 3 if self.use_portfolio_features else 0
        n_corr_features = int(self.n_assets * (self.n_assets - 1) / 2) if self.use_correlation_features else 0
        
        self.observation_dim = (
            n_price_features * self.n_assets +
            n_tech_indicators +
            n_portfolio_features +
            n_corr_features
        ) * self.lookback_window
        
        # Espace d'observation
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )
        
        # Espace d'action selon le type
        if self.action_type == ActionType.DISCRETE:
            # 3 actions par actif : Buy, Hold, Sell
            self.action_space = spaces.MultiDiscrete([3] * self.n_assets)
        elif self.action_type == ActionType.CONTINUOUS:
            # Allocation continue [0, 1] par actif + cash
            self.action_space = spaces.Box(
                low=0,
                high=1,
                shape=(self.n_assets + 1,),  # +1 pour cash
                dtype=np.float32
            )
        else:  # MIXED
            # Discrete (direction) + Continuous (taille)
            self.action_space = spaces.Dict({
                'direction': spaces.MultiDiscrete([3] * self.n_assets),
                'size': spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
            })
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Définit la seed pour la reproductibilité"""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self, validation: bool = False) -> np.ndarray:
        """Réinitialise l'environnement"""
        self.current_step = 0
        self.done = False
        self.episode_count += 1
        
        # Choisir les données (train ou validation)
        self.current_data = self.val_data if validation else self.train_data
        
        # Point de départ aléatoire ou fixe
        if self.random_start and not validation:
            max_start = len(self.current_data) - self.max_episode_steps - self.lookback_window
            self.start_idx = self.np_random.randint(self.lookback_window, max_start)
        else:
            self.start_idx = self.lookback_window
        
        # Réinitialiser le portefeuille
        self.portfolio = PortfolioState(
            cash=self.initial_capital,
            positions={symbol: 0.0 for symbol in self.symbols},
            values={symbol: 0.0 for symbol in self.symbols},
            total_value=self.initial_capital,
            weights={symbol: 0.0 for symbol in self.symbols}
        )
        
        # Réinitialiser les historiques
        self.portfolio_history = [self.portfolio.total_value]
        self.action_history = []
        self.reward_history = []
        self.trade_history = []
        
        for symbol in self.symbols:
            self.price_history[symbol] = deque(maxlen=self.lookback_window)
        
        # Obtenir l'observation initiale
        obs = self._get_observation()
        
        self.logger.debug(
            "env_reset",
            episode=self.episode_count,
            validation=validation,
            start_idx=self.start_idx
        )
        
        return obs
    
    def step(self, action: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute une action dans l'environnement"""
        if self.done:
            raise ValueError("Episode terminé. Appelez reset().")
        
        # Sauvegarder l'état précédent
        prev_portfolio_value = self.portfolio.total_value
        
        # Obtenir les prix actuels
        current_prices = self._get_current_prices()
        
        # Exécuter l'action
        trades = self._execute_action(action, current_prices)
        
        # Mettre à jour le portefeuille
        self.portfolio.update_values(current_prices)
        
        # Calculer la récompense
        reward = self._calculate_reward(prev_portfolio_value)
        
        # Normaliser la récompense si configuré
        if self.normalize_rewards:
            reward = self._normalize_reward(reward)
        
        # Enregistrer dans l'historique
        self.portfolio_history.append(self.portfolio.total_value)
        self.action_history.append(action)
        self.reward_history.append(reward)
        if trades:
            self.trade_history.extend(trades)
        
        # Avancer d'un pas
        self.current_step += 1
        
        # Vérifier si l'épisode est terminé
        self.done = self._is_done()
        
        # Obtenir la nouvelle observation
        obs = self._get_observation()
        
        # Informations supplémentaires
        info = self._get_info()
        
        return obs, reward, self.done, info
    
    def _get_observation(self) -> np.ndarray:
        """Construit le vecteur d'observation"""
        obs_components = []
        
        # Obtenir la fenêtre de données
        current_idx = self.start_idx + self.current_step
        window_start = current_idx - self.lookback_window + 1
        window_end = current_idx + 1
        window_data = self.current_data.iloc[window_start:window_end]
        
        # Features de prix pour chaque actif
        for symbol in self.symbols:
            # OHLCV normalisés
            close_col = f"{symbol}_close"
            if close_col in window_data.columns:
                closes = window_data[close_col].values
                opens = window_data.get(f"{symbol}_open", closes).values
                highs = window_data.get(f"{symbol}_high", closes).values
                lows = window_data.get(f"{symbol}_low", closes).values
                
                # Normaliser par rapport au dernier close
                last_close = closes[-1] if closes[-1] != 0 else 1
                obs_components.extend(opens / last_close)
                obs_components.extend(highs / last_close)
                obs_components.extend(lows / last_close)
                obs_components.extend(closes / last_close)
        
        # Indicateurs techniques
        for symbol in self.symbols:
            for indicator in self.technical_indicators:
                col = f"{symbol}_{indicator}"
                if col in window_data.columns:
                    values = window_data[col].values
                    obs_components.extend(values)
        
        # Features du portefeuille
        if self.use_portfolio_features:
            # Positions actuelles (normalisées)
            for symbol in self.symbols:
                weight = self.portfolio.weights.get(symbol, 0.0)
                obs_components.append(weight)
            
            # Cash ratio
            cash_ratio = self.portfolio.cash / self.portfolio.total_value if self.portfolio.total_value > 0 else 1.0
            obs_components.append(cash_ratio)
            
            # Rendement cumulé
            obs_components.append(self.portfolio.cumulative_returns)
            
            # Drawdown actuel
            if len(self.portfolio_history) > 1:
                peak = max(self.portfolio_history)
                drawdown = (peak - self.portfolio.total_value) / peak if peak > 0 else 0
                obs_components.append(drawdown)
            else:
                obs_components.append(0.0)
        
        # Features de corrélation
        if self.use_correlation_features:
            for i in range(len(self.symbols)):
                for j in range(i+1, len(self.symbols)):
                    col = f"corr_{self.symbols[i]}_{self.symbols[j]}"
                    if col in window_data.columns:
                        values = window_data[col].values
                        obs_components.extend(values)
        
        # Convertir en array numpy
        obs = np.array(obs_components, dtype=np.float32)
        
        # Normaliser si configuré
        if self.normalize_observations and self.obs_mean is not None:
            obs = (obs - self.obs_mean) / self.obs_std
        
        # Gérer les NaN
        obs = np.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
        
        return obs
    
    def _get_current_prices(self) -> Dict[str, float]:
        """Obtient les prix actuels"""
        current_idx = self.start_idx + self.current_step
        prices = {}
        
        for symbol in self.symbols:
            price_col = f"{symbol}_close"
            if price_col in self.current_data.columns:
                prices[symbol] = float(self.current_data.iloc[current_idx][price_col])
        
        return prices
    
    def _execute_action(
        self, 
        action: Union[np.ndarray, Dict[str, np.ndarray]], 
        current_prices: Dict[str, float]
    ) -> List[Dict[str, Any]]:
        """Exécute l'action et retourne les trades effectués"""
        trades = []
        
        if self.action_type == ActionType.CONTINUOUS:
            # Action = allocation cible du portefeuille
            target_weights = action / action.sum() if action.sum() > 0 else np.ones(len(action)) / len(action)
            
            # Cash weight est le dernier élément
            cash_weight = target_weights[-1]
            asset_weights = target_weights[:-1]
            
            # Calculer les positions cibles
            for i, symbol in enumerate(self.symbols):
                target_value = self.portfolio.total_value * asset_weights[i]
                current_value = self.portfolio.values.get(symbol, 0)
                
                # Différence à trader
                trade_value = target_value - current_value
                
                if abs(trade_value) > self.trading_fees.min_trade_size:
                    # Calculer la quantité
                    price = current_prices[symbol]
                    quantity = trade_value / price if price > 0 else 0
                    
                    # Calculer les frais
                    trade_cost = self.trading_fees.calculate_cost(trade_value)
                    
                    # Vérifier si on a assez de cash
                    if trade_value > 0 and self.portfolio.cash >= trade_value + trade_cost:
                        # Achat
                        self.portfolio.positions[symbol] = self.portfolio.positions.get(symbol, 0) + quantity
                        self.portfolio.cash -= (trade_value + trade_cost)
                        
                        trades.append({
                            'timestamp': self._get_current_timestamp(),
                            'symbol': symbol,
                            'side': 'BUY',
                            'quantity': quantity,
                            'price': price,
                            'value': trade_value,
                            'fees': trade_cost
                        })
                    
                    elif trade_value < 0 and self.portfolio.positions.get(symbol, 0) >= abs(quantity):
                        # Vente
                        self.portfolio.positions[symbol] -= abs(quantity)
                        self.portfolio.cash += (abs(trade_value) - trade_cost)
                        
                        trades.append({
                            'timestamp': self._get_current_timestamp(),
                            'symbol': symbol,
                            'side': 'SELL',
                            'quantity': abs(quantity),
                            'price': price,
                            'value': abs(trade_value),
                            'fees': trade_cost
                        })
        
        elif self.action_type == ActionType.DISCRETE:
            # Action = [0: Hold, 1: Buy, 2: Sell] pour chaque actif
            for i, (symbol, act) in enumerate(zip(self.symbols, action)):
                if act == 1:  # Buy
                    # Acheter avec une fraction du cash disponible
                    trade_value = self.portfolio.cash / (self.n_assets * 2)  # Conservative
                    if trade_value > self.trading_fees.min_trade_size:
                        price = current_prices[symbol]
                        quantity = trade_value / price if price > 0 else 0
                        trade_cost = self.trading_fees.calculate_cost(trade_value)
                        
                        if self.portfolio.cash >= trade_value + trade_cost:
                            self.portfolio.positions[symbol] = self.portfolio.positions.get(symbol, 0) + quantity
                            self.portfolio.cash -= (trade_value + trade_cost)
                            
                            trades.append({
                                'timestamp': self._get_current_timestamp(),
                                'symbol': symbol,
                                'side': 'BUY',
                                'quantity': quantity,
                                'price': price,
                                'value': trade_value,
                                'fees': trade_cost
                            })
                
                elif act == 2:  # Sell
                    # Vendre toute la position
                    position = self.portfolio.positions.get(symbol, 0)
                    if position > 0:
                        price = current_prices[symbol]
                        trade_value = position * price
                        trade_cost = self.trading_fees.calculate_cost(trade_value)
                        
                        self.portfolio.positions[symbol] = 0
                        self.portfolio.cash += (trade_value - trade_cost)
                        
                        trades.append({
                            'timestamp': self._get_current_timestamp(),
                            'symbol': symbol,
                            'side': 'SELL',
                            'quantity': position,
                            'price': price,
                            'value': trade_value,
                            'fees': trade_cost
                        })
        
        return trades
    
    def _calculate_reward(self, prev_portfolio_value: float) -> float:
        """Calcule la récompense selon le type configuré"""
        if self.reward_type == RewardType.SIMPLE_RETURN:
            # Rendement simple
            if prev_portfolio_value > 0:
                return_pct = (self.portfolio.total_value - prev_portfolio_value) / prev_portfolio_value
            else:
                return_pct = 0.0
            reward = return_pct * 100  # Scaling
        
        elif self.reward_type == RewardType.SHARPE_RATIO:
            # Sharpe ratio sur fenêtre glissante
            if len(self.portfolio_history) > 20:
                returns = np.diff(self.portfolio_history[-21:]) / self.portfolio_history[-21:-1]
                if np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                    reward = sharpe / 10  # Scaling
                else:
                    reward = 0.0
            else:
                reward = 0.0
        
        elif self.reward_type == RewardType.RISK_ADJUSTED:
            # Rendement ajusté au risque avec pénalités
            if prev_portfolio_value > 0:
                return_pct = (self.portfolio.total_value - prev_portfolio_value) / prev_portfolio_value
                
                # Pénalité pour drawdown
                if len(self.portfolio_history) > 1:
                    peak = max(self.portfolio_history)
                    drawdown = (peak - self.portfolio.total_value) / peak if peak > 0 else 0
                    drawdown_penalty = drawdown * 0.5
                else:
                    drawdown_penalty = 0
                
                # Pénalité pour concentration
                max_weight = max(self.portfolio.weights.values()) if self.portfolio.weights else 0
                concentration_penalty = max(0, max_weight - 0.4) * 0.1
                
                reward = return_pct * 100 - drawdown_penalty - concentration_penalty
            else:
                reward = 0.0
        
        elif self.reward_type == RewardType.DIFFERENTIAL_SHARPE:
            # Amélioration du Sharpe ratio
            if len(self.portfolio_history) > 40:
                # Sharpe actuel
                recent_returns = np.diff(self.portfolio_history[-21:]) / self.portfolio_history[-21:-1]
                if np.std(recent_returns) > 0:
                    current_sharpe = np.mean(recent_returns) / np.std(recent_returns) * np.sqrt(252)
                else:
                    current_sharpe = 0
                
                # Sharpe précédent
                past_returns = np.diff(self.portfolio_history[-41:-20]) / self.portfolio_history[-41:-21]
                if np.std(past_returns) > 0:
                    past_sharpe = np.mean(past_returns) / np.std(past_returns) * np.sqrt(252)
                else:
                    past_sharpe = 0
                
                reward = (current_sharpe - past_sharpe) / 10
            else:
                reward = 0.0
        
        else:
            # Sortino ratio par défaut
            if len(self.portfolio_history) > 20:
                returns = np.diff(self.portfolio_history[-21:]) / self.portfolio_history[-21:-1]
                downside_returns = returns[returns < 0]
                
                if len(downside_returns) > 0 and np.std(downside_returns) > 0:
                    sortino = np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
                    reward = sortino / 10
                else:
                    reward = np.mean(returns) * 100 if len(returns) > 0 else 0
            else:
                reward = 0.0
        
        # Ajouter le coût des transactions
        if self.trade_history:
            recent_fees = sum(t['fees'] for t in self.trade_history[-10:])
            fee_penalty = recent_fees / self.portfolio.total_value * 100
            reward -= fee_penalty
        
        return float(reward)
    
    def _normalize_reward(self, reward: float) -> float:
        """Normalise la récompense"""
        # Mise à jour des statistiques avec moving average
        alpha = 0.01
        self.reward_mean = alpha * reward + (1 - alpha) * self.reward_mean
        self.reward_std = alpha * abs(reward - self.reward_mean) + (1 - alpha) * self.reward_std
        
        # Normaliser
        if self.reward_std > 0:
            normalized = (reward - self.reward_mean) / self.reward_std
        else:
            normalized = reward
        
        # Clip pour stabilité
        return np.clip(normalized, -10, 10)
    
    def _is_done(self) -> bool:
        """Vérifie si l'épisode est terminé"""
        # Fin des données
        if self.current_step >= self.max_episode_steps:
            return True
        
        if self.start_idx + self.current_step >= len(self.current_data) - 1:
            return True
        
        # Bankruptcy
        if self.portfolio.total_value < self.initial_capital * 0.1:
            self.logger.warning("episode_ended_bankruptcy", value=self.portfolio.total_value)
            return True
        
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """Retourne les informations supplémentaires"""
        info = {
            'portfolio_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'positions': dict(self.portfolio.positions),
            'weights': dict(self.portfolio.weights),
            'n_trades': len(self.trade_history),
            'cumulative_return': (self.portfolio.total_value / self.initial_capital - 1) * 100,
            'current_step': self.current_step,
            'current_date': self._get_current_timestamp()
        }
        
        # Ajouter les métriques si assez de données
        if len(self.portfolio_history) > 20:
            returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
            
            info['metrics'] = {
                'total_return': (self.portfolio.total_value / self.initial_capital - 1) * 100,
                'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0,
                'sharpe_ratio': self._calculate_sharpe_ratio(returns),
                'max_drawdown': self._calculate_max_drawdown(),
                'win_rate': self._calculate_win_rate()
            }
        
        return info
    
    def _get_current_timestamp(self) -> datetime:
        """Obtient le timestamp actuel"""
        current_idx = self.start_idx + self.current_step
        return self.current_data.index[current_idx]
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calcule le Sharpe ratio"""
        if len(returns) > 0 and np.std(returns) > 0:
            return np.mean(returns) / np.std(returns) * np.sqrt(252)
        return 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """Calcule le drawdown maximum"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        peak = self.portfolio_history[0]
        max_dd = 0.0
        
        for value in self.portfolio_history[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)
        
        return max_dd * 100
    
    def _calculate_win_rate(self) -> float:
        """Calcule le taux de trades gagnants"""
        if not self.trade_history:
            return 0.0
        
        winning_trades = 0
        for i, trade in enumerate(self.trade_history):
            if trade['side'] == 'SELL':
                # Trouver l'achat correspondant
                symbol = trade['symbol']
                sell_value = trade['value']
                
                # Chercher en arrière
                for j in range(i-1, -1, -1):
                    prev_trade = self.trade_history[j]
                    if prev_trade['symbol'] == symbol and prev_trade['side'] == 'BUY':
                        buy_value = prev_trade['value']
                        if sell_value > buy_value:
                            winning_trades += 1
                        break
        
        sell_trades = sum(1 for t in self.trade_history if t['side'] == 'SELL')
        return winning_trades / sell_trades * 100 if sell_trades > 0 else 0.0
    
    def render(self, mode: str = 'human') -> Optional[np.ndarray]:
        """Affiche l'état de l'environnement"""
        if mode == 'terminal':
            # Affichage texte dans le terminal
            print(f"\n=== Step {self.current_step} ===")
            print(f"Date: {self._get_current_timestamp()}")
            print(f"Portfolio Value: ${self.portfolio.total_value:,.2f}")
            print(f"Cash: ${self.portfolio.cash:,.2f}")
            print(f"Return: {(self.portfolio.total_value / self.initial_capital - 1) * 100:.2f}%")
            
            print("\nPositions:")
            for symbol, quantity in self.portfolio.positions.items():
                if quantity > 0:
                    value = self.portfolio.values.get(symbol, 0)
                    weight = self.portfolio.weights.get(symbol, 0)
                    print(f"  {symbol}: {quantity:.4f} (${value:,.2f}, {weight:.1%})")
            
            if len(self.portfolio_history) > 20:
                returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
                print(f"\nSharpe Ratio: {self._calculate_sharpe_ratio(returns):.2f}")
                print(f"Max Drawdown: {self._calculate_max_drawdown():.2f}%")
        
        elif mode == 'human':
            # Visualisation graphique
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Portfolio value
            ax = axes[0, 0]
            ax.plot(self.portfolio_history)
            ax.set_title('Portfolio Value')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Value ($)')
            ax.grid(True)
            
            # Asset weights
            ax = axes[0, 1]
            weights = list(self.portfolio.weights.values())
            if sum(weights) > 0:
                ax.pie(weights + [self.portfolio.cash / self.portfolio.total_value], 
                      labels=self.symbols + ['Cash'],
                      autopct='%1.1f%%')
                ax.set_title('Current Allocation')
            
            # Returns distribution
            ax = axes[1, 0]
            if len(self.portfolio_history) > 2:
                returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
                ax.hist(returns * 100, bins=30, alpha=0.7)
                ax.set_title('Returns Distribution')
                ax.set_xlabel('Return (%)')
                ax.set_ylabel('Frequency')
                ax.grid(True)
            
            # Cumulative returns by asset
            ax = axes[1, 1]
            for symbol in self.symbols:
                prices = []
                for i in range(len(self.portfolio_history)):
                    idx = self.start_idx + i
                    if idx < len(self.current_data):
                        price = self.current_data.iloc[idx][f"{symbol}_close"]
                        prices.append(price)
                
                if prices and prices[0] > 0:
                    cum_returns = [(p/prices[0] - 1) * 100 for p in prices]
                    ax.plot(cum_returns, label=symbol)
            
            ax.set_title('Cumulative Returns by Asset')
            ax.set_xlabel('Steps')
            ax.set_ylabel('Return (%)')
            ax.legend()
            ax.grid(True)
            
            plt.tight_layout()
            plt.show()
            
            if mode == 'rgb_array':
                # Convertir en array pour enregistrement vidéo
                fig.canvas.draw()
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)
                return img
        
        return None
    
    def close(self) -> None:
        """Ferme l'environnement et libère les ressources"""
        plt.close('all')
        self.logger.info("environment_closed")
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques complètes de l'épisode"""
        if not self.portfolio_history:
            return {}
        
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        
        stats = {
            'total_return': (self.portfolio.total_value / self.initial_capital - 1) * 100,
            'annualized_return': ((self.portfolio.total_value / self.initial_capital) ** (252 / len(self.portfolio_history)) - 1) * 100 if len(self.portfolio_history) > 0 else 0,
            'volatility': np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(returns),
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'max_drawdown': self._calculate_max_drawdown(),
            'calmar_ratio': self._calculate_calmar_ratio(),
            'win_rate': self._calculate_win_rate(),
            'n_trades': len(self.trade_history),
            'avg_trade_size': np.mean([t['value'] for t in self.trade_history]) if self.trade_history else 0,
            'total_fees': sum(t['fees'] for t in self.trade_history),
            'final_cash': self.portfolio.cash,
            'final_positions': dict(self.portfolio.positions),
            'episode_length': len(self.portfolio_history)
        }
        
        return stats
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calcule le ratio de Sortino"""
        if len(returns) == 0:
            return 0.0
        
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0 and np.std(downside_returns) > 0:
            return np.mean(returns) / np.std(downside_returns) * np.sqrt(252)
        return 0.0
    
    def _calculate_calmar_ratio(self) -> float:
        """Calcule le ratio de Calmar"""
        if len(self.portfolio_history) < 2:
            return 0.0
        
        annual_return = ((self.portfolio.total_value / self.initial_capital) ** (252 / len(self.portfolio_history)) - 1)
        max_dd = self._calculate_max_drawdown() / 100
        
        if max_dd > 0:
            return annual_return / max_dd
        return 0.0


# Wrapper pour compatibilité avec vecenv
class VecCompatibleWrapper(gym.Wrapper):
    """Wrapper pour rendre l'environnement compatible avec vectorized environments"""
    
    def __init__(self, env):
        super().__init__(env)
        
    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return obs
    
    def step(self, action):
        return self.env.step(action)


# Factory function pour créer des environnements
def create_multi_asset_env(
    symbols: List[str],
    data_path: Optional[str] = None,
    **kwargs
) -> MultiAssetTradingEnv:
    """
    Crée un environnement de trading multi-actifs configuré
    
    Args:
        symbols: Liste des symboles à trader
        data_path: Chemin vers les données (optionnel)
        **kwargs: Arguments supplémentaires pour l'environnement
        
    Returns:
        Environnement configuré
    """
    # Charger les données si un chemin est fourni
    data_source = None
    if data_path:
        try:
            data_source = pd.read_csv(data_path, index_col=0, parse_dates=True)
        except Exception as e:
            print(f"Erreur lors du chargement des données: {e}")
    
    # Créer l'environnement
    env = MultiAssetTradingEnv(
        symbols=symbols,
        data_source=data_source,
        **kwargs
    )
    
    return env