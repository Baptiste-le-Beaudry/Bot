"""
Trading Environment Module
Environnement de trading pour Deep Reinforcement Learning compatible OpenAI Gym.
Supporte multiple stratégies : arbitrage, market making, scalping, trend following.
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from collections import deque
import random

# Import des modules internes
from ..features.feature_engineering import FeatureEngineer
from ..features.technical_indicators import TechnicalIndicators
from ..features.market_regime import MarketRegimeDetector, MarketRegime
from ...utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown

# Configuration du logger
logger = logging.getLogger(__name__)


class ActionType(Enum):
    """Types d'actions disponibles"""
    # Actions de base
    HOLD = 0
    BUY = 1
    SELL = 2
    
    # Actions avancées pour market making
    PLACE_BID = 3
    PLACE_ASK = 4
    CANCEL_ORDER = 5
    
    # Actions de gestion du risque
    CLOSE_POSITION = 6
    HEDGE = 7


class PositionType(Enum):
    """Types de positions"""
    FLAT = 0
    LONG = 1
    SHORT = -1


class RewardType(Enum):
    """Types de récompenses"""
    PROFIT = "profit"
    SHARPE = "sharpe"
    RISK_ADJUSTED = "risk_adjusted"
    SORTINO = "sortino"
    CALMAR = "calmar"


@dataclass
class TradingState:
    """État complet du trading"""
    # Marché
    timestamp: pd.Timestamp
    current_price: float
    bid: float
    ask: float
    spread: float
    volume: float
    
    # Position
    position: float  # -1 to 1 (short to long)
    entry_price: float
    position_value: float
    unrealized_pnl: float
    
    # Portfolio
    cash: float
    total_value: float
    buying_power: float
    margin_used: float
    
    # Historique
    n_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Features techniques
    features: np.ndarray
    
    # Régime de marché
    market_regime: MarketRegime
    regime_confidence: float


class TradingEnv(gym.Env):
    """
    Environnement de trading pour Deep Reinforcement Learning.
    Compatible avec OpenAI Gym et optimisé pour les stratégies haute fréquence.
    """
    
    metadata = {'render.modes': ['human', 'ansi']}
    
    def __init__(self,
                 data: pd.DataFrame,
                 initial_balance: float = 100000,
                 leverage: float = 1.0,
                 commission_rate: float = 0.001,
                 slippage_rate: float = 0.0005,
                 min_position_size: float = 0.01,
                 max_position_size: float = 1.0,
                 reward_type: RewardType = RewardType.SHARPE,
                 lookback_window: int = 50,
                 features_list: Optional[List[str]] = None,
                 enable_short_selling: bool = True,
                 enable_market_making: bool = False,
                 risk_free_rate: float = 0.02,
                 episode_length: Optional[int] = None):
        """
        Initialisation de l'environnement
        
        Args:
            data: DataFrame avec OHLCV et features
            initial_balance: Capital initial
            leverage: Levier maximum
            commission_rate: Taux de commission
            slippage_rate: Taux de slippage
            min_position_size: Taille min de position (% du capital)
            max_position_size: Taille max de position (% du capital)
            reward_type: Type de récompense
            lookback_window: Fenêtre d'observation
            features_list: Liste des features à utiliser
            enable_short_selling: Autoriser la vente à découvert
            enable_market_making: Mode market making
            risk_free_rate: Taux sans risque pour Sharpe
            episode_length: Longueur max d'un épisode
        """
        super().__init__()
        
        # Configuration
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.reward_type = reward_type
        self.lookback_window = lookback_window
        self.enable_short_selling = enable_short_selling
        self.enable_market_making = enable_market_making
        self.risk_free_rate = risk_free_rate
        self.episode_length = episode_length
        
        # Données
        self.data = data
        self.n_steps = len(data)
        
        # Feature engineering
        self.feature_engineer = FeatureEngineer()
        self.technical_indicators = TechnicalIndicators()
        self.regime_detector = MarketRegimeDetector()
        
        # Préparer les features
        self._prepare_features(features_list)
        
        # Espaces d'action et d'observation
        self._setup_spaces()
        
        # État
        self.current_step = 0
        self.done = False
        self._reset_state()
        
        # Historique pour calculs
        self.trade_history = []
        self.portfolio_values = []
        self.positions_history = []
        self.returns_history = []
        
        # Métriques
        self.total_trades = 0
        self.winning_trades = 0
        self.total_commission_paid = 0
        self.total_slippage_cost = 0
        
        # Market making specifics
        if self.enable_market_making:
            self.order_book = {'bids': {}, 'asks': {}}
            self.filled_orders = []
    
    def _prepare_features(self, features_list: Optional[List[str]] = None):
        """Préparer les features pour l'observation"""
        
        # Calculer toutes les features techniques
        logger.info("Calcul des features techniques...")
        
        # Features de base
        self.data['returns'] = self.data['close'].pct_change()
        self.data['log_returns'] = np.log(self.data['close'] / self.data['close'].shift(1))
        
        # Indicateurs techniques
        indicators = self.technical_indicators.all_indicators(
            self.data,
            indicators=['sma_20', 'ema_20', 'rsi_14', 'macd', 'bb_20', 
                       'atr_14', 'adx_14', 'obv', 'mfi_14']
        )
        
        # Fusionner avec les données
        for col in indicators.columns:
            if col not in self.data.columns:
                self.data[col] = indicators[col]
        
        # Feature engineering avancé
        self.data = self.feature_engineer.engineer_features(self.data)
        
        # Détecter les régimes de marché
        regime_analysis = self.regime_detector.detect_regime(self.data)
        self.data['market_regime'] = regime_analysis.regime_history['regime']
        self.data['regime_confidence'] = regime_analysis.regime_history['confidence']
        
        # Sélectionner les features
        if features_list:
            self.feature_columns = features_list
        else:
            # Features par défaut optimisées pour le trading
            self.feature_columns = [
                # Prix et returns
                'returns', 'log_returns', 'volatility_20',
                
                # Moyennes mobiles
                'sma_20', 'ema_20', 'price_to_sma20',
                
                # Momentum
                'rsi_14', 'macd', 'macd_signal', 'macd_hist',
                
                # Volatilité
                'atr_14', 'bb_width', 'bb_position',
                
                # Volume
                'volume_ratio', 'obv', 'mfi_14',
                
                # Microstructure (si disponible)
                'bid_ask_spread_pct', 'order_flow_imbalance',
                
                # Régime
                'regime_confidence', 'adx_14'
            ]
        
        # Filtrer les colonnes disponibles
        self.feature_columns = [col for col in self.feature_columns 
                               if col in self.data.columns]
        
        logger.info(f"Features sélectionnées: {len(self.feature_columns)}")
        
        # Normaliser les features
        self._normalize_features()
    
    def _normalize_features(self):
        """Normaliser les features pour l'apprentissage"""
        from sklearn.preprocessing import StandardScaler
        
        # Créer une copie pour la normalisation
        features_df = self.data[self.feature_columns].copy()
        
        # Remplacer les infinis et NaN
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # Normaliser
        self.scaler = StandardScaler()
        self.normalized_features = self.scaler.fit_transform(features_df)
    
    def _setup_spaces(self):
        """Définir les espaces d'action et d'observation"""
        
        # Espace d'action
        if self.enable_market_making:
            # Actions complexes pour market making
            # [action_type, size, bid_offset, ask_offset]
            self.action_space = spaces.Box(
                low=np.array([0, 0, -0.01, -0.01]),
                high=np.array([7, 1, 0.01, 0.01]),
                dtype=np.float32
            )
        else:
            # Actions simples : [position_target] de -1 (full short) à 1 (full long)
            self.action_space = spaces.Box(
                low=-1 if self.enable_short_selling else 0,
                high=1,
                shape=(1,),
                dtype=np.float32
            )
        
        # Espace d'observation
        # Features du marché + état du portfolio
        n_features = len(self.feature_columns)
        n_portfolio_features = 15  # Position, PnL, cash, etc.
        
        # Observation = [features_marché, features_portfolio]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.lookback_window, n_features + n_portfolio_features),
            dtype=np.float32
        )
    
    def _reset_state(self):
        """Réinitialiser l'état interne"""
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.position_value = 0.0
        
        # Historiques
        self.trade_history = []
        self.portfolio_values = [self.balance]
        self.positions_history = [0.0]
        self.returns_history = [0.0]
        
        # Métriques
        self.total_trades = 0
        self.winning_trades = 0
        self.total_commission_paid = 0
        self.total_slippage_cost = 0
        
        # Market making
        if self.enable_market_making:
            self.order_book = {'bids': {}, 'asks': {}}
            self.filled_orders = []
    
    def reset(self, start_idx: Optional[int] = None) -> np.ndarray:
        """
        Réinitialiser l'environnement
        
        Args:
            start_idx: Index de départ (random si None)
            
        Returns:
            Observation initiale
        """
        self._reset_state()
        
        # Choisir un point de départ
        if start_idx is None:
            # Assurer qu'on a assez de données pour le lookback
            max_start = self.n_steps - self.lookback_window - (self.episode_length or 1000)
            self.current_step = np.random.randint(self.lookback_window, max_start)
        else:
            self.current_step = max(self.lookback_window, start_idx)
        
        self.start_step = self.current_step
        self.done = False
        
        return self._get_observation()
    
    def step(self, action: Union[float, np.ndarray]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Exécuter une action dans l'environnement
        
        Args:
            action: Action à exécuter
            
        Returns:
            observation, reward, done, info
        """
        # Convertir l'action
        if isinstance(action, np.ndarray):
            action = action.flatten()
        
        # Prix actuel
        current_price = self.data['close'].iloc[self.current_step]
        current_bid = self.data.get('bid', current_price * 0.9999).iloc[self.current_step]
        current_ask = self.data.get('ask', current_price * 1.0001).iloc[self.current_step]
        
        # Exécuter l'action
        if self.enable_market_making:
            reward = self._execute_market_making_action(action, current_bid, current_ask)
        else:
            reward = self._execute_simple_action(action[0], current_price)
        
        # Mettre à jour l'état
        self.current_step += 1
        
        # Calculer la valeur du portfolio
        self._update_portfolio_value(current_price)
        
        # Vérifier la fin de l'épisode
        self.done = self._check_done()
        
        # Observation suivante
        observation = self._get_observation()
        
        # Informations supplémentaires
        info = self._get_info()
        
        return observation, reward, self.done, info
    
    def _execute_simple_action(self, action: float, current_price: float) -> float:
        """Exécuter une action simple (position cible)"""
        
        # Interpréter l'action comme position cible
        target_position = np.clip(action, 
                                 -self.max_position_size if self.enable_short_selling else 0,
                                 self.max_position_size)
        
        # Calculer le changement de position
        position_change = target_position - self.position
        
        # Pas de changement significatif
        if abs(position_change) < self.min_position_size:
            return self._calculate_holding_reward(current_price)
        
        # Calculer les coûts de transaction
        trade_value = abs(position_change) * current_price * self.balance
        
        # Appliquer slippage
        if position_change > 0:  # Achat
            execution_price = current_price * (1 + self.slippage_rate)
        else:  # Vente
            execution_price = current_price * (1 - self.slippage_rate)
        
        # Commission
        commission = trade_value * self.commission_rate
        self.total_commission_paid += commission
        
        # Coût du slippage
        slippage_cost = abs(execution_price - current_price) * abs(position_change) * self.balance
        self.total_slippage_cost += slippage_cost
        
        # Mettre à jour la position
        old_position = self.position
        self.position = target_position
        
        # Mettre à jour le prix d'entrée (moyenne pondérée)
        if old_position == 0:
            self.entry_price = execution_price
        elif np.sign(old_position) == np.sign(target_position):
            # Ajout à la position existante
            total_value = old_position * self.entry_price + position_change * execution_price
            self.entry_price = total_value / target_position
        else:
            # Changement de direction
            self.entry_price = execution_price
        
        # Enregistrer le trade
        self.total_trades += 1
        self.trade_history.append({
            'timestamp': self.data.index[self.current_step],
            'action': 'BUY' if position_change > 0 else 'SELL',
            'price': execution_price,
            'size': abs(position_change),
            'commission': commission,
            'position_after': self.position
        })
        
        # Calculer la récompense
        return self._calculate_trade_reward(current_price, commission, slippage_cost)
    
    def _execute_market_making_action(self, action: np.ndarray, 
                                    current_bid: float, current_ask: float) -> float:
        """Exécuter une action de market making"""
        
        action_type = int(action[0])
        size = action[1] * self.max_position_size
        bid_offset = action[2]  # Offset from mid price
        ask_offset = action[3]
        
        mid_price = (current_bid + current_ask) / 2
        
        if action_type == ActionType.PLACE_BID.value:
            # Placer un ordre d'achat
            bid_price = mid_price * (1 + bid_offset)
            self.order_book['bids'][len(self.order_book['bids'])] = {
                'price': bid_price,
                'size': size,
                'timestamp': self.current_step
            }
            
        elif action_type == ActionType.PLACE_ASK.value:
            # Placer un ordre de vente
            ask_price = mid_price * (1 + ask_offset)
            self.order_book['asks'][len(self.order_book['asks'])] = {
                'price': ask_price,
                'size': size,
                'timestamp': self.current_step
            }
            
        elif action_type == ActionType.CANCEL_ORDER.value:
            # Annuler les ordres anciens
            self._cancel_old_orders()
        
        # Vérifier les exécutions d'ordres
        filled_value = self._check_order_fills(current_bid, current_ask)
        
        # Calculer la récompense basée sur le spread capturé
        spread_captured = self._calculate_spread_capture()
        inventory_penalty = self._calculate_inventory_penalty()
        
        return spread_captured - inventory_penalty + filled_value * 0.001
    
    def _calculate_holding_reward(self, current_price: float) -> float:
        """Calculer la récompense pour maintenir la position"""
        if self.position == 0:
            return 0.0
        
        # PnL non réalisé
        price_change = (current_price - self.entry_price) / self.entry_price
        unrealized_pnl = self.position * price_change * self.balance
        
        # Pénalité pour tenir une position (encourage l'action)
        holding_penalty = -abs(self.position) * 0.0001
        
        return unrealized_pnl * 0.001 + holding_penalty
    
    def _calculate_trade_reward(self, current_price: float, 
                              commission: float, slippage_cost: float) -> float:
        """Calculer la récompense après un trade"""
        
        # Retour immédiat (peut être négatif à cause des coûts)
        immediate_cost = -(commission + slippage_cost) / self.balance
        
        # Potentiel de profit basé sur le régime
        regime = self.data['market_regime'].iloc[self.current_step]
        regime_bonus = self._get_regime_bonus(regime)
        
        # Récompense basée sur le type configuré
        if self.reward_type == RewardType.PROFIT:
            return immediate_cost
            
        elif self.reward_type == RewardType.SHARPE:
            # Calculer le Sharpe ratio des derniers returns
            if len(self.returns_history) > 20:
                recent_returns = self.returns_history[-20:]
                sharpe = calculate_sharpe_ratio(recent_returns, self.risk_free_rate)
                return immediate_cost + sharpe * 0.01
            else:
                return immediate_cost
                
        elif self.reward_type == RewardType.RISK_ADJUSTED:
            # Récompense ajustée au risque
            position_risk = abs(self.position) * self.data['volatility_20'].iloc[self.current_step]
            risk_adjustment = 1 / (1 + position_risk)
            return immediate_cost * risk_adjustment + regime_bonus
        
        return immediate_cost
    
    def _get_regime_bonus(self, regime: MarketRegime) -> float:
        """Bonus/pénalité selon le régime de marché"""
        regime_rewards = {
            MarketRegime.BULL_TREND: 0.001 if self.position > 0 else -0.001,
            MarketRegime.BEAR_TREND: 0.001 if self.position < 0 else -0.001,
            MarketRegime.RANGING: 0.0005 if abs(self.position) < 0.3 else -0.0005,
            MarketRegime.HIGH_VOLATILITY: -0.001 * abs(self.position),
            MarketRegime.CRASH: 0.002 if self.position <= 0 else -0.002
        }
        
        return regime_rewards.get(regime, 0.0)
    
    def _update_portfolio_value(self, current_price: float):
        """Mettre à jour la valeur du portfolio"""
        
        # Valeur de la position
        if self.position != 0:
            self.position_value = self.position * current_price * self.balance
            unrealized_pnl = self.position * (current_price - self.entry_price) / self.entry_price * self.balance
        else:
            self.position_value = 0
            unrealized_pnl = 0
        
        # Valeur totale
        total_value = self.balance + unrealized_pnl
        
        # Calculer le return
        if len(self.portfolio_values) > 0:
            period_return = (total_value - self.portfolio_values[-1]) / self.portfolio_values[-1]
        else:
            period_return = 0
        
        # Mettre à jour les historiques
        self.portfolio_values.append(total_value)
        self.positions_history.append(self.position)
        self.returns_history.append(period_return)
    
    def _check_done(self) -> bool:
        """Vérifier si l'épisode est terminé"""
        
        # Fin des données
        if self.current_step >= self.n_steps - 1:
            return True
        
        # Longueur maximale d'épisode atteinte
        if self.episode_length and (self.current_step - self.start_step) >= self.episode_length:
            return True
        
        # Ruine (perte > 50%)
        current_value = self.portfolio_values[-1]
        if current_value < self.initial_balance * 0.5:
            logger.warning("Episode terminé: perte > 50%")
            return True
        
        # Drawdown excessif
        if len(self.portfolio_values) > 10:
            recent_peak = max(self.portfolio_values[-100:])
            current_drawdown = (recent_peak - current_value) / recent_peak
            if current_drawdown > 0.25:  # 25% drawdown
                logger.warning(f"Episode terminé: drawdown {current_drawdown:.2%}")
                return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Construire l'observation actuelle"""
        
        # Indices pour le lookback
        end_idx = self.current_step
        start_idx = max(0, end_idx - self.lookback_window)
        
        # Features du marché
        market_features = self.normalized_features[start_idx:end_idx]
        
        # Si pas assez de données, padder avec des zéros
        if len(market_features) < self.lookback_window:
            padding = np.zeros((self.lookback_window - len(market_features), 
                              market_features.shape[1]))
            market_features = np.vstack([padding, market_features])
        
        # Features du portfolio (répétées pour chaque timestep)
        portfolio_features = self._get_portfolio_features()
        portfolio_array = np.tile(portfolio_features, (self.lookback_window, 1))
        
        # Combiner
        observation = np.hstack([market_features, portfolio_array])
        
        return observation.astype(np.float32)
    
    def _get_portfolio_features(self) -> np.ndarray:
        """Obtenir les features du portfolio"""
        
        # Valeur actuelle
        current_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_balance
        
        # Métriques de performance
        if len(self.returns_history) > 20:
            sharpe = calculate_sharpe_ratio(self.returns_history[-20:], self.risk_free_rate)
            max_dd = calculate_max_drawdown(self.portfolio_values[-100:])
            win_rate = self.winning_trades / max(self.total_trades, 1)
        else:
            sharpe = 0
            max_dd = 0
            win_rate = 0
        
        features = np.array([
            # Position actuelle
            self.position,
            self.position_value / current_value if current_value > 0 else 0,
            
            # PnL
            (current_value - self.initial_balance) / self.initial_balance,
            self.returns_history[-1] if self.returns_history else 0,
            
            # Risque
            np.std(self.returns_history[-20:]) if len(self.returns_history) > 20 else 0,
            max_dd,
            
            # Performance
            sharpe,
            win_rate,
            
            # Exposition
            abs(self.position),
            self.balance / self.initial_balance,
            
            # Coûts
            self.total_commission_paid / self.initial_balance,
            self.total_slippage_cost / self.initial_balance,
            
            # Trading activity
            self.total_trades / max(self.current_step - self.start_step, 1),
            
            # Temps dans l'épisode
            (self.current_step - self.start_step) / (self.episode_length or 1000),
            
            # Marge disponible
            (self.balance * self.leverage - abs(self.position_value)) / (self.balance * self.leverage)
        ])
        
        return features
    
    def _get_info(self) -> Dict[str, Any]:
        """Obtenir les informations de debug"""
        
        current_value = self.portfolio_values[-1]
        
        info = {
            'current_step': self.current_step,
            'position': self.position,
            'balance': self.balance,
            'total_value': current_value,
            'total_return': (current_value - self.initial_balance) / self.initial_balance,
            'n_trades': self.total_trades,
            'commission_paid': self.total_commission_paid,
            'slippage_cost': self.total_slippage_cost,
            'current_price': self.data['close'].iloc[self.current_step],
            'market_regime': self.data['market_regime'].iloc[self.current_step].value
        }
        
        # Métriques de performance
        if len(self.returns_history) > 20:
            info['sharpe_ratio'] = calculate_sharpe_ratio(self.returns_history, self.risk_free_rate)
            info['max_drawdown'] = calculate_max_drawdown(self.portfolio_values)
            info['win_rate'] = self.winning_trades / max(self.total_trades, 1)
        
        return info
    
    def render(self, mode: str = 'human'):
        """Afficher l'état actuel"""
        
        if mode == 'ansi':
            return self._render_ansi()
        
        current_price = self.data['close'].iloc[self.current_step]
        current_value = self.portfolio_values[-1]
        total_return = (current_value - self.initial_balance) / self.initial_balance
        
        print(f"\n=== Step {self.current_step} ===")
        print(f"Price: ${current_price:.2f}")
        print(f"Position: {self.position:.3f}")
        print(f"Portfolio Value: ${current_value:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Trades: {self.total_trades}")
        
        if len(self.returns_history) > 20:
            sharpe = calculate_sharpe_ratio(self.returns_history, self.risk_free_rate)
            print(f"Sharpe Ratio: {sharpe:.2f}")
    
    def _render_ansi(self) -> str:
        """Rendu ANSI pour logging"""
        current_value = self.portfolio_values[-1]
        total_return = (current_value - self.initial_balance) / self.initial_balance
        
        return (f"Step: {self.current_step} | "
                f"Pos: {self.position:+.2f} | "
                f"Val: ${current_value:,.0f} | "
                f"Ret: {total_return:+.1%} | "
                f"Trades: {self.total_trades}")
    
    def get_trading_history(self) -> pd.DataFrame:
        """Obtenir l'historique complet des trades"""
        if not self.trade_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trade_history)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculer toutes les métriques de performance"""
        
        current_value = self.portfolio_values[-1]
        total_return = (current_value - self.initial_balance) / self.initial_balance
        
        metrics = {
            'total_return': total_return,
            'final_value': current_value,
            'n_trades': self.total_trades,
            'commission_paid': self.total_commission_paid,
            'slippage_cost': self.total_slippage_cost,
            'total_costs': self.total_commission_paid + self.total_slippage_cost,
            'net_return': total_return - (self.total_commission_paid + self.total_slippage_cost) / self.initial_balance
        }
        
        if len(self.returns_history) > 20:
            metrics['sharpe_ratio'] = calculate_sharpe_ratio(self.returns_history, self.risk_free_rate)
            metrics['sortino_ratio'] = calculate_sharpe_ratio(
                [r for r in self.returns_history if r < 0], self.risk_free_rate
            )
            metrics['max_drawdown'] = calculate_max_drawdown(self.portfolio_values)
            metrics['volatility'] = np.std(self.returns_history) * np.sqrt(252)
            metrics['win_rate'] = self.winning_trades / max(self.total_trades, 1)
            
            # Calculer le profit factor
            winning_returns = [r for r in self.returns_history if r > 0]
            losing_returns = [r for r in self.returns_history if r < 0]
            
            if losing_returns:
                metrics['profit_factor'] = sum(winning_returns) / abs(sum(losing_returns))
            else:
                metrics['profit_factor'] = np.inf
        
        return metrics
    
    # Méthodes pour market making
    
    def _check_order_fills(self, current_bid: float, current_ask: float) -> float:
        """Vérifier si des ordres ont été exécutés"""
        filled_value = 0.0
        
        # Vérifier les ordres d'achat
        for order_id, order in list(self.order_book['bids'].items()):
            if current_ask <= order['price']:
                # Ordre exécuté
                self.position += order['size']
                self.balance -= order['size'] * order['price'] * self.balance
                filled_value += order['size'] * (current_bid - order['price'])
                
                self.filled_orders.append({
                    'type': 'bid',
                    'price': order['price'],
                    'size': order['size'],
                    'step': self.current_step
                })
                
                del self.order_book['bids'][order_id]
        
        # Vérifier les ordres de vente
        for order_id, order in list(self.order_book['asks'].items()):
            if current_bid >= order['price']:
                # Ordre exécuté
                self.position -= order['size']
                self.balance += order['size'] * order['price'] * self.balance
                filled_value += order['size'] * (order['price'] - current_ask)
                
                self.filled_orders.append({
                    'type': 'ask',
                    'price': order['price'],
                    'size': order['size'],
                    'step': self.current_step
                })
                
                del self.order_book['asks'][order_id]
        
        return filled_value
    
    def _calculate_spread_capture(self) -> float:
        """Calculer le spread capturé par market making"""
        if not self.filled_orders:
            return 0.0
        
        # Calculer le spread moyen capturé sur les derniers trades
        recent_fills = self.filled_orders[-10:]
        
        bid_fills = [f for f in recent_fills if f['type'] == 'bid']
        ask_fills = [f for f in recent_fills if f['type'] == 'ask']
        
        if bid_fills and ask_fills:
            avg_bid = np.mean([f['price'] for f in bid_fills])
            avg_ask = np.mean([f['price'] for f in ask_fills])
            spread = (avg_ask - avg_bid) / avg_bid
            return spread * len(recent_fills) * 0.001
        
        return 0.0
    
    def _calculate_inventory_penalty(self) -> float:
        """Pénalité pour inventory risk en market making"""
        # Pénaliser les positions trop importantes
        inventory_risk = abs(self.position) ** 2
        return -inventory_risk * 0.0001
    
    def _cancel_old_orders(self):
        """Annuler les ordres anciens"""
        current_step = self.current_step
        
        # Annuler les ordres de plus de 10 steps
        for order_book in [self.order_book['bids'], self.order_book['asks']]:
            orders_to_cancel = []
            for order_id, order in order_book.items():
                if current_step - order['timestamp'] > 10:
                    orders_to_cancel.append(order_id)
            
            for order_id in orders_to_cancel:
                del order_book[order_id]


# Environnement multi-assets (version simplifiée)
class MultiAssetTradingEnv(TradingEnv):
    """
    Extension pour trading multi-actifs
    Permet l'arbitrage et la diversification
    """
    
    def __init__(self, 
                 data_dict: Dict[str, pd.DataFrame],
                 correlation_threshold: float = 0.7,
                 **kwargs):
        """
        Args:
            data_dict: Dict {symbol: DataFrame}
            correlation_threshold: Seuil pour pair trading
            **kwargs: Arguments pour TradingEnv
        """
        # Combiner les données
        self.symbols = list(data_dict.keys())
        self.n_assets = len(self.symbols)
        
        # Calculer les corrélations
        self._calculate_correlations(data_dict)
        
        # Créer un DataFrame combiné
        combined_data = self._combine_data(data_dict)
        
        # Initialiser avec les données combinées
        super().__init__(combined_data, **kwargs)
        
        # Redéfinir l'espace d'action pour multi-assets
        self.action_space = spaces.Box(
            low=-1 if self.enable_short_selling else 0,
            high=1,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # Positions par actif
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.entry_prices = {symbol: 0.0 for symbol in self.symbols}
    
    def _calculate_correlations(self, data_dict: Dict[str, pd.DataFrame]):
        """Calculer les corrélations entre actifs"""
        returns = pd.DataFrame()
        
        for symbol, data in data_dict.items():
            returns[symbol] = data['close'].pct_change()
        
        self.correlation_matrix = returns.corr()
        
        # Identifier les paires pour trading
        self.trading_pairs = []
        for i in range(len(self.symbols)):
            for j in range(i+1, len(self.symbols)):
                corr = self.correlation_matrix.iloc[i, j]
                if abs(corr) > self.correlation_threshold:
                    self.trading_pairs.append((self.symbols[i], self.symbols[j], corr))
    
    def _combine_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combiner les données multi-assets"""
        # Prendre la première série comme référence
        base_symbol = self.symbols[0]
        combined = data_dict[base_symbol].copy()
        
        # Ajouter les colonnes des autres actifs
        for symbol in self.symbols[1:]:
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in data_dict[symbol].columns:
                    combined[f'{symbol}_{col}'] = data_dict[symbol][col]
        
        return combined


# Fonction utilitaire pour créer des environnements
def create_trading_env(data: pd.DataFrame,
                      env_type: str = 'simple',
                      **kwargs) -> TradingEnv:
    """
    Créer un environnement de trading
    
    Args:
        data: Données de marché
        env_type: 'simple', 'market_making', 'multi_asset'
        **kwargs: Arguments additionnels
        
    Returns:
        Environnement configuré
    """
    
    default_config = {
        'initial_balance': 100000,
        'commission_rate': 0.001,
        'reward_type': RewardType.SHARPE,
        'lookback_window': 50
    }
    
    # Fusionner avec les kwargs
    config = {**default_config, **kwargs}
    
    if env_type == 'simple':
        return TradingEnv(data, **config)
    
    elif env_type == 'market_making':
        config['enable_market_making'] = True
        return TradingEnv(data, **config)
    
    elif env_type == 'multi_asset':
        if not isinstance(data, dict):
            raise ValueError("Multi-asset env nécessite un dict de DataFrames")
        return MultiAssetTradingEnv(data, **config)
    
    else:
        raise ValueError(f"Type d'environnement inconnu: {env_type}")


# Exemple d'utilisation
def main():
    """Exemple d'utilisation de l'environnement"""
    
    # Créer des données de test
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1h')
    n = len(dates)
    
    # Simuler des données OHLCV
    np.random.seed(42)
    returns = np.random.normal(0.0001, 0.01, n)
    close = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': close * (1 + np.random.normal(0, 0.001, n)),
        'high': close * (1 + np.abs(np.random.normal(0, 0.002, n))),
        'low': close * (1 - np.abs(np.random.normal(0, 0.002, n))),
        'close': close,
        'volume': np.random.exponential(1000000, n)
    }, index=dates)
    
    # Créer l'environnement
    env = create_trading_env(
        data,
        env_type='simple',
        initial_balance=100000,
        commission_rate=0.001,
        episode_length=1000
    )
    
    # Test de l'environnement
    obs = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Exécuter quelques steps
    for i in range(10):
        # Action aléatoire
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        
        env.render()
        
        if done:
            break
    
    # Afficher les métriques finales
    metrics = env.get_performance_metrics()
    print(f"\nMétriques finales:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()