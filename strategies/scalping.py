"""
Scalping Strategy - Stratégie de Scalping Haute Fréquence
Capture des profits rapides sur des mouvements de prix minimes

Caractéristiques principales:
- Analyse de la microstructure du marché en temps réel
- Signaux basés sur le momentum court terme
- Latence ultra-faible (<1ms cible)
- Stop-loss automatiques très serrés
- Gestion agressive des positions
Performance cible: 100-1000+ trades/jour, Sharpe 2.5-4.0
"""

import asyncio
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Set, Deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict
from enum import Enum
import numba
import talib

from strategies.base_strategy import (
    BaseStrategy, TradingSignal, SignalType, MarketData,
    Symbol, StrategyId, SignalStrength, Confidence,
    Price, Quantity, MarketRegime, OrderBookSnapshot
)


class MomentumState(Enum):
    """État du momentum du marché"""
    STRONG_UP = "STRONG_UP"
    WEAK_UP = "WEAK_UP"
    NEUTRAL = "NEUTRAL"
    WEAK_DOWN = "WEAK_DOWN"
    STRONG_DOWN = "STRONG_DOWN"


class LiquidityState(Enum):
    """État de la liquidité du marché"""
    HIGH = "HIGH"
    NORMAL = "NORMAL"
    LOW = "LOW"
    CRITICAL = "CRITICAL"


@dataclass
class MicrostructureSignal:
    """Signal basé sur la microstructure du marché"""
    timestamp: datetime
    signal_type: str  # 'momentum', 'imbalance', 'sweep', 'breakout'
    direction: str  # 'long', 'short'
    strength: float  # 0-1
    confidence: float  # 0-1
    expected_move: Decimal  # Mouvement attendu en points
    time_horizon: int  # Secondes
    entry_price: Price
    stop_loss: Price
    take_profit: Price
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalpTrade:
    """Trade de scalping actif"""
    symbol: Symbol
    entry_time: datetime
    entry_price: Price
    quantity: Quantity
    direction: str  # 'long' or 'short'
    stop_loss: Price
    take_profit: Price
    trailing_stop: Optional[Price] = None
    max_profit: Decimal = Decimal("0")
    current_pnl: Decimal = Decimal("0")
    ticks_in_profit: int = 0
    signal_source: str = ""  # Type de signal qui a déclenché le trade
    
    def update_pnl(self, current_price: Price) -> None:
        """Met à jour le P&L et les métriques"""
        if self.direction == 'long':
            self.current_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.current_pnl = (self.entry_price - current_price) * self.quantity
        
        if self.current_pnl > self.max_profit:
            self.max_profit = self.current_pnl
            
        if self.current_pnl > 0:
            self.ticks_in_profit += 1


@dataclass
class OrderFlowMetrics:
    """Métriques du flux d'ordres"""
    buy_volume: Decimal
    sell_volume: Decimal
    buy_trades: int
    sell_trades: int
    large_buy_volume: Decimal  # Ordres > seuil
    large_sell_volume: Decimal
    vwap: Price
    volume_weighted_spread: Decimal
    tick_direction: int  # +1, 0, -1
    cumulative_delta: Decimal
    
    @property
    def volume_imbalance(self) -> float:
        """Ratio d'imbalance du volume"""
        total = self.buy_volume + self.sell_volume
        if total > 0:
            return float((self.buy_volume - self.sell_volume) / total)
        return 0.0
    
    @property
    def trade_imbalance(self) -> float:
        """Ratio d'imbalance des trades"""
        total = self.buy_trades + self.sell_trades
        if total > 0:
            return float((self.buy_trades - self.sell_trades) / total)
        return 0.0


class ScalpingStrategy(BaseStrategy):
    """
    Stratégie de scalping haute fréquence ultra-optimisée
    Conçue pour capturer des mouvements rapides avec une latence minimale
    """
    
    def __init__(
        self,
        strategy_id: StrategyId,
        symbols: List[Symbol],
        data_provider,
        risk_manager,
        config: Optional[Dict[str, Any]] = None
    ):
        # Configuration optimisée pour le scalping HFT
        default_config = {
            # Paramètres de timing
            'tick_window': 20,              # Fenêtre d'analyse en ticks
            'entry_threshold': 0.0002,      # 2 bps de mouvement minimum
            'take_profit_ticks': 5,         # TP en ticks
            'stop_loss_ticks': 3,           # SL en ticks
            'max_hold_time': 60,            # Secondes max par trade
            
            # Gestion des positions
            'max_positions': 3,             # Positions simultanées max
            'position_size_pct': 0.005,     # 0.5% du capital par trade
            'scale_in_enabled': False,      # Pas de scaling pour le scalping
            'partial_exit_enabled': True,   # Sorties partielles
            
            # Filtres de signal
            'min_volume_threshold': 100,    # Volume min pour entrer
            'min_liquidity_depth': 10,      # Profondeur min du carnet
            'max_spread_bps': 5,            # Spread max acceptable
            'momentum_period': 10,          # Période pour le momentum
            
            # Microstructure
            'use_order_flow': True,         # Analyse du flux d'ordres
            'use_tape_reading': True,       # Lecture du tape
            'imbalance_threshold': 0.6,     # Seuil d'imbalance
            'sweep_detection': True,        # Détection des sweeps
            
            # Risk management
            'max_daily_trades': 500,        # Limite journalière
            'max_consecutive_losses': 5,    # Pertes consécutives max
            'trailing_stop_activation': 3,  # Activation en ticks
            'breakeven_threshold': 2,       # Move to breakeven en ticks
            
            # Optimisations
            'use_numba': True,              # Accélération Numba
            'tick_buffer_size': 1000,       # Buffer de ticks
            'orderbook_levels': 10,         # Niveaux du carnet à analyser
            
            # Machine Learning
            'use_ml_filter': True,          # Filtre ML pour les signaux
            'ml_confidence_threshold': 0.7,  # Seuil de confiance ML
            
            'buffer_size': 100,             # Buffer réduit pour la vitesse
            'update_interval': 0.01         # 10ms - très rapide
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(strategy_id, symbols, data_provider, risk_manager, default_config)
        
        # Structures de données optimisées pour la vitesse
        self.tick_buffer: Dict[Symbol, deque] = {
            symbol: deque(maxlen=config.get('tick_buffer_size', 1000))
            for symbol in symbols
        }
        
        self.order_flow_metrics: Dict[Symbol, OrderFlowMetrics] = {}
        self.active_trades: Dict[Symbol, ScalpTrade] = {}
        self.momentum_state: Dict[Symbol, MomentumState] = {}
        self.liquidity_state: Dict[Symbol, LiquidityState] = {}
        
        # Métriques de scalping
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.win_streak = 0
        self.tick_precision: Dict[Symbol, Decimal] = {}
        
        # Caches pour optimisation
        self._momentum_cache: Dict[Symbol, float] = {}
        self._spread_cache: Dict[Symbol, Decimal] = {}
        self._last_signal_time: Dict[Symbol, datetime] = {}
        
        # Buffers circulaires optimisés
        self._price_deltas: Dict[Symbol, np.ndarray] = {}
        self._volume_profile: Dict[Symbol, np.ndarray] = {}
        
        # ML components
        self.signal_predictor = None  # Modèle de prédiction rapide
        self.feature_extractor = None  # Extracteur de features optimisé
    
    async def analyze_market(self, symbol: Symbol) -> Optional[TradingSignal]:
        """
        Analyse ultra-rapide pour détecter les opportunités de scalping
        """
        # Vérifications rapides de pré-filtrage
        if not await self._pre_trade_checks(symbol):
            return None
        
        # Vérifier si nous avons déjà une position
        if symbol in self.active_trades:
            return await self._manage_active_trade(symbol)
        
        # Analyser la microstructure
        microstructure_signals = await self._analyze_microstructure(symbol)
        if not microstructure_signals:
            return None
        
        # Sélectionner le meilleur signal
        best_signal = self._select_best_signal(microstructure_signals)
        if not best_signal:
            return None
        
        # Convertir en signal de trading
        return self._convert_to_trading_signal(symbol, best_signal)
    
    async def _pre_trade_checks(self, symbol: Symbol) -> bool:
        """Vérifications rapides avant analyse approfondie"""
        # Limite de trades journaliers
        if self.daily_trades >= self.config['max_daily_trades']:
            return False
        
        # Pertes consécutives
        if self.consecutive_losses >= self.config['max_consecutive_losses']:
            return False
        
        # Vérifier la liquidité
        liquidity = await self._check_liquidity(symbol)
        if liquidity == LiquidityState.CRITICAL:
            return False
        
        # Vérifier le spread
        spread = await self._get_current_spread(symbol)
        if spread > self.config['max_spread_bps'] / 10000:
            return False
        
        # Cooldown entre signaux
        last_signal = self._last_signal_time.get(symbol)
        if last_signal:
            if (datetime.now(timezone.utc) - last_signal).total_seconds() < 1:
                return False
        
        return True
    
    async def _analyze_microstructure(self, symbol: Symbol) -> List[MicrostructureSignal]:
        """Analyse approfondie de la microstructure pour détecter les signaux"""
        signals = []
        
        # 1. Analyse du momentum
        momentum_signal = await self._detect_momentum_signal(symbol)
        if momentum_signal:
            signals.append(momentum_signal)
        
        # 2. Analyse de l'imbalance du carnet
        imbalance_signal = await self._detect_imbalance_signal(symbol)
        if imbalance_signal:
            signals.append(imbalance_signal)
        
        # 3. Détection des sweeps
        if self.config['sweep_detection']:
            sweep_signal = await self._detect_sweep_signal(symbol)
            if sweep_signal:
                signals.append(sweep_signal)
        
        # 4. Analyse du flux d'ordres
        if self.config['use_order_flow']:
            flow_signal = await self._analyze_order_flow(symbol)
            if flow_signal:
                signals.append(flow_signal)
        
        # 5. Patterns de prix courts
        pattern_signal = await self._detect_micro_patterns(symbol)
        if pattern_signal:
            signals.append(pattern_signal)
        
        # Filtrage ML si activé
        if self.config['use_ml_filter'] and self.signal_predictor:
            signals = await self._ml_filter_signals(signals, symbol)
        
        return signals
    
    async def _detect_momentum_signal(self, symbol: Symbol) -> Optional[MicrostructureSignal]:
        """Détecte les signaux de momentum court terme"""
        prices = self.get_data_series(symbol, 'price', self.config['momentum_period'])
        
        if len(prices) < self.config['momentum_period']:
            return None
        
        # Calcul du momentum avec optimisation Numba si disponible
        if self.config['use_numba']:
            momentum = self._calculate_momentum_numba(prices)
        else:
            momentum = self._calculate_momentum_numpy(prices)
        
        # Mise en cache
        self._momentum_cache[symbol] = momentum
        
        # Déterminer l'état du momentum
        if abs(momentum) < 0.0001:  # Momentum neutre
            self.momentum_state[symbol] = MomentumState.NEUTRAL
            return None
        
        # Signal fort
        if momentum > self.config['entry_threshold']:
            self.momentum_state[symbol] = MomentumState.STRONG_UP
            current_price = Price(prices[-1])
            
            return MicrostructureSignal(
                timestamp=datetime.now(timezone.utc),
                signal_type='momentum',
                direction='long',
                strength=min(momentum / (self.config['entry_threshold'] * 2), 1.0),
                confidence=0.8,
                expected_move=Decimal(str(momentum * prices[-1])),
                time_horizon=30,
                entry_price=current_price,
                stop_loss=Price(current_price - self._get_tick_size(symbol) * self.config['stop_loss_ticks']),
                take_profit=Price(current_price + self._get_tick_size(symbol) * self.config['take_profit_ticks']),
                metadata={'momentum_value': momentum, 'momentum_state': 'STRONG_UP'}
            )
        
        elif momentum < -self.config['entry_threshold']:
            self.momentum_state[symbol] = MomentumState.STRONG_DOWN
            current_price = Price(prices[-1])
            
            return MicrostructureSignal(
                timestamp=datetime.now(timezone.utc),
                signal_type='momentum',
                direction='short',
                strength=min(abs(momentum) / (self.config['entry_threshold'] * 2), 1.0),
                confidence=0.8,
                expected_move=Decimal(str(abs(momentum) * prices[-1])),
                time_horizon=30,
                entry_price=current_price,
                stop_loss=Price(current_price + self._get_tick_size(symbol) * self.config['stop_loss_ticks']),
                take_profit=Price(current_price - self._get_tick_size(symbol) * self.config['take_profit_ticks']),
                metadata={'momentum_value': momentum, 'momentum_state': 'STRONG_DOWN'}
            )
        
        return None
    
    async def _detect_imbalance_signal(self, symbol: Symbol) -> Optional[MicrostructureSignal]:
        """Détecte les déséquilibres dans le carnet d'ordres"""
        order_book = await self._get_order_book_data(symbol)
        if not order_book:
            return None
        
        # Calculer l'imbalance sur plusieurs niveaux
        imbalances = []
        for levels in [1, 3, 5]:
            imbalance = order_book.get_imbalance(levels)
            imbalances.append(imbalance)
        
        # Moyenne pondérée (plus de poids aux niveaux proches)
        weighted_imbalance = (imbalances[0] * 0.5 + imbalances[1] * 0.3 + imbalances[2] * 0.2)
        
        if abs(weighted_imbalance) < self.config['imbalance_threshold']:
            return None
        
        current_price = order_book.bids[0][0] if weighted_imbalance > 0 else order_book.asks[0][0]
        tick_size = self._get_tick_size(symbol)
        
        if weighted_imbalance > self.config['imbalance_threshold']:
            # Plus d'acheteurs - signal long
            return MicrostructureSignal(
                timestamp=datetime.now(timezone.utc),
                signal_type='imbalance',
                direction='long',
                strength=min(weighted_imbalance, 1.0),
                confidence=0.75,
                expected_move=tick_size * Decimal(str(self.config['take_profit_ticks'])),
                time_horizon=20,
                entry_price=current_price,
                stop_loss=Price(current_price - tick_size * self.config['stop_loss_ticks']),
                take_profit=Price(current_price + tick_size * self.config['take_profit_ticks']),
                metadata={'imbalance': weighted_imbalance, 'levels_analyzed': [1, 3, 5]}
            )
        
        else:
            # Plus de vendeurs - signal short
            return MicrostructureSignal(
                timestamp=datetime.now(timezone.utc),
                signal_type='imbalance',
                direction='short',
                strength=min(abs(weighted_imbalance), 1.0),
                confidence=0.75,
                expected_move=tick_size * Decimal(str(self.config['take_profit_ticks'])),
                time_horizon=20,
                entry_price=current_price,
                stop_loss=Price(current_price + tick_size * self.config['stop_loss_ticks']),
                take_profit=Price(current_price - tick_size * self.config['take_profit_ticks']),
                metadata={'imbalance': weighted_imbalance, 'levels_analyzed': [1, 3, 5]}
            )
    
    async def _detect_sweep_signal(self, symbol: Symbol) -> Optional[MicrostructureSignal]:
        """Détecte les sweeps (absorption rapide de liquidité)"""
        # Analyser les changements rapides dans le carnet
        recent_ticks = list(self.tick_buffer[symbol])[-10:]
        if len(recent_ticks) < 10:
            return None
        
        # Détecter les mouvements brusques de prix avec volume élevé
        price_changes = [tick['price'] - recent_ticks[i]['price'] 
                        for i, tick in enumerate(recent_ticks[1:])]
        volumes = [tick['volume'] for tick in recent_ticks]
        
        avg_volume = np.mean(volumes)
        large_volume_threshold = avg_volume * 3  # 3x le volume moyen
        
        # Chercher un mouvement rapide avec gros volume
        for i, (price_change, volume) in enumerate(zip(price_changes, volumes[1:])):
            if volume > large_volume_threshold and abs(price_change) > 0:
                # Sweep détecté
                direction = 'long' if price_change > 0 else 'short'
                current_price = Price(recent_ticks[-1]['price'])
                tick_size = self._get_tick_size(symbol)
                
                return MicrostructureSignal(
                    timestamp=datetime.now(timezone.utc),
                    signal_type='sweep',
                    direction=direction,
                    strength=min(volume / large_volume_threshold, 1.0),
                    confidence=0.85,
                    expected_move=tick_size * Decimal(str(self.config['take_profit_ticks'] * 1.5)),
                    time_horizon=15,
                    entry_price=current_price,
                    stop_loss=Price(
                        current_price - tick_size * self.config['stop_loss_ticks']
                        if direction == 'long'
                        else current_price + tick_size * self.config['stop_loss_ticks']
                    ),
                    take_profit=Price(
                        current_price + tick_size * self.config['take_profit_ticks'] * 1.5
                        if direction == 'long'
                        else current_price - tick_size * self.config['take_profit_ticks'] * 1.5
                    ),
                    metadata={'sweep_volume': float(volume), 'avg_volume': float(avg_volume)}
                )
        
        return None
    
    async def _analyze_order_flow(self, symbol: Symbol) -> Optional[MicrostructureSignal]:
        """Analyse le flux d'ordres pour détecter les patterns"""
        if symbol not in self.order_flow_metrics:
            await self._update_order_flow_metrics(symbol)
        
        metrics = self.order_flow_metrics.get(symbol)
        if not metrics:
            return None
        
        # Analyser les déséquilibres de volume et de trades
        volume_imb = metrics.volume_imbalance
        trade_imb = metrics.trade_imbalance
        
        # Signal basé sur la convergence des métriques
        if abs(volume_imb) > 0.4 and abs(trade_imb) > 0.3:
            if np.sign(volume_imb) == np.sign(trade_imb):
                # Les deux métriques pointent dans la même direction
                direction = 'long' if volume_imb > 0 else 'short'
                strength = (abs(volume_imb) + abs(trade_imb)) / 2
                
                current_price = metrics.vwap
                tick_size = self._get_tick_size(symbol)
                
                return MicrostructureSignal(
                    timestamp=datetime.now(timezone.utc),
                    signal_type='order_flow',
                    direction=direction,
                    strength=min(strength, 1.0),
                    confidence=0.7 + min(strength * 0.2, 0.2),  # 0.7-0.9
                    expected_move=tick_size * Decimal(str(self.config['take_profit_ticks'])),
                    time_horizon=25,
                    entry_price=current_price,
                    stop_loss=Price(
                        current_price - tick_size * self.config['stop_loss_ticks']
                        if direction == 'long'
                        else current_price + tick_size * self.config['stop_loss_ticks']
                    ),
                    take_profit=Price(
                        current_price + tick_size * self.config['take_profit_ticks']
                        if direction == 'long'
                        else current_price - tick_size * self.config['take_profit_ticks']
                    ),
                    metadata={
                        'volume_imbalance': volume_imb,
                        'trade_imbalance': trade_imb,
                        'cumulative_delta': float(metrics.cumulative_delta)
                    }
                )
        
        return None
    
    async def _detect_micro_patterns(self, symbol: Symbol) -> Optional[MicrostructureSignal]:
        """Détecte des patterns de prix très courts (2-5 bougies)"""
        prices = self.get_data_series(symbol, 'price', 10)
        volumes = self.get_data_series(symbol, 'volume', 10)
        
        if len(prices) < 5:
            return None
        
        # Pattern 1: Three-push pattern (micro version)
        if self._detect_three_push(prices[-5:]):
            direction = 'long' if prices[-1] > prices[-5] else 'short'
            current_price = Price(prices[-1])
            tick_size = self._get_tick_size(symbol)
            
            return MicrostructureSignal(
                timestamp=datetime.now(timezone.utc),
                signal_type='pattern',
                direction=direction,
                strength=0.7,
                confidence=0.65,
                expected_move=tick_size * Decimal(str(self.config['take_profit_ticks'])),
                time_horizon=30,
                entry_price=current_price,
                stop_loss=Price(
                    current_price - tick_size * self.config['stop_loss_ticks']
                    if direction == 'long'
                    else current_price + tick_size * self.config['stop_loss_ticks']
                ),
                take_profit=Price(
                    current_price + tick_size * self.config['take_profit_ticks']
                    if direction == 'long'
                    else current_price - tick_size * self.config['take_profit_ticks']
                ),
                metadata={'pattern': 'three_push'}
            )
        
        # Pattern 2: Breakout rapide
        if self._detect_micro_breakout(prices[-5:], volumes[-5:]):
            direction = 'long' if prices[-1] > max(prices[-5:-1]) else 'short'
            current_price = Price(prices[-1])
            tick_size = self._get_tick_size(symbol)
            
            return MicrostructureSignal(
                timestamp=datetime.now(timezone.utc),
                signal_type='pattern',
                direction=direction,
                strength=0.8,
                confidence=0.7,
                expected_move=tick_size * Decimal(str(self.config['take_profit_ticks'] * 1.2)),
                time_horizon=20,
                entry_price=current_price,
                stop_loss=Price(
                    current_price - tick_size * self.config['stop_loss_ticks']
                    if direction == 'long'
                    else current_price + tick_size * self.config['stop_loss_ticks']
                ),
                take_profit=Price(
                    current_price + tick_size * self.config['take_profit_ticks'] * 1.2
                    if direction == 'long'
                    else current_price - tick_size * self.config['take_profit_ticks'] * 1.2
                ),
                metadata={'pattern': 'micro_breakout'}
            )
        
        return None
    
    def _select_best_signal(self, signals: List[MicrostructureSignal]) -> Optional[MicrostructureSignal]:
        """Sélectionne le meilleur signal parmi plusieurs"""
        if not signals:
            return None
        
        # Scorer chaque signal
        scored_signals = []
        for signal in signals:
            score = self._calculate_signal_score(signal)
            scored_signals.append((score, signal))
        
        # Retourner le signal avec le meilleur score
        scored_signals.sort(key=lambda x: x[0], reverse=True)
        best_score, best_signal = scored_signals[0]
        
        # Seuil minimum de score
        if best_score < 0.6:
            return None
        
        return best_signal
    
    def _calculate_signal_score(self, signal: MicrostructureSignal) -> float:
        """Calcule un score composite pour un signal"""
        # Pondération des différents facteurs
        strength_weight = 0.3
        confidence_weight = 0.4
        signal_type_weight = 0.3
        
        # Scores par type de signal (basé sur l'historique de performance)
        signal_type_scores = {
            'sweep': 0.9,
            'momentum': 0.8,
            'order_flow': 0.75,
            'imbalance': 0.7,
            'pattern': 0.65
        }
        
        type_score = signal_type_scores.get(signal.signal_type, 0.5)
        
        score = (
            signal.strength * strength_weight +
            signal.confidence * confidence_weight +
            type_score * signal_type_weight
        )
        
        # Ajustements basés sur le contexte
        if self.momentum_state.get(signal.metadata.get('symbol')) == MomentumState.STRONG_UP and signal.direction == 'long':
            score *= 1.1
        elif self.momentum_state.get(signal.metadata.get('symbol')) == MomentumState.STRONG_DOWN and signal.direction == 'short':
            score *= 1.1
        
        return min(score, 1.0)
    
    def _convert_to_trading_signal(self, symbol: Symbol, micro_signal: MicrostructureSignal) -> TradingSignal:
        """Convertit un signal de microstructure en signal de trading"""
        signal_type = SignalType.BUY if micro_signal.direction == 'long' else SignalType.SELL
        
        # Calculer la quantité basée sur le risque
        risk_amount = self.current_capital * Decimal(str(self.config['position_size_pct']))
        stop_distance = abs(micro_signal.entry_price - micro_signal.stop_loss)
        quantity = Quantity(risk_amount / stop_distance) if stop_distance > 0 else Quantity(Decimal("1"))
        
        trading_signal = TradingSignal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            strength=SignalStrength(micro_signal.strength),
            confidence=Confidence(micro_signal.confidence),
            quantity=quantity,
            price=micro_signal.entry_price,
            stop_loss=micro_signal.stop_loss,
            take_profit=micro_signal.take_profit,
            time_in_force="IOC",  # Immediate or Cancel pour le scalping
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=5),  # Expire rapidement
            metadata={
                'signal_source': micro_signal.signal_type,
                'expected_move': float(micro_signal.expected_move),
                'time_horizon': micro_signal.time_horizon,
                'micro_metadata': micro_signal.metadata
            }
        )
        
        # Enregistrer le temps du signal
        self._last_signal_time[symbol] = datetime.now(timezone.utc)
        
        return trading_signal
    
    async def _manage_active_trade(self, symbol: Symbol) -> Optional[TradingSignal]:
        """Gère un trade actif - sorties et ajustements"""
        trade = self.active_trades.get(symbol)
        if not trade:
            return None
        
        current_price = self.get_latest_data(symbol).price
        trade.update_pnl(current_price)
        
        # Vérifier les conditions de sortie
        exit_signal = None
        
        # 1. Take Profit atteint
        if trade.direction == 'long' and current_price >= trade.take_profit:
            exit_signal = self._create_exit_signal(symbol, trade, "take_profit")
        elif trade.direction == 'short' and current_price <= trade.take_profit:
            exit_signal = self._create_exit_signal(symbol, trade, "take_profit")
        
        # 2. Stop Loss atteint
        elif trade.direction == 'long' and current_price <= trade.stop_loss:
            exit_signal = self._create_exit_signal(symbol, trade, "stop_loss")
        elif trade.direction == 'short' and current_price >= trade.stop_loss:
            exit_signal = self._create_exit_signal(symbol, trade, "stop_loss")
        
        # 3. Temps maximum dépassé
        elif (datetime.now(timezone.utc) - trade.entry_time).total_seconds() > self.config['max_hold_time']:
            exit_signal = self._create_exit_signal(symbol, trade, "time_exit")
        
        # 4. Trailing stop
        elif trade.trailing_stop:
            if trade.direction == 'long' and current_price <= trade.trailing_stop:
                exit_signal = self._create_exit_signal(symbol, trade, "trailing_stop")
            elif trade.direction == 'short' and current_price >= trade.trailing_stop:
                exit_signal = self._create_exit_signal(symbol, trade, "trailing_stop")
        
        # Mise à jour du trailing stop si nécessaire
        if not exit_signal and trade.ticks_in_profit >= self.config['trailing_stop_activation']:
            self._update_trailing_stop(trade, current_price)
        
        # Breakeven stop
        if not exit_signal and trade.ticks_in_profit >= self.config['breakeven_threshold']:
            self._move_to_breakeven(trade)
        
        return exit_signal
    
    def _create_exit_signal(self, symbol: Symbol, trade: ScalpTrade, reason: str) -> TradingSignal:
        """Crée un signal de sortie"""
        signal_type = SignalType.CLOSE_LONG if trade.direction == 'long' else SignalType.CLOSE_SHORT
        
        exit_signal = TradingSignal(
            strategy_id=self.strategy_id,
            symbol=symbol,
            signal_type=signal_type,
            strength=SignalStrength(0.95),  # Forte conviction pour les sorties
            confidence=Confidence(0.95),
            quantity=trade.quantity,
            time_in_force="IOC",
            metadata={
                'exit_reason': reason,
                'pnl': float(trade.current_pnl),
                'ticks_in_profit': trade.ticks_in_profit,
                'hold_time': str(datetime.now(timezone.utc) - trade.entry_time),
                'max_profit': float(trade.max_profit)
            }
        )
        
        # Mettre à jour les métriques
        if trade.current_pnl > 0:
            self.win_streak += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.win_streak = 0
        
        # Supprimer le trade actif
        del self.active_trades[symbol]
        self.daily_trades += 1
        
        return exit_signal
    
    def _update_trailing_stop(self, trade: ScalpTrade, current_price: Price) -> None:
        """Met à jour le trailing stop"""
        tick_size = self._get_tick_size(trade.symbol)
        trailing_distance = tick_size * 2  # 2 ticks de trailing
        
        if trade.direction == 'long':
            new_stop = current_price - trailing_distance
            if not trade.trailing_stop or new_stop > trade.trailing_stop:
                trade.trailing_stop = Price(new_stop)
        else:
            new_stop = current_price + trailing_distance
            if not trade.trailing_stop or new_stop < trade.trailing_stop:
                trade.trailing_stop = Price(new_stop)
    
    def _move_to_breakeven(self, trade: ScalpTrade) -> None:
        """Déplace le stop loss au breakeven"""
        if trade.direction == 'long':
            if trade.stop_loss < trade.entry_price:
                trade.stop_loss = trade.entry_price
        else:
            if trade.stop_loss > trade.entry_price:
                trade.stop_loss = trade.entry_price
    
    # Méthodes utilitaires optimisées
    
    @staticmethod
    @numba.jit(nopython=True)
    def _calculate_momentum_numba(prices: np.ndarray) -> float:
        """Calcul optimisé du momentum avec Numba"""
        if len(prices) < 2:
            return 0.0
        
        # ROC (Rate of Change) sur la période
        roc = (prices[-1] - prices[0]) / prices[0]
        
        # Ajustement par la volatilité
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        if volatility > 0:
            return roc / volatility
        return roc
    
    def _calculate_momentum_numpy(self, prices: np.ndarray) -> float:
        """Calcul du momentum sans Numba"""
        if len(prices) < 2:
            return 0.0
        
        roc = (prices[-1] - prices[0]) / prices[0]
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        if volatility > 0:
            return roc / volatility
        return roc
    
    def _detect_three_push(self, prices: np.ndarray) -> bool:
        """Détecte un pattern three-push"""
        if len(prices) < 5:
            return False
        
        # Chercher 3 pushes dans la même direction
        diffs = np.diff(prices)
        same_direction = np.all(diffs > 0) or np.all(diffs < 0)
        
        if same_direction:
            # Vérifier que chaque push est plus faible
            abs_diffs = np.abs(diffs)
            weakening = all(abs_diffs[i] < abs_diffs[i-1] for i in range(1, len(abs_diffs)))
            return weakening
        
        return False
    
    def _detect_micro_breakout(self, prices: np.ndarray, volumes: np.ndarray) -> bool:
        """Détecte un micro breakout"""
        if len(prices) < 5 or len(volumes) < 5:
            return False
        
        # Range des 4 premières bougies
        range_high = np.max(prices[:-1])
        range_low = np.min(prices[:-1])
        current_price = prices[-1]
        
        # Breakout avec volume
        avg_volume = np.mean(volumes[:-1])
        current_volume = volumes[-1]
        
        is_breakout = (current_price > range_high or current_price < range_low)
        has_volume = current_volume > avg_volume * 1.5
        
        return is_breakout and has_volume
    
    async def _check_liquidity(self, symbol: Symbol) -> LiquidityState:
        """Vérifie l'état de la liquidité"""
        order_book = await self._get_order_book_data(symbol)
        if not order_book:
            return LiquidityState.CRITICAL
        
        # Calculer la profondeur totale
        bid_depth = sum(q for p, q in order_book.bids[:5])
        ask_depth = sum(q for p, q in order_book.asks[:5])
        total_depth = bid_depth + ask_depth
        
        # Seuils de liquidité (à ajuster selon le marché)
        if total_depth < self.config['min_liquidity_depth']:
            return LiquidityState.CRITICAL
        elif total_depth < self.config['min_liquidity_depth'] * 2:
            return LiquidityState.LOW
        elif total_depth > self.config['min_liquidity_depth'] * 5:
            return LiquidityState.HIGH
        else:
            return LiquidityState.NORMAL
    
    async def _get_current_spread(self, symbol: Symbol) -> float:
        """Obtient le spread actuel en basis points"""
        latest = self.get_latest_data(symbol)
        if latest:
            spread = latest.ask - latest.bid
            mid_price = (latest.ask + latest.bid) / 2
            spread_bps = float(spread / mid_price * 10000)
            self._spread_cache[symbol] = spread
            return spread_bps
        return float('inf')
    
    def _get_tick_size(self, symbol: Symbol) -> Decimal:
        """Obtient la taille du tick pour un symbole"""
        if symbol not in self.tick_precision:
            # Détecter automatiquement basé sur les prix
            prices = self.get_data_series(symbol, 'price', 100)
            if len(prices) > 10:
                # Calculer les différences minimales non-nulles
                diffs = np.abs(np.diff(prices))
                non_zero_diffs = diffs[diffs > 0]
                if len(non_zero_diffs) > 0:
                    tick_size = Decimal(str(np.min(non_zero_diffs)))
                    self.tick_precision[symbol] = tick_size
                    return tick_size
            
            # Valeur par défaut
            self.tick_precision[symbol] = Decimal("0.01")
        
        return self.tick_precision[symbol]
    
    async def _update_order_flow_metrics(self, symbol: Symbol) -> None:
        """Met à jour les métriques de flux d'ordres"""
        recent_ticks = list(self.tick_buffer[symbol])[-100:]
        if len(recent_ticks) < 10:
            return
        
        buy_volume = Decimal("0")
        sell_volume = Decimal("0")
        buy_trades = 0
        sell_trades = 0
        
        prices = []
        volumes = []
        
        for tick in recent_ticks:
            if tick.get('side') == 'BUY':
                buy_volume += tick['volume']
                buy_trades += 1
            else:
                sell_volume += tick['volume']
                sell_trades += 1
            
            prices.append(float(tick['price']))
            volumes.append(float(tick['volume']))
        
        # VWAP
        if volumes:
            vwap = Price(Decimal(str(np.average(prices, weights=volumes))))
        else:
            vwap = Price(Decimal(str(np.mean(prices))))
        
        self.order_flow_metrics[symbol] = OrderFlowMetrics(
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            buy_trades=buy_trades,
            sell_trades=sell_trades,
            large_buy_volume=Decimal("0"),  # À implémenter
            large_sell_volume=Decimal("0"),
            vwap=vwap,
            volume_weighted_spread=Decimal("0"),
            tick_direction=0,
            cumulative_delta=buy_volume - sell_volume
        )
    
    async def _get_order_book_data(self, symbol: Symbol) -> Optional[OrderBookSnapshot]:
        """Obtient les données du carnet d'ordres"""
        if self.config.get('use_orderbook', True):
            buffer = self.order_book_buffer.get(symbol)
            if buffer and len(buffer) > 0:
                return buffer[-1]
        return None
    
    async def _ml_filter_signals(
        self, 
        signals: List[MicrostructureSignal], 
        symbol: Symbol
    ) -> List[MicrostructureSignal]:
        """Filtre les signaux avec le ML"""
        # Placeholder pour le filtrage ML
        # En production, utiliserait un modèle entraîné pour prédire la probabilité de succès
        filtered = []
        
        for signal in signals:
            # Simuler une prédiction ML
            ml_confidence = signal.confidence * 0.9  # Placeholder
            
            if ml_confidence >= self.config['ml_confidence_threshold']:
                filtered.append(signal)
        
        return filtered
    
    async def calculate_indicators(self, symbol: Symbol) -> Dict[str, Any]:
        """Calcule les indicateurs de scalping"""
        indicators = {}
        
        # Momentum
        indicators['momentum'] = self._momentum_cache.get(symbol, 0.0)
        indicators['momentum_state'] = self.momentum_state.get(symbol, MomentumState.NEUTRAL).value
        
        # Liquidité
        indicators['liquidity_state'] = (await self._check_liquidity(symbol)).value
        indicators['spread_bps'] = await self._get_current_spread(symbol)
        
        # Order flow
        if symbol in self.order_flow_metrics:
            metrics = self.order_flow_metrics[symbol]
            indicators['volume_imbalance'] = metrics.volume_imbalance
            indicators['trade_imbalance'] = metrics.trade_imbalance
            indicators['cumulative_delta'] = float(metrics.cumulative_delta)
        
        # Performance
        indicators['daily_trades'] = self.daily_trades
        indicators['consecutive_losses'] = self.consecutive_losses
        indicators['win_streak'] = self.win_streak
        indicators['active_trade'] = symbol in self.active_trades
        
        return indicators
    
    def get_required_history_size(self) -> int:
        """Retourne la taille d'historique requise"""
        return max(100, self.config['tick_window'] * 5)
    
    async def _on_initialize(self) -> None:
        """Initialisation spécifique au scalping"""
        self.logger.info(f"Initialisation du scalping HFT pour {len(self.symbols)} symboles")
        
        # Précalculer les tick sizes
        for symbol in self.symbols:
            self._get_tick_size(symbol)
        
        # Initialiser les buffers numpy si Numba est activé
        if self.config['use_numba']:
            for symbol in self.symbols:
                self._price_deltas[symbol] = np.zeros(self.config['tick_window'])
                self._volume_profile[symbol] = np.zeros(100)
    
    async def _save_custom_state(self) -> Dict[str, Any]:
        """Sauvegarde l'état du scalping"""
        return {
            'daily_trades': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'win_streak': self.win_streak,
            'tick_precision': {str(s): str(t) for s, t in self.tick_precision.items()},
            'active_trades': {
                str(symbol): {
                    'direction': trade.direction,
                    'entry_price': float(trade.entry_price),
                    'current_pnl': float(trade.current_pnl),
                    'ticks_in_profit': trade.ticks_in_profit
                }
                for symbol, trade in self.active_trades.items()
            }
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Diagnostics étendus pour le scalping"""
        base_diagnostics = super().get_diagnostics()
        
        base_diagnostics.update({
            'daily_trades': self.daily_trades,
            'consecutive_losses': self.consecutive_losses,
            'win_streak': self.win_streak,
            'trades_remaining': self.config['max_daily_trades'] - self.daily_trades,
            'active_trades': {
                str(symbol): {
                    'direction': trade.direction,
                    'pnl': float(trade.current_pnl),
                    'time_held': str(datetime.now(timezone.utc) - trade.entry_time)
                }
                for symbol, trade in self.active_trades.items()
            },
            'momentum_states': {
                str(symbol): state.value 
                for symbol, state in self.momentum_state.items()
            }
        })
        
        return base_diagnostics


# Test et exemple
if __name__ == "__main__":
    # Configuration pour scalping agressif
    config = {
        'tick_window': 20,
        'entry_threshold': 0.0002,
        'take_profit_ticks': 5,
        'stop_loss_ticks': 3,
        'max_hold_time': 60,
        'max_daily_trades': 500,
        'use_numba': True,
        'update_interval': 0.01  # 10ms
    }
    
    # Symboles liquides pour le scalping
    symbols = [
        Symbol("ES"),     # E-mini S&P 500
        Symbol("NQ"),     # E-mini Nasdaq
        Symbol("EUR/USD"),
        Symbol("BTC-USD")
    ]
    
    print("Configuration du Scalping HFT:")
    print(f"- Symboles: {symbols}")
    print(f"- Fenêtre d'analyse: {config['tick_window']} ticks")
    print(f"- TP/SL: {config['take_profit_ticks']}/{config['stop_loss_ticks']} ticks")
    print(f"- Max trades/jour: {config['max_daily_trades']}")
    print(f"- Latence cible: <{config['update_interval']*1000}ms")