"""
Market Making Strategy - Market Making Dynamique Adaptatif
Fournit de la liquidité en cotant continuellement des prix bid/ask

Caractéristiques principales:
- Spread dynamique basé sur la volatilité et le carnet d'ordres
- Gestion intelligente de l'inventaire avec hedging automatique
- Optimisation des quotes selon la microstructure du marché
- Intégration Deep RL pour l'adaptation en temps réel
Performance cible: 20-35% de rendement annuel (Sharpe 2.0-3.0)
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
import math

from strategies.base_strategy import (
    BaseStrategy, TradingSignal, SignalType, MarketData,
    Symbol, StrategyId, SignalStrength, Confidence,
    Price, Quantity, MarketRegime, OrderBookSnapshot
)


class InventoryState(Enum):
    """État de l'inventaire"""
    BALANCED = "BALANCED"
    LONG_SKEWED = "LONG_SKEWED"
    SHORT_SKEWED = "SHORT_SKEWED"
    CRITICAL_LONG = "CRITICAL_LONG"
    CRITICAL_SHORT = "CRITICAL_SHORT"


@dataclass
class MarketMicrostructure:
    """Analyse de la microstructure du marché"""
    bid_ask_spread: Decimal
    order_book_imbalance: float
    trade_flow_imbalance: float
    volatility_1min: float
    volatility_5min: float
    volume_profile: Dict[Price, Quantity]
    price_levels: List[Price]
    effective_spread: Decimal
    realized_spread: Decimal
    price_impact: float
    lambda_buy: float  # Taux d'arrivée des ordres d'achat
    lambda_sell: float  # Taux d'arrivée des ordres de vente
    
    @property
    def spread_bps(self) -> float:
        """Spread en basis points"""
        mid_price = (self.price_levels[0] + self.price_levels[-1]) / 2
        return float(self.bid_ask_spread / mid_price * 10000)


@dataclass
class InventoryPosition:
    """Position d'inventaire pour un symbole"""
    symbol: Symbol
    quantity: Quantity
    avg_cost: Price
    current_value: Decimal
    target_quantity: Quantity
    max_position: Quantity
    min_position: Quantity
    last_updated: datetime
    
    @property
    def inventory_ratio(self) -> float:
        """Ratio d'inventaire par rapport à la limite"""
        if self.max_position > 0:
            return float(self.quantity / self.max_position)
        return 0.0
    
    @property
    def is_skewed(self) -> bool:
        """Vérifie si l'inventaire est déséquilibré"""
        return abs(self.inventory_ratio) > 0.7
    
    def get_inventory_state(self) -> InventoryState:
        """Détermine l'état de l'inventaire"""
        ratio = self.inventory_ratio
        if abs(ratio) < 0.3:
            return InventoryState.BALANCED
        elif 0.3 <= ratio < 0.7:
            return InventoryState.LONG_SKEWED
        elif -0.7 < ratio <= -0.3:
            return InventoryState.SHORT_SKEWED
        elif ratio >= 0.7:
            return InventoryState.CRITICAL_LONG
        else:
            return InventoryState.CRITICAL_SHORT


@dataclass
class QuoteParameters:
    """Paramètres pour la génération de quotes"""
    base_spread: Decimal
    bid_size: Quantity
    ask_size: Quantity
    bid_offset: Decimal
    ask_offset: Decimal
    skew_factor: float
    urgency_factor: float
    competitive_factor: float
    
    def apply_inventory_skew(self, inventory_ratio: float) -> None:
        """Ajuste les paramètres selon l'inventaire"""
        # Si long, favoriser les ventes (ask plus agressif)
        if inventory_ratio > 0:
            self.ask_offset *= Decimal(str(1 - inventory_ratio * 0.5))
            self.bid_offset *= Decimal(str(1 + inventory_ratio * 0.3))
            self.ask_size *= Quantity(Decimal(str(1 + inventory_ratio)))
            self.bid_size *= Quantity(Decimal(str(1 - inventory_ratio * 0.5)))
        # Si short, favoriser les achats (bid plus agressif)
        else:
            self.bid_offset *= Decimal(str(1 + inventory_ratio * 0.5))
            self.ask_offset *= Decimal(str(1 - inventory_ratio * 0.3))
            self.bid_size *= Quantity(Decimal(str(1 - inventory_ratio)))
            self.ask_size *= Quantity(Decimal(str(1 + inventory_ratio * 0.5)))


class MarketMakingStrategy(BaseStrategy):
    """
    Stratégie de Market Making sophistiquée avec gestion d'inventaire avancée
    Optimisée pour fournir de la liquidité tout en gérant le risque
    """
    
    def __init__(
        self,
        strategy_id: StrategyId,
        symbols: List[Symbol],
        data_provider,
        risk_manager,
        config: Optional[Dict[str, Any]] = None
    ):
        # Configuration par défaut optimisée pour le market making
        default_config = {
            # Paramètres de spread
            'min_spread_bps': 10,          # Spread minimum en basis points
            'target_spread_bps': 25,       # Spread cible
            'max_spread_bps': 100,         # Spread maximum
            
            # Gestion d'inventaire
            'max_inventory_pct': 0.05,     # 5% du capital max par symbole
            'target_inventory': 0,         # Inventaire cible (neutre)
            'inventory_skew_factor': 0.5,  # Facteur d'ajustement pour l'inventaire
            
            # Paramètres de quote
            'quote_levels': 3,             # Nombre de niveaux de prix à coter
            'level_spacing_bps': 5,        # Espacement entre niveaux
            'min_order_size': 0.001,       # Taille minimale d'ordre
            'max_order_size_pct': 0.01,    # 1% du volume journalier max
            
            # Risk management
            'max_drawdown_pct': 0.02,      # 2% drawdown max avant pause
            'stop_loss_pct': 0.01,         # 1% stop loss par position
            'volatility_multiplier': 2.5,   # Multiplicateur pour ajuster au vol
            
            # Machine Learning
            'use_ml_quotes': True,         # Utiliser ML pour optimiser les quotes
            'ml_features': ['spread', 'volume', 'volatility', 'flow'],
            
            # Timing
            'quote_update_interval': 0.1,   # 100ms entre mises à jour
            'inventory_check_interval': 1,  # 1s pour vérifier l'inventaire
            'microstructure_update': 5,     # 5s pour recalculer la microstructure
            
            # Stratégies avancées
            'enable_adverse_selection': True,  # Protection contre la sélection adverse
            'enable_penny_jumping': True,      # Stratégie de penny jumping
            'enable_time_priority': True,      # Gestion de la priorité temporelle
            
            'buffer_size': 1000,
            'orderbook_buffer_size': 200
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(strategy_id, symbols, data_provider, risk_manager, default_config)
        
        # Structures spécifiques au market making
        self.inventory: Dict[Symbol, InventoryPosition] = {}
        self.microstructure: Dict[Symbol, MarketMicrostructure] = {}
        self.active_quotes: Dict[Symbol, List[TradingSignal]] = defaultdict(list)
        self.filled_orders: Dict[Symbol, deque] = {
            symbol: deque(maxlen=1000) for symbol in symbols
        }
        
        # Métriques de market making
        self.spread_capture: Dict[Symbol, Decimal] = defaultdict(Decimal)
        self.volume_traded: Dict[Symbol, Decimal] = defaultdict(Decimal)
        self.adverse_selection_cost: Dict[Symbol, Decimal] = defaultdict(Decimal)
        
        # ML pour l'optimisation des quotes
        self.quote_optimizer = None  # Sera initialisé si ML activé
        self.feature_history: deque = deque(maxlen=5000)
        
        # Caches pour performance
        self._spread_cache: Dict[Symbol, Decimal] = {}
        self._volatility_cache: Dict[Symbol, float] = {}
        self._last_microstructure_update: Dict[Symbol, datetime] = {}
    
    async def analyze_market(self, symbol: Symbol) -> Optional[TradingSignal]:
        """
        Génère des quotes bid/ask optimales pour le market making
        """
        # Vérifier que nous avons assez de données
        if len(self.market_data_buffer[symbol]) < 100:
            return None
        
        # Mettre à jour la microstructure si nécessaire
        await self._update_microstructure(symbol)
        
        # Vérifier l'état de l'inventaire
        inventory_state = await self._check_inventory_state(symbol)
        
        # Générer les paramètres de quote optimaux
        quote_params = await self._calculate_optimal_quotes(symbol, inventory_state)
        
        # Créer les signaux de quote
        signals = await self._generate_quote_signals(symbol, quote_params)
        
        # Retourner le signal principal (bid ou ask selon l'inventaire)
        if signals:
            # Prioriser selon l'inventaire
            if inventory_state in [InventoryState.LONG_SKEWED, InventoryState.CRITICAL_LONG]:
                # Favoriser les ventes
                return next((s for s in signals if s.signal_type == SignalType.SELL), signals[0])
            elif inventory_state in [InventoryState.SHORT_SKEWED, InventoryState.CRITICAL_SHORT]:
                # Favoriser les achats
                return next((s for s in signals if s.signal_type == SignalType.BUY), signals[0])
            else:
                # Neutre - retourner le plus fort
                return max(signals, key=lambda s: s.strength * s.confidence)
        
        return None
    
    async def _update_microstructure(self, symbol: Symbol) -> None:
        """Met à jour l'analyse de microstructure du marché"""
        now = datetime.now(timezone.utc)
        
        # Vérifier si mise à jour nécessaire
        last_update = self._last_microstructure_update.get(symbol, datetime.min.replace(tzinfo=timezone.utc))
        if (now - last_update).total_seconds() < self.config['microstructure_update']:
            return
        
        # Obtenir le carnet d'ordres
        order_book = await self._get_order_book_data(symbol)
        if not order_book:
            return
        
        # Calculer les métriques de microstructure
        microstructure = await self._calculate_microstructure_metrics(symbol, order_book)
        self.microstructure[symbol] = microstructure
        self._last_microstructure_update[symbol] = now
    
    async def _calculate_microstructure_metrics(
        self, 
        symbol: Symbol, 
        order_book: OrderBookSnapshot
    ) -> MarketMicrostructure:
        """Calcule les métriques détaillées de microstructure"""
        # Spread de base
        best_bid = order_book.bids[0][0] if order_book.bids else Price(Decimal("0"))
        best_ask = order_book.asks[0][0] if order_book.asks else Price(Decimal("0"))
        bid_ask_spread = best_ask - best_bid
        
        # Imbalance du carnet
        order_book_imbalance = order_book.get_imbalance(levels=5)
        
        # Calcul de la volatilité
        prices = self.get_data_series(symbol, 'price', 60)  # 1 minute
        volatility_1min = float(np.std(prices[-60:])) if len(prices) >= 60 else 0.0
        volatility_5min = float(np.std(prices[-300:])) if len(prices) >= 300 else 0.0
        
        # Volume profile
        volume_profile = self._calculate_volume_profile(symbol)
        
        # Trade flow imbalance
        trade_flow_imbalance = await self._calculate_trade_flow_imbalance(symbol)
        
        # Taux d'arrivée des ordres (simplifié)
        recent_trades = self.filled_orders.get(symbol, deque())
        buy_count = sum(1 for t in recent_trades if t.get('side') == 'BUY')
        sell_count = len(recent_trades) - buy_count
        
        lambda_buy = buy_count / 60 if recent_trades else 0.1  # Par minute
        lambda_sell = sell_count / 60 if recent_trades else 0.1
        
        # Effective spread (simplifié)
        effective_spread = bid_ask_spread * Decimal("0.8")  # 80% du quoted spread
        
        # Price impact
        total_bid_depth = sum(q for p, q in order_book.bids[:10])
        total_ask_depth = sum(q for p, q in order_book.asks[:10])
        avg_depth = (total_bid_depth + total_ask_depth) / 2
        
        price_impact = float(bid_ask_spread / 2 / avg_depth) if avg_depth > 0 else 0.0
        
        return MarketMicrostructure(
            bid_ask_spread=bid_ask_spread,
            order_book_imbalance=order_book_imbalance,
            trade_flow_imbalance=trade_flow_imbalance,
            volatility_1min=volatility_1min,
            volatility_5min=volatility_5min,
            volume_profile=volume_profile,
            price_levels=[best_bid, best_ask],
            effective_spread=effective_spread,
            realized_spread=effective_spread * Decimal("0.9"),
            price_impact=price_impact,
            lambda_buy=lambda_buy,
            lambda_sell=lambda_sell
        )
    
    async def _check_inventory_state(self, symbol: Symbol) -> InventoryState:
        """Vérifie et met à jour l'état de l'inventaire"""
        if symbol not in self.inventory:
            # Initialiser l'inventaire
            max_position = Quantity(
                self.current_capital * Decimal(str(self.config['max_inventory_pct']))
                / self.get_latest_data(symbol).price
            )
            
            self.inventory[symbol] = InventoryPosition(
                symbol=symbol,
                quantity=Quantity(Decimal("0")),
                avg_cost=Price(Decimal("0")),
                current_value=Decimal("0"),
                target_quantity=Quantity(Decimal("0")),
                max_position=max_position,
                min_position=-max_position,
                last_updated=datetime.now(timezone.utc)
            )
        
        inv = self.inventory[symbol]
        
        # Mettre à jour la valeur actuelle
        current_price = self.get_latest_data(symbol).price
        inv.current_value = inv.quantity * current_price
        inv.last_updated = datetime.now(timezone.utc)
        
        return inv.get_inventory_state()
    
    async def _calculate_optimal_quotes(
        self, 
        symbol: Symbol, 
        inventory_state: InventoryState
    ) -> QuoteParameters:
        """Calcule les paramètres optimaux pour les quotes"""
        microstructure = self.microstructure.get(symbol)
        if not microstructure:
            # Paramètres par défaut
            return self._get_default_quote_parameters(symbol)
        
        # Spread de base ajusté à la volatilité
        vol_adjustment = 1 + (microstructure.volatility_1min * self.config['volatility_multiplier'])
        base_spread = Decimal(str(self.config['target_spread_bps'] / 10000)) * vol_adjustment
        
        # Ajustement selon l'imbalance du carnet
        if abs(microstructure.order_book_imbalance) > 0.3:
            # Marché déséquilibré - élargir le spread
            base_spread *= Decimal("1.2")
        
        # Tailles de quote basées sur la profondeur du marché
        avg_order_size = self._calculate_average_order_size(symbol)
        bid_size = ask_size = min(
            avg_order_size * Decimal("0.5"),  # 50% de la taille moyenne
            Quantity(self.current_capital * Decimal(str(self.config['max_order_size_pct'])))
        )
        
        # Offsets initiaux
        current_price = self.get_latest_data(symbol).price
        bid_offset = current_price * base_spread / 2
        ask_offset = current_price * base_spread / 2
        
        # Facteur de skew basé sur l'inventaire
        inv = self.inventory[symbol]
        skew_factor = inv.inventory_ratio * self.config['inventory_skew_factor']
        
        # Facteur d'urgence (basé sur le P&L et le drawdown)
        urgency_factor = self._calculate_urgency_factor(symbol)
        
        # Facteur de compétitivité (basé sur notre position dans la queue)
        competitive_factor = await self._calculate_competitive_factor(symbol)
        
        params = QuoteParameters(
            base_spread=base_spread,
            bid_size=bid_size,
            ask_size=ask_size,
            bid_offset=bid_offset,
            ask_offset=ask_offset,
            skew_factor=skew_factor,
            urgency_factor=urgency_factor,
            competitive_factor=competitive_factor
        )
        
        # Appliquer l'ajustement d'inventaire
        params.apply_inventory_skew(inv.inventory_ratio)
        
        # ML optimization si activé
        if self.config['use_ml_quotes'] and self.quote_optimizer:
            params = await self._ml_optimize_quotes(params, symbol, microstructure)
        
        return params
    
    async def _generate_quote_signals(
        self, 
        symbol: Symbol, 
        params: QuoteParameters
    ) -> List[TradingSignal]:
        """Génère les signaux de quote pour plusieurs niveaux"""
        signals = []
        current_price = self.get_latest_data(symbol).price
        
        # Générer les quotes pour chaque niveau
        for level in range(self.config['quote_levels']):
            level_adjustment = Decimal(str(1 + level * self.config['level_spacing_bps'] / 10000))
            
            # Quote BID
            bid_price = current_price - params.bid_offset * level_adjustment
            bid_signal = TradingSignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=SignalStrength(0.8 - level * 0.2),  # Décroissant par niveau
                confidence=Confidence(0.9 - level * 0.1),
                quantity=params.bid_size * Decimal(str(1 - level * 0.3)),  # Taille décroissante
                price=Price(bid_price),
                time_in_force="IOC",  # Immediate or Cancel pour market making
                metadata={
                    'quote_type': 'bid',
                    'level': level,
                    'spread_bps': float(params.base_spread * 10000),
                    'inventory_state': self.inventory[symbol].get_inventory_state().value,
                    'microstructure': {
                        'volatility': self.microstructure[symbol].volatility_1min,
                        'imbalance': self.microstructure[symbol].order_book_imbalance
                    } if symbol in self.microstructure else {}
                }
            )
            signals.append(bid_signal)
            
            # Quote ASK
            ask_price = current_price + params.ask_offset * level_adjustment
            ask_signal = TradingSignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=SignalStrength(0.8 - level * 0.2),
                confidence=Confidence(0.9 - level * 0.1),
                quantity=params.ask_size * Decimal(str(1 - level * 0.3)),
                price=Price(ask_price),
                time_in_force="IOC",
                metadata={
                    'quote_type': 'ask',
                    'level': level,
                    'spread_bps': float(params.base_spread * 10000),
                    'inventory_state': self.inventory[symbol].get_inventory_state().value,
                    'microstructure': {
                        'volatility': self.microstructure[symbol].volatility_1min,
                        'imbalance': self.microstructure[symbol].order_book_imbalance
                    } if symbol in self.microstructure else {}
                }
            )
            signals.append(ask_signal)
        
        # Ajouter des ordres de hedging si l'inventaire est critique
        inv_state = self.inventory[symbol].get_inventory_state()
        if inv_state == InventoryState.CRITICAL_LONG:
            # Ordre de vente agressif pour réduire l'inventaire
            hedge_signal = TradingSignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.SELL,
                strength=SignalStrength(0.95),
                confidence=Confidence(0.95),
                quantity=self.inventory[symbol].quantity * Decimal("0.3"),  # 30% de l'inventaire
                price=Price(current_price * Decimal("0.998")),  # Prix agressif
                time_in_force="FOK",  # Fill or Kill
                metadata={'order_type': 'inventory_hedge', 'reason': 'critical_long'}
            )
            signals.append(hedge_signal)
            
        elif inv_state == InventoryState.CRITICAL_SHORT:
            # Ordre d'achat agressif
            hedge_signal = TradingSignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=SignalStrength(0.95),
                confidence=Confidence(0.95),
                quantity=abs(self.inventory[symbol].quantity) * Decimal("0.3"),
                price=Price(current_price * Decimal("1.002")),
                time_in_force="FOK",
                metadata={'order_type': 'inventory_hedge', 'reason': 'critical_short'}
            )
            signals.append(hedge_signal)
        
        return signals
    
    def _calculate_volume_profile(self, symbol: Symbol) -> Dict[Price, Quantity]:
        """Calcule le profil de volume par niveau de prix"""
        profile = defaultdict(Decimal)
        
        # Analyser les trades récents
        for trade in self.filled_orders.get(symbol, []):
            price_level = Price(round(float(trade['price']), 2))  # Arrondir au centime
            profile[price_level] += trade['quantity']
        
        return dict(profile)
    
    async def _calculate_trade_flow_imbalance(self, symbol: Symbol) -> float:
        """Calcule l'imbalance du flux de trades"""
        recent_trades = list(self.filled_orders.get(symbol, []))[-100:]  # 100 derniers trades
        
        if not recent_trades:
            return 0.0
        
        buy_volume = sum(t['quantity'] for t in recent_trades if t['side'] == 'BUY')
        sell_volume = sum(t['quantity'] for t in recent_trades if t['side'] == 'SELL')
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            return float((buy_volume - sell_volume) / total_volume)
        return 0.0
    
    def _calculate_average_order_size(self, symbol: Symbol) -> Quantity:
        """Calcule la taille moyenne des ordres récents"""
        recent_trades = list(self.filled_orders.get(symbol, []))[-50:]
        
        if recent_trades:
            avg_size = sum(t['quantity'] for t in recent_trades) / len(recent_trades)
            return Quantity(avg_size)
        
        # Valeur par défaut
        return Quantity(Decimal("1"))
    
    def _calculate_urgency_factor(self, symbol: Symbol) -> float:
        """Calcule le facteur d'urgence basé sur le P&L et les métriques"""
        # Vérifier le drawdown
        if hasattr(self, 'metrics') and self.metrics.max_drawdown > self.config['max_drawdown_pct']:
            return 1.5  # Plus urgent de clôturer des positions
        
        # Vérifier le P&L de la journée
        daily_pnl = self.metrics.total_pnl if hasattr(self, 'metrics') else Decimal("0")
        if daily_pnl < -self.current_capital * Decimal(str(self.config['stop_loss_pct'])):
            return 2.0  # Très urgent
        
        return 1.0  # Normal
    
    async def _calculate_competitive_factor(self, symbol: Symbol) -> float:
        """Calcule notre position compétitive dans le carnet d'ordres"""
        order_book = await self._get_order_book_data(symbol)
        if not order_book:
            return 1.0
        
        # Vérifier si nos prix sont compétitifs
        our_quotes = self.active_quotes.get(symbol, [])
        if not our_quotes:
            return 1.0
        
        # Simplification: vérifier si nous sommes dans le top 3
        best_bid = order_book.bids[0][0] if order_book.bids else Price(Decimal("0"))
        best_ask = order_book.asks[0][0] if order_book.asks else Price(Decimal("0"))
        
        competitive_score = 1.0
        for quote in our_quotes:
            if quote.signal_type == SignalType.BUY and quote.price >= best_bid:
                competitive_score *= 0.8  # Nous sommes compétitifs
            elif quote.signal_type == SignalType.SELL and quote.price <= best_ask:
                competitive_score *= 0.8
        
        return competitive_score
    
    async def _get_order_book_data(self, symbol: Symbol) -> Optional[OrderBookSnapshot]:
        """Obtient les données du carnet d'ordres"""
        if self.config.get('use_orderbook', True):
            buffer = self.order_book_buffer.get(symbol)
            if buffer and len(buffer) > 0:
                return buffer[-1]
        
        # Fallback: créer un snapshot basique
        latest_data = self.get_latest_data(symbol)
        if latest_data:
            return OrderBookSnapshot(
                symbol=symbol,
                timestamp=latest_data.timestamp,
                bids=[(latest_data.bid, latest_data.bid_size)],
                asks=[(latest_data.ask, latest_data.ask_size)]
            )
        
        return None
    
    def _get_default_quote_parameters(self, symbol: Symbol) -> QuoteParameters:
        """Retourne des paramètres de quote par défaut"""
        current_price = self.get_latest_data(symbol).price
        base_spread = current_price * Decimal(str(self.config['target_spread_bps'] / 10000))
        
        return QuoteParameters(
            base_spread=base_spread,
            bid_size=Quantity(Decimal("1")),
            ask_size=Quantity(Decimal("1")),
            bid_offset=base_spread / 2,
            ask_offset=base_spread / 2,
            skew_factor=0.0,
            urgency_factor=1.0,
            competitive_factor=1.0
        )
    
    async def _ml_optimize_quotes(
        self,
        params: QuoteParameters,
        symbol: Symbol,
        microstructure: MarketMicrostructure
    ) -> QuoteParameters:
        """Optimise les quotes avec le machine learning"""
        # Placeholder pour l'optimisation ML
        # En production, utiliserait un modèle DRL entraîné
        
        # Features pour le ML
        features = {
            'volatility': microstructure.volatility_1min,
            'order_book_imbalance': microstructure.order_book_imbalance,
            'trade_flow_imbalance': microstructure.trade_flow_imbalance,
            'inventory_ratio': self.inventory[symbol].inventory_ratio,
            'spread_bps': microstructure.spread_bps,
            'lambda_ratio': microstructure.lambda_buy / (microstructure.lambda_sell + 1e-6)
        }
        
        # Stocker pour l'entraînement futur
        self.feature_history.append({
            'timestamp': datetime.now(timezone.utc),
            'features': features,
            'params': params,
            'symbol': symbol
        })
        
        # Ajustements simples basés sur les features
        if microstructure.volatility_1min > 0.02:  # Haute volatilité
            params.base_spread *= Decimal("1.3")
        
        if abs(microstructure.order_book_imbalance) > 0.5:  # Fort déséquilibre
            params.base_spread *= Decimal("1.2")
            # Ajuster les tailles selon le déséquilibre
            if microstructure.order_book_imbalance > 0:  # Plus d'acheteurs
                params.ask_size *= Decimal("1.5")
                params.bid_size *= Decimal("0.7")
            else:
                params.bid_size *= Decimal("1.5")
                params.ask_size *= Decimal("0.7")
        
        return params
    
    def update_filled_order(
        self, 
        symbol: Symbol, 
        side: str, 
        price: Price, 
        quantity: Quantity,
        timestamp: datetime
    ) -> None:
        """Met à jour l'inventaire et les métriques après un ordre rempli"""
        # Enregistrer le trade
        self.filled_orders[symbol].append({
            'side': side,
            'price': price,
            'quantity': quantity,
            'timestamp': timestamp
        })
        
        # Mettre à jour l'inventaire
        inv = self.inventory[symbol]
        if side == 'BUY':
            # Calcul du nouveau coût moyen
            total_cost = inv.avg_cost * inv.quantity + price * quantity
            inv.quantity += quantity
            inv.avg_cost = Price(total_cost / inv.quantity) if inv.quantity > 0 else price
        else:  # SELL
            inv.quantity -= quantity
            # Si position inversée, recalculer le coût moyen
            if inv.quantity < 0:
                inv.avg_cost = price
        
        # Mettre à jour les métriques
        self.volume_traded[symbol] += quantity
        
        # Calculer le spread capturé (simplifié)
        if len(self.filled_orders[symbol]) >= 2:
            last_two = list(self.filled_orders[symbol])[-2:]
            if last_two[0]['side'] != last_two[1]['side']:
                spread_captured = abs(last_two[1]['price'] - last_two[0]['price'])
                self.spread_capture[symbol] += spread_captured
    
    async def calculate_indicators(self, symbol: Symbol) -> Dict[str, Any]:
        """Calcule les indicateurs spécifiques au market making"""
        indicators = {}
        
        # Indicateurs d'inventaire
        if symbol in self.inventory:
            inv = self.inventory[symbol]
            indicators['inventory_quantity'] = float(inv.quantity)
            indicators['inventory_ratio'] = inv.inventory_ratio
            indicators['inventory_value'] = float(inv.current_value)
            indicators['inventory_state'] = inv.get_inventory_state().value
        
        # Indicateurs de microstructure
        if symbol in self.microstructure:
            micro = self.microstructure[symbol]
            indicators['bid_ask_spread_bps'] = micro.spread_bps
            indicators['order_book_imbalance'] = micro.order_book_imbalance
            indicators['volatility_1min'] = micro.volatility_1min
            indicators['price_impact'] = micro.price_impact
        
        # Métriques de performance
        indicators['volume_traded'] = float(self.volume_traded.get(symbol, 0))
        indicators['spread_capture'] = float(self.spread_capture.get(symbol, 0))
        indicators['active_quotes'] = len(self.active_quotes.get(symbol, []))
        
        return indicators
    
    def get_required_history_size(self) -> int:
        """Retourne la taille d'historique requise"""
        return 300  # 5 minutes à 1 seconde d'intervalle
    
    async def _on_initialize(self) -> None:
        """Hook d'initialisation spécifique"""
        self.logger.info(f"Initialisation du market making pour {len(self.symbols)} symboles")
        
        # Initialiser l'inventaire pour chaque symbole
        for symbol in self.symbols:
            await self._check_inventory_state(symbol)
        
        # Démarrer la tâche de mise à jour des quotes
        if not hasattr(self, '_quote_update_task'):
            self._quote_update_task = asyncio.create_task(self._quote_update_loop())
    
    async def _quote_update_loop(self) -> None:
        """Boucle de mise à jour continue des quotes"""
        while self._running:
            try:
                for symbol in self.symbols:
                    # Annuler les anciennes quotes
                    await self._cancel_stale_quotes(symbol)
                    
                    # Générer de nouvelles quotes si nécessaire
                    if self.state == StrategyState.RUNNING:
                        signal = await self.analyze_market(symbol)
                        if signal:
                            await self._process_signal(signal)
                
                await asyncio.sleep(self.config['quote_update_interval'])
                
            except Exception as e:
                self.logger.error(f"Erreur dans la boucle de quotes: {str(e)}")
    
    async def _cancel_stale_quotes(self, symbol: Symbol) -> None:
        """Annule les quotes périmées"""
        active_quotes = self.active_quotes.get(symbol, [])
        current_time = datetime.now(timezone.utc)
        
        # Filtrer les quotes encore valides
        valid_quotes = []
        for quote in active_quotes:
            age = (current_time - quote.generated_at).total_seconds()
            if age < 1.0:  # Quotes valides pendant 1 seconde
                valid_quotes.append(quote)
        
        self.active_quotes[symbol] = valid_quotes
    
    async def _save_custom_state(self) -> Dict[str, Any]:
        """Sauvegarde l'état spécifique au market making"""
        return {
            'inventory': {
                str(symbol): {
                    'quantity': float(inv.quantity),
                    'avg_cost': float(inv.avg_cost),
                    'inventory_state': inv.get_inventory_state().value
                }
                for symbol, inv in self.inventory.items()
            },
            'volume_traded': {str(s): float(v) for s, v in self.volume_traded.items()},
            'spread_capture': {str(s): float(v) for s, v in self.spread_capture.items()},
            'feature_history_size': len(self.feature_history)
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Retourne des diagnostics étendus"""
        base_diagnostics = super().get_diagnostics()
        
        # Résumé de l'inventaire
        inventory_summary = {}
        for symbol, inv in self.inventory.items():
            inventory_summary[str(symbol)] = {
                'quantity': float(inv.quantity),
                'ratio': inv.inventory_ratio,
                'state': inv.get_inventory_state().value,
                'value': float(inv.current_value)
            }
        
        # Métriques de market making
        total_volume = sum(self.volume_traded.values())
        total_spread = sum(self.spread_capture.values())
        
        base_diagnostics.update({
            'inventory_summary': inventory_summary,
            'total_volume_traded': float(total_volume),
            'total_spread_capture': float(total_spread),
            'avg_spread_per_trade': float(total_spread / total_volume) if total_volume > 0 else 0,
            'active_quotes_total': sum(len(q) for q in self.active_quotes.values()),
            'microstructure_data': {
                str(symbol): {
                    'spread_bps': micro.spread_bps,
                    'volatility': micro.volatility_1min,
                    'imbalance': micro.order_book_imbalance
                }
                for symbol, micro in self.microstructure.items()
            }
        })
        
        return base_diagnostics


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration pour le market making
    config = {
        'min_spread_bps': 10,
        'target_spread_bps': 25,
        'max_spread_bps': 100,
        'max_inventory_pct': 0.05,
        'quote_levels': 3,
        'quote_update_interval': 0.1,
        'use_ml_quotes': True
    }
    
    # Symboles liquides pour le market making
    symbols = [
        Symbol("BTC-USD"),
        Symbol("ETH-USD"),
        Symbol("BNB-USD")
    ]
    
    print("Configuration du Market Making:")
    print(f"- Symboles: {symbols}")
    print(f"- Spread cible: {config['target_spread_bps']} bps")
    print(f"- Niveaux de quotes: {config['quote_levels']}")
    print(f"- Inventaire max: {config['max_inventory_pct']*100}% du capital")