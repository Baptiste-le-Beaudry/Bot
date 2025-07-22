"""
Position Sizer - Calcul Intelligent de la Taille des Positions
==============================================================

Ce module implémente des algorithmes sophistiqués pour déterminer la taille
optimale des positions en fonction du risque, de la volatilité et de la
performance historique. Optimisé pour maximiser les rendements tout en
contrôlant le risque.

Méthodes supportées:
- Kelly Criterion (standard et fractionnaire)
- Volatility Targeting avec ajustement dynamique
- Risk Parity pour allocation multi-stratégies
- Fixed Fractional pour simplicité
- Machine Learning pour adaptation continue

Architecture:
- Calculs vectorisés avec NumPy pour performance
- Cache intelligent des calculs coûteux
- Integration avec portfolio et risk management
- Support multi-actifs et multi-stratégies
- Limites de sécurité configurables

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import asyncio
import numpy as np
import pandas as pd
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_DOWN
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Protocol
import json
from functools import lru_cache
import warnings

# Imports internes
from core.portfolio_manager import Symbol, Price, Quantity, Capital
from strategies.base_strategy import TradingSignal, SignalType
from utils.logger import get_structured_logger
from utils.metrics import MetricsCalculator, PerformanceMetrics

# Suppression des warnings NumPy pour division par zéro
warnings.filterwarnings('ignore', category=RuntimeWarning)


class SizingMethod(Enum):
    """Méthodes de calcul de taille de position"""
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_TARGET = "volatility_target"
    RISK_PARITY = "risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    MACHINE_LEARNING = "machine_learning"
    MARTINGALE = "martingale"  # Dangereux, désactivé par défaut
    ANTI_MARTINGALE = "anti_martingale"


class RiskLevel(Enum):
    """Niveaux de risque pour ajustement dynamique"""
    CONSERVATIVE = "conservative"   # 0.5x taille normale
    MODERATE = "moderate"          # 1.0x taille normale
    AGGRESSIVE = "aggressive"      # 1.5x taille normale
    VERY_AGGRESSIVE = "very_aggressive"  # 2.0x taille normale


@dataclass
class PositionSizeResult:
    """Résultat du calcul de taille de position"""
    quantity: Quantity
    risk_amount: Decimal
    position_value: Decimal
    leverage_used: float
    method_used: SizingMethod
    confidence_score: float  # 0-1, confiance dans le calcul
    risk_adjusted: bool
    adjustments_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_valid(self) -> bool:
        """Vérifie si la taille est valide"""
        return self.quantity > 0 and self.confidence_score > 0.5


@dataclass
class MarketContext:
    """Contexte de marché pour le sizing"""
    symbol: Symbol
    current_price: Price
    volatility: float  # Volatilité annualisée
    liquidity_score: float  # 0-1, liquidité du marché
    spread_pct: float  # Spread en pourcentage
    correlation_matrix: Optional[np.ndarray] = None
    regime: str = "normal"  # normal, trending, volatile, crisis


@dataclass
class PortfolioContext:
    """Contexte du portefeuille pour le sizing"""
    total_capital: Capital
    available_capital: Capital
    current_positions: Dict[Symbol, Quantity]
    position_values: Dict[Symbol, Decimal]
    total_exposure: Decimal
    current_leverage: float
    recent_performance: PerformanceMetrics
    correlation_exposure: float  # Exposition aux actifs corrélés


class PositionSizer:
    """
    Gestionnaire principal pour le calcul de taille de positions
    avec support multi-méthodes et ajustements dynamiques
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        metrics_calculator: Optional[MetricsCalculator] = None
    ):
        self.config = config or {}
        self.metrics_calculator = metrics_calculator or MetricsCalculator()
        
        # Logger
        self.logger = get_structured_logger(
            "position_sizer",
            module="risk"
        )
        
        # Configuration par défaut
        self._setup_default_config()
        
        # État et historique
        self.position_history: Dict[Symbol, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.win_loss_ratios: Dict[Symbol, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )
        self.volatility_history: Dict[Symbol, deque] = defaultdict(
            lambda: deque(maxlen=500)
        )
        
        # Cache pour calculs coûteux
        self._kelly_cache: Dict[str, Tuple[float, datetime]] = {}
        self._correlation_cache: Optional[Tuple[np.ndarray, datetime]] = None
        
        # Statistiques
        self.sizing_stats = defaultdict(int)
        
        self.logger.info("position_sizer_initialized", config=self.config)
    
    def _setup_default_config(self) -> None:
        """Configure les paramètres par défaut"""
        defaults = {
            # Limites de risque
            'max_risk_per_trade': 0.02,      # 2% max par trade
            'max_position_size_pct': 0.10,    # 10% max du capital
            'max_leverage': 2.0,              # Levier max
            'max_correlation_exposure': 0.60,  # 60% max en actifs corrélés
            
            # Paramètres des méthodes
            'kelly_fraction': 0.25,           # Kelly conservateur (25%)
            'target_volatility': 0.12,        # 12% volatilité annuelle cible
            'min_position_size': 100,         # Taille minimum en USD
            'round_lot_size': 1,              # Arrondir aux unités
            
            # Ajustements dynamiques
            'enable_volatility_adjustment': True,
            'enable_drawdown_adjustment': True,
            'enable_correlation_adjustment': True,
            'enable_liquidity_adjustment': True,
            
            # Fenêtres de calcul
            'volatility_lookback_days': 20,
            'performance_lookback_days': 30,
            'correlation_lookback_days': 60,
            
            # Méthode par défaut
            'default_method': SizingMethod.VOLATILITY_TARGET.value,
            'fallback_method': SizingMethod.FIXED_FRACTIONAL.value,
            
            # Sécurités
            'enable_martingale': False,  # Désactivé par défaut
            'max_consecutive_losses_scaling': 3,  # Limite anti-martingale
            'emergency_reduction_factor': 0.5,  # Réduction d'urgence
        }
        
        # Fusionner avec la config fournie
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    async def calculate_position_size(
        self,
        signal: TradingSignal,
        market_context: MarketContext,
        portfolio_context: PortfolioContext,
        method: Optional[SizingMethod] = None
    ) -> PositionSizeResult:
        """
        Calcule la taille optimale de position pour un signal donné
        
        Args:
            signal: Signal de trading
            market_context: Contexte du marché
            portfolio_context: Contexte du portefeuille
            method: Méthode à utiliser (optionnel)
            
        Returns:
            PositionSizeResult avec la taille calculée
        """
        start_time = datetime.now(timezone.utc)
        
        # Sélectionner la méthode
        if not method:
            method = self._select_sizing_method(signal, market_context)
        
        self.logger.debug(
            "calculating_position_size",
            symbol=signal.symbol,
            method=method.value,
            signal_strength=signal.strength
        )
        
        try:
            # Calculer la taille de base selon la méthode
            if method == SizingMethod.FIXED_FRACTIONAL:
                result = await self._fixed_fractional_sizing(
                    signal, market_context, portfolio_context
                )
            elif method == SizingMethod.KELLY_CRITERION:
                result = await self._kelly_criterion_sizing(
                    signal, market_context, portfolio_context
                )
            elif method == SizingMethod.VOLATILITY_TARGET:
                result = await self._volatility_target_sizing(
                    signal, market_context, portfolio_context
                )
            elif method == SizingMethod.RISK_PARITY:
                result = await self._risk_parity_sizing(
                    signal, market_context, portfolio_context
                )
            elif method == SizingMethod.EQUAL_WEIGHT:
                result = await self._equal_weight_sizing(
                    signal, market_context, portfolio_context
                )
            elif method == SizingMethod.ANTI_MARTINGALE:
                result = await self._anti_martingale_sizing(
                    signal, market_context, portfolio_context
                )
            else:
                # Fallback
                result = await self._fixed_fractional_sizing(
                    signal, market_context, portfolio_context
                )
            
            # Appliquer les ajustements
            result = await self._apply_adjustments(
                result, signal, market_context, portfolio_context
            )
            
            # Appliquer les limites de sécurité
            result = self._apply_safety_limits(
                result, market_context, portfolio_context
            )
            
            # Enregistrer dans l'historique
            self._record_position_size(signal.symbol, result)
            
            # Métriques
            calc_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.logger.info(
                "position_size_calculated",
                symbol=signal.symbol,
                quantity=float(result.quantity),
                risk_amount=float(result.risk_amount),
                method=result.method_used.value,
                confidence=result.confidence_score,
                calc_time_ms=calc_time * 1000
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "position_size_calculation_error",
                symbol=signal.symbol,
                method=method.value,
                error=str(e)
            )
            
            # Retourner une taille sûre en cas d'erreur
            return self._get_safe_fallback_size(
                signal, market_context, portfolio_context
            )
    
    def _select_sizing_method(
        self,
        signal: TradingSignal,
        market_context: MarketContext
    ) -> SizingMethod:
        """Sélectionne la méthode appropriée selon le contexte"""
        # Logique de sélection basée sur le contexte
        
        # Si haute volatilité, préférer volatility targeting
        if market_context.volatility > 0.5:  # 50% annualisée
            return SizingMethod.VOLATILITY_TARGET
        
        # Si signal très fort et bon historique, Kelly
        if signal.strength > 0.8 and self._has_good_track_record(signal.symbol):
            return SizingMethod.KELLY_CRITERION
        
        # Si faible liquidité, fixed fractional pour simplicité
        if market_context.liquidity_score < 0.3:
            return SizingMethod.FIXED_FRACTIONAL
        
        # Défaut configuré
        default = self.config.get('default_method', SizingMethod.VOLATILITY_TARGET.value)
        return SizingMethod(default)
    
    async def _fixed_fractional_sizing(
        self,
        signal: TradingSignal,
        market_context: MarketContext,
        portfolio_context: PortfolioContext
    ) -> PositionSizeResult:
        """Calcul Fixed Fractional : risque fixe par trade"""
        risk_fraction = Decimal(str(self.config['max_risk_per_trade']))
        
        # Capital à risquer
        risk_amount = portfolio_context.available_capital * risk_fraction
        
        # Si stop loss défini, calculer la quantité
        if signal.stop_loss:
            price_risk = abs(market_context.current_price - signal.stop_loss)
            if price_risk > 0:
                quantity = risk_amount / price_risk
            else:
                # Stop loss au même prix, utiliser volatilité
                atr_stop = market_context.current_price * Decimal(str(market_context.volatility * 0.02))
                quantity = risk_amount / atr_stop
        else:
            # Pas de stop, utiliser la volatilité pour estimer le risque
            estimated_risk = market_context.current_price * Decimal(str(market_context.volatility * 0.02))
            quantity = risk_amount / estimated_risk
        
        # Arrondir selon la taille de lot
        quantity = self._round_to_lot_size(quantity)
        
        position_value = quantity * market_context.current_price
        leverage = float(position_value / portfolio_context.available_capital)
        
        return PositionSizeResult(
            quantity=Quantity(quantity),
            risk_amount=risk_amount,
            position_value=position_value,
            leverage_used=leverage,
            method_used=SizingMethod.FIXED_FRACTIONAL,
            confidence_score=0.8,  # Méthode simple, haute confiance
            risk_adjusted=True,
            metadata={
                'risk_fraction': float(risk_fraction),
                'stop_loss_used': signal.stop_loss is not None
            }
        )
    
    async def _kelly_criterion_sizing(
        self,
        signal: TradingSignal,
        market_context: MarketContext,
        portfolio_context: PortfolioContext
    ) -> PositionSizeResult:
        """Calcul Kelly Criterion : taille optimale selon probabilités"""
        symbol = signal.symbol
        
        # Obtenir ou calculer les statistiques win/loss
        win_rate, avg_win, avg_loss = self._get_win_loss_stats(symbol)
        
        if win_rate == 0 or avg_loss == 0:
            # Pas assez de données, fallback
            self.logger.warning(
                "kelly_insufficient_data",
                symbol=symbol,
                win_rate=win_rate
            )
            return await self._fixed_fractional_sizing(
                signal, market_context, portfolio_context
            )
        
        # Kelly formula: f = (p * b - q) / b
        # où p = probabilité de gain, b = ratio gain/perte, q = 1-p
        p = win_rate
        q = 1 - p
        b = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        if b == 0:
            kelly_fraction = 0
        else:
            kelly_fraction = (p * b - q) / b
        
        # Appliquer le facteur de Kelly conservateur
        kelly_fraction *= self.config['kelly_fraction']
        
        # Limiter entre 0 et max risk
        kelly_fraction = max(0, min(kelly_fraction, self.config['max_risk_per_trade']))
        
        # Calculer la taille de position
        position_value = portfolio_context.available_capital * Decimal(str(kelly_fraction))
        quantity = position_value / market_context.current_price
        
        # Arrondir
        quantity = self._round_to_lot_size(quantity)
        
        # Calculer le risque estimé
        if signal.stop_loss:
            risk_per_share = abs(market_context.current_price - signal.stop_loss)
            risk_amount = quantity * risk_per_share
        else:
            risk_amount = position_value * Decimal(str(avg_loss))
        
        leverage = float(position_value / portfolio_context.available_capital)
        
        # Confiance basée sur le nombre d'échantillons
        sample_count = len(self.win_loss_ratios.get(symbol, []))
        confidence = min(0.9, 0.5 + sample_count / 200)  # Max 0.9
        
        return PositionSizeResult(
            quantity=Quantity(quantity),
            risk_amount=risk_amount,
            position_value=position_value,
            leverage_used=leverage,
            method_used=SizingMethod.KELLY_CRITERION,
            confidence_score=confidence,
            risk_adjusted=True,
            metadata={
                'kelly_fraction': float(kelly_fraction),
                'win_rate': win_rate,
                'win_loss_ratio': b,
                'sample_count': sample_count
            }
        )
    
    async def _volatility_target_sizing(
        self,
        signal: TradingSignal,
        market_context: MarketContext,
        portfolio_context: PortfolioContext
    ) -> PositionSizeResult:
        """Calcul Volatility Targeting : taille pour volatilité cible"""
        target_vol = self.config['target_volatility']
        current_vol = market_context.volatility
        
        if current_vol == 0:
            # Pas de volatilité, utiliser une estimation
            current_vol = 0.15  # 15% par défaut
        
        # Ratio pour atteindre la volatilité cible
        vol_scalar = target_vol / current_vol
        
        # Position de base (pourcentage du capital)
        base_position_pct = self.config['max_position_size_pct']
        
        # Ajuster selon la volatilité
        adjusted_position_pct = base_position_pct * vol_scalar
        
        # Limiter pour sécurité
        adjusted_position_pct = min(
            adjusted_position_pct,
            self.config['max_position_size_pct']
        )
        
        # Calculer la valeur et quantité
        position_value = portfolio_context.available_capital * Decimal(str(adjusted_position_pct))
        quantity = position_value / market_context.current_price
        
        # Arrondir
        quantity = self._round_to_lot_size(quantity)
        
        # Estimer le risque basé sur la volatilité
        daily_vol = current_vol / np.sqrt(252)
        risk_amount = position_value * Decimal(str(daily_vol * 2))  # 2 std dev
        
        leverage = float(position_value / portfolio_context.available_capital)
        
        return PositionSizeResult(
            quantity=Quantity(quantity),
            risk_amount=risk_amount,
            position_value=position_value,
            leverage_used=leverage,
            method_used=SizingMethod.VOLATILITY_TARGET,
            confidence_score=0.85,
            risk_adjusted=True,
            metadata={
                'target_volatility': target_vol,
                'market_volatility': current_vol,
                'volatility_scalar': vol_scalar,
                'position_pct': float(adjusted_position_pct)
            }
        )
    
    async def _risk_parity_sizing(
        self,
        signal: TradingSignal,
        market_context: MarketContext,
        portfolio_context: PortfolioContext
    ) -> PositionSizeResult:
        """Calcul Risk Parity : allocation équilibrée du risque"""
        # Obtenir les volatilités de tous les actifs
        all_volatilities = self._get_portfolio_volatilities(portfolio_context)
        
        if not all_volatilities:
            # Pas assez de données, fallback
            return await self._volatility_target_sizing(
                signal, market_context, portfolio_context
            )
        
        # Calculer les poids inversement proportionnels à la volatilité
        symbol_vol = market_context.volatility
        total_inv_vol = sum(1/v for v in all_volatilities.values() if v > 0)
        
        if total_inv_vol == 0 or symbol_vol == 0:
            weight = 1 / max(len(all_volatilities), 1)
        else:
            weight = (1/symbol_vol) / total_inv_vol
        
        # Appliquer le poids au capital disponible
        position_value = portfolio_context.available_capital * Decimal(str(weight))
        quantity = position_value / market_context.current_price
        
        # Arrondir
        quantity = self._round_to_lot_size(quantity)
        
        # Risque proportionnel
        risk_amount = position_value * Decimal(str(self.config['max_risk_per_trade']))
        
        leverage = float(position_value / portfolio_context.available_capital)
        
        return PositionSizeResult(
            quantity=Quantity(quantity),
            risk_amount=risk_amount,
            position_value=position_value,
            leverage_used=leverage,
            method_used=SizingMethod.RISK_PARITY,
            confidence_score=0.75,
            risk_adjusted=True,
            metadata={
                'risk_parity_weight': weight,
                'portfolio_assets': len(all_volatilities),
                'symbol_volatility': symbol_vol
            }
        )
    
    async def _equal_weight_sizing(
        self,
        signal: TradingSignal,
        market_context: MarketContext,
        portfolio_context: PortfolioContext
    ) -> PositionSizeResult:
        """Calcul Equal Weight : allocation égale simple"""
        # Nombre d'actifs dans le portefeuille
        num_positions = len(portfolio_context.current_positions) + 1
        
        # Poids égal
        weight = 1 / num_positions
        
        # Ne pas dépasser le max par position
        weight = min(weight, self.config['max_position_size_pct'])
        
        position_value = portfolio_context.available_capital * Decimal(str(weight))
        quantity = position_value / market_context.current_price
        
        # Arrondir
        quantity = self._round_to_lot_size(quantity)
        
        # Risque estimé
        risk_amount = position_value * Decimal(str(self.config['max_risk_per_trade']))
        
        leverage = float(position_value / portfolio_context.available_capital)
        
        return PositionSizeResult(
            quantity=Quantity(quantity),
            risk_amount=risk_amount,
            position_value=position_value,
            leverage_used=leverage,
            method_used=SizingMethod.EQUAL_WEIGHT,
            confidence_score=0.7,
            risk_adjusted=False,
            metadata={
                'equal_weight': weight,
                'num_positions': num_positions
            }
        )
    
    async def _anti_martingale_sizing(
        self,
        signal: TradingSignal,
        market_context: MarketContext,
        portfolio_context: PortfolioContext
    ) -> PositionSizeResult:
        """Anti-Martingale : augmente après gains, réduit après pertes"""
        # Obtenir la performance récente
        recent_trades = self._get_recent_trades(signal.symbol)
        
        if not recent_trades:
            # Pas d'historique, utiliser la méthode de base
            return await self._fixed_fractional_sizing(
                signal, market_context, portfolio_context
            )
        
        # Calculer les gains/pertes consécutifs
        consecutive_wins = 0
        consecutive_losses = 0
        
        for trade in reversed(recent_trades[-10:]):  # 10 derniers trades
            if trade['pnl'] > 0:
                if consecutive_losses > 0:
                    break
                consecutive_wins += 1
            else:
                if consecutive_wins > 0:
                    break
                consecutive_losses += 1
        
        # Facteur d'ajustement
        if consecutive_wins > 0:
            # Augmenter progressivement
            scale_factor = 1 + (consecutive_wins * 0.1)  # +10% par gain
            scale_factor = min(scale_factor, 2.0)  # Max 2x
        elif consecutive_losses > 0:
            # Réduire progressivement
            scale_factor = 1 - (consecutive_losses * 0.2)  # -20% par perte
            scale_factor = max(scale_factor, 0.3)  # Min 0.3x
        else:
            scale_factor = 1.0
        
        # Obtenir la taille de base
        base_result = await self._fixed_fractional_sizing(
            signal, market_context, portfolio_context
        )
        
        # Appliquer le facteur
        adjusted_quantity = base_result.quantity * Decimal(str(scale_factor))
        adjusted_quantity = self._round_to_lot_size(adjusted_quantity)
        
        position_value = adjusted_quantity * market_context.current_price
        risk_amount = base_result.risk_amount * Decimal(str(scale_factor))
        
        return PositionSizeResult(
            quantity=Quantity(adjusted_quantity),
            risk_amount=risk_amount,
            position_value=position_value,
            leverage_used=float(position_value / portfolio_context.available_capital),
            method_used=SizingMethod.ANTI_MARTINGALE,
            confidence_score=0.7,
            risk_adjusted=True,
            adjustments_applied=['anti_martingale'],
            metadata={
                'scale_factor': scale_factor,
                'consecutive_wins': consecutive_wins,
                'consecutive_losses': consecutive_losses
            }
        )
    
    async def _apply_adjustments(
        self,
        result: PositionSizeResult,
        signal: TradingSignal,
        market_context: MarketContext,
        portfolio_context: PortfolioContext
    ) -> PositionSizeResult:
        """Applique tous les ajustements configurés"""
        adjustments = []
        original_quantity = result.quantity
        
        # 1. Ajustement de volatilité
        if self.config.get('enable_volatility_adjustment', True):
            vol_factor = self._get_volatility_adjustment(market_context)
            if vol_factor != 1.0:
                result.quantity = Quantity(result.quantity * Decimal(str(vol_factor)))
                adjustments.append(f"volatility_adj_{vol_factor:.2f}")
        
        # 2. Ajustement de drawdown
        if self.config.get('enable_drawdown_adjustment', True):
            dd_factor = self._get_drawdown_adjustment(portfolio_context)
            if dd_factor != 1.0:
                result.quantity = Quantity(result.quantity * Decimal(str(dd_factor)))
                adjustments.append(f"drawdown_adj_{dd_factor:.2f}")
        
        # 3. Ajustement de corrélation
        if self.config.get('enable_correlation_adjustment', True):
            corr_factor = self._get_correlation_adjustment(
                signal.symbol, market_context, portfolio_context
            )
            if corr_factor != 1.0:
                result.quantity = Quantity(result.quantity * Decimal(str(corr_factor)))
                adjustments.append(f"correlation_adj_{corr_factor:.2f}")
        
        # 4. Ajustement de liquidité
        if self.config.get('enable_liquidity_adjustment', True):
            liq_factor = self._get_liquidity_adjustment(market_context)
            if liq_factor != 1.0:
                result.quantity = Quantity(result.quantity * Decimal(str(liq_factor)))
                adjustments.append(f"liquidity_adj_{liq_factor:.2f}")
        
        # 5. Ajustement de confiance du signal
        conf_factor = float(signal.confidence)
        if conf_factor < 1.0:
            result.quantity = Quantity(result.quantity * Decimal(str(conf_factor)))
            adjustments.append(f"confidence_adj_{conf_factor:.2f}")
        
        # Recalculer les valeurs si ajusté
        if result.quantity != original_quantity:
            result.position_value = result.quantity * market_context.current_price
            result.leverage_used = float(
                result.position_value / portfolio_context.available_capital
            )
            adjustment_ratio = float(result.quantity / original_quantity)
            result.risk_amount = result.risk_amount * Decimal(str(adjustment_ratio))
        
        result.adjustments_applied.extend(adjustments)
        result.risk_adjusted = True
        
        return result
    
    def _get_volatility_adjustment(self, market_context: MarketContext) -> float:
        """Calcule l'ajustement basé sur la volatilité"""
        current_vol = market_context.volatility
        
        # Volatilité normale : 15-25% annualisée
        if 0.15 <= current_vol <= 0.25:
            return 1.0
        
        # Haute volatilité : réduire
        if current_vol > 0.25:
            if current_vol > 0.50:  # Très haute
                return 0.5
            else:
                # Réduction linéaire
                return 1.0 - (current_vol - 0.25) * 2
        
        # Basse volatilité : augmenter légèrement
        if current_vol < 0.15:
            return min(1.2, 1.0 + (0.15 - current_vol) * 2)
        
        return 1.0
    
    def _get_drawdown_adjustment(self, portfolio_context: PortfolioContext) -> float:
        """Calcule l'ajustement basé sur le drawdown actuel"""
        if not hasattr(portfolio_context.recent_performance, 'current_drawdown'):
            return 1.0
        
        current_dd = abs(portfolio_context.recent_performance.current_drawdown)
        
        # Drawdown faible : normal
        if current_dd < 0.05:  # < 5%
            return 1.0
        
        # Drawdown modéré : réduire progressivement
        if current_dd < 0.10:  # 5-10%
            return 1.0 - (current_dd - 0.05) * 2  # 1.0 à 0.9
        
        # Drawdown élevé : réduction forte
        if current_dd < 0.20:  # 10-20%
            return 0.9 - (current_dd - 0.10) * 4  # 0.9 à 0.5
        
        # Drawdown critique : réduction d'urgence
        return self.config.get('emergency_reduction_factor', 0.5)
    
    def _get_correlation_adjustment(
        self,
        symbol: Symbol,
        market_context: MarketContext,
        portfolio_context: PortfolioContext
    ) -> float:
        """Calcule l'ajustement basé sur la corrélation du portefeuille"""
        # Si pas de matrice de corrélation, pas d'ajustement
        if market_context.correlation_matrix is None:
            return 1.0
        
        # Calculer l'exposition aux actifs corrélés
        high_corr_exposure = portfolio_context.correlation_exposure
        max_allowed = self.config.get('max_correlation_exposure', 0.60)
        
        if high_corr_exposure >= max_allowed:
            # Déjà trop exposé aux actifs corrélés
            return 0.5
        
        # Réduction progressive
        if high_corr_exposure > max_allowed * 0.8:
            return 1.0 - (high_corr_exposure - max_allowed * 0.8) * 2.5
        
        return 1.0
    
    def _get_liquidity_adjustment(self, market_context: MarketContext) -> float:
        """Calcule l'ajustement basé sur la liquidité"""
        liquidity = market_context.liquidity_score
        
        # Bonne liquidité : pas d'ajustement
        if liquidity > 0.7:
            return 1.0
        
        # Liquidité moyenne : légère réduction
        if liquidity > 0.4:
            return 0.9
        
        # Faible liquidité : réduction importante
        if liquidity > 0.2:
            return 0.7
        
        # Très faible liquidité : forte réduction
        return 0.5
    
    def _apply_safety_limits(
        self,
        result: PositionSizeResult,
        market_context: MarketContext,
        portfolio_context: PortfolioContext
    ) -> PositionSizeResult:
        """Applique les limites de sécurité finales"""
        warnings = []
        
        # 1. Limite de taille max par position
        max_position_value = portfolio_context.total_capital * Decimal(
            str(self.config['max_position_size_pct'])
        )
        if result.position_value > max_position_value:
            result.quantity = Quantity(max_position_value / market_context.current_price)
            result.position_value = max_position_value
            warnings.append("position_size_limit_applied")
        
        # 2. Limite de risque max
        max_risk = portfolio_context.total_capital * Decimal(
            str(self.config['max_risk_per_trade'])
        )
        if result.risk_amount > max_risk:
            ratio = max_risk / result.risk_amount
            result.quantity = Quantity(result.quantity * ratio)
            result.position_value = result.quantity * market_context.current_price
            result.risk_amount = max_risk
            warnings.append("risk_limit_applied")
        
        # 3. Limite de levier
        max_leverage = self.config['max_leverage']
        if result.leverage_used > max_leverage:
            result.leverage_used = max_leverage
            max_value = portfolio_context.available_capital * Decimal(str(max_leverage))
            result.quantity = Quantity(max_value / market_context.current_price)
            result.position_value = max_value
            warnings.append("leverage_limit_applied")
        
        # 4. Taille minimum
        min_size = Decimal(str(self.config['min_position_size']))
        if result.position_value < min_size:
            result.quantity = Quantity(Decimal("0"))
            result.position_value = Decimal("0")
            result.risk_amount = Decimal("0")
            warnings.append("below_minimum_size")
        
        # 5. Capital disponible
        if result.position_value > portfolio_context.available_capital:
            result.quantity = Quantity(
                portfolio_context.available_capital / market_context.current_price
            )
            result.position_value = portfolio_context.available_capital
            warnings.append("insufficient_capital")
        
        # Arrondir final
        result.quantity = Quantity(self._round_to_lot_size(result.quantity))
        
        # Ajouter les warnings
        result.warnings.extend(warnings)
        
        # Ajuster la confiance si des limites ont été appliquées
        if warnings:
            result.confidence_score *= 0.8
        
        return result
    
    def _round_to_lot_size(self, quantity: Decimal) -> Decimal:
        """Arrondit à la taille de lot appropriée"""
        lot_size = Decimal(str(self.config.get('round_lot_size', 1)))
        
        if lot_size == 1:
            # Arrondir à l'entier
            return quantity.quantize(Decimal('1'), rounding=ROUND_DOWN)
        else:
            # Arrondir au lot le plus proche
            lots = (quantity / lot_size).quantize(Decimal('1'), rounding=ROUND_DOWN)
            return lots * lot_size
    
    def _get_win_loss_stats(self, symbol: Symbol) -> Tuple[float, float, float]:
        """Obtient les statistiques win/loss pour un symbole"""
        ratios = self.win_loss_ratios.get(symbol, deque())
        
        if len(ratios) < 10:  # Pas assez de données
            return 0.5, 0.01, 0.01  # Valeurs par défaut
        
        wins = [r for r in ratios if r > 0]
        losses = [r for r in ratios if r < 0]
        
        win_rate = len(wins) / len(ratios) if ratios else 0.5
        avg_win = sum(wins) / len(wins) if wins else 0.01
        avg_loss = sum(losses) / len(losses) if losses else -0.01
        
        return win_rate, avg_win, abs(avg_loss)
    
    def _has_good_track_record(self, symbol: Symbol) -> bool:
        """Vérifie si le symbole a un bon historique"""
        ratios = self.win_loss_ratios.get(symbol, deque())
        
        if len(ratios) < 20:
            return False
        
        win_rate = sum(1 for r in ratios if r > 0) / len(ratios)
        avg_return = sum(ratios) / len(ratios)
        
        return win_rate > 0.55 and avg_return > 0.001
    
    def _get_portfolio_volatilities(
        self,
        portfolio_context: PortfolioContext
    ) -> Dict[Symbol, float]:
        """Obtient les volatilités de tous les actifs du portefeuille"""
        volatilities = {}
        
        for symbol in portfolio_context.current_positions:
            vol_history = self.volatility_history.get(symbol, deque())
            if vol_history:
                # Moyenne des volatilités récentes
                recent_vols = list(vol_history)[-20:]
                volatilities[symbol] = sum(recent_vols) / len(recent_vols)
            else:
                volatilities[symbol] = 0.20  # 20% par défaut
        
        return volatilities
    
    def _get_recent_trades(self, symbol: Symbol) -> List[Dict[str, Any]]:
        """Obtient les trades récents pour un symbole"""
        history = self.position_history.get(symbol, deque())
        return list(history)[-50:]  # 50 derniers trades
    
    def _get_safe_fallback_size(
        self,
        signal: TradingSignal,
        market_context: MarketContext,
        portfolio_context: PortfolioContext
    ) -> PositionSizeResult:
        """Retourne une taille sûre en cas d'erreur"""
        # 0.5% du capital ou min position size
        safe_value = min(
            portfolio_context.available_capital * Decimal("0.005"),
            Decimal(str(self.config['min_position_size'] * 2))
        )
        
        quantity = safe_value / market_context.current_price
        quantity = self._round_to_lot_size(quantity)
        
        return PositionSizeResult(
            quantity=Quantity(quantity),
            risk_amount=safe_value * Decimal("0.02"),  # 2% du montant
            position_value=safe_value,
            leverage_used=float(safe_value / portfolio_context.available_capital),
            method_used=SizingMethod.FIXED_FRACTIONAL,
            confidence_score=0.5,
            risk_adjusted=False,
            warnings=["fallback_size_used"]
        )
    
    def _record_position_size(self, symbol: Symbol, result: PositionSizeResult) -> None:
        """Enregistre la taille de position dans l'historique"""
        record = {
            'timestamp': datetime.now(timezone.utc),
            'quantity': float(result.quantity),
            'risk_amount': float(result.risk_amount),
            'method': result.method_used.value,
            'confidence': result.confidence_score,
            'adjustments': result.adjustments_applied
        }
        
        self.position_history[symbol].append(record)
        self.sizing_stats[result.method_used.value] += 1
    
    def update_trade_result(
        self,
        symbol: Symbol,
        pnl: Decimal,
        entry_price: Price,
        exit_price: Price
    ) -> None:
        """Met à jour les résultats de trade pour l'apprentissage"""
        # Calculer le ratio de gain/perte
        if entry_price > 0:
            return_pct = float((exit_price - entry_price) / entry_price)
        else:
            return_pct = 0.0
        
        self.win_loss_ratios[symbol].append(return_pct)
        
        # Logger pour analyse
        self.logger.info(
            "trade_result_updated",
            symbol=symbol,
            pnl=float(pnl),
            return_pct=return_pct
        )
    
    def update_market_volatility(self, symbol: Symbol, volatility: float) -> None:
        """Met à jour la volatilité historique d'un symbole"""
        self.volatility_history[symbol].append(volatility)
    
    def get_sizing_statistics(self) -> Dict[str, Any]:
        """Retourne les statistiques d'utilisation des méthodes"""
        total_sizings = sum(self.sizing_stats.values())
        
        return {
            'total_sizings': total_sizings,
            'method_usage': {
                method: {
                    'count': count,
                    'percentage': count / total_sizings * 100 if total_sizings > 0 else 0
                }
                for method, count in self.sizing_stats.items()
            },
            'average_confidence': self._calculate_average_confidence(),
            'adjustment_frequency': self._calculate_adjustment_frequency()
        }
    
    def _calculate_average_confidence(self) -> float:
        """Calcule la confiance moyenne des calculs récents"""
        all_records = []
        for history in self.position_history.values():
            all_records.extend(list(history)[-100:])
        
        if not all_records:
            return 0.0
        
        confidences = [r['confidence'] for r in all_records]
        return sum(confidences) / len(confidences)
    
    def _calculate_adjustment_frequency(self) -> Dict[str, float]:
        """Calcule la fréquence des ajustements appliqués"""
        adjustment_counts = defaultdict(int)
        total_records = 0
        
        for history in self.position_history.values():
            for record in list(history)[-500:]:
                total_records += 1
                for adj in record.get('adjustments', []):
                    adj_type = adj.split('_')[0]
                    adjustment_counts[adj_type] += 1
        
        if total_records == 0:
            return {}
        
        return {
            adj: count / total_records * 100
            for adj, count in adjustment_counts.items()
        }
    
    def clear_cache(self) -> None:
        """Nettoie les caches pour libérer la mémoire"""
        self._kelly_cache.clear()
        self._correlation_cache = None
        
        # Conserver seulement les données récentes
        for symbol in list(self.position_history.keys()):
            if len(self.position_history[symbol]) == 0:
                del self.position_history[symbol]
        
        self.logger.debug("position_sizer_cache_cleared")


# Classe helper pour batch processing
class BatchPositionSizer:
    """Helper pour calculer plusieurs positions en parallèle"""
    
    def __init__(self, position_sizer: PositionSizer):
        self.sizer = position_sizer
    
    async def calculate_batch(
        self,
        signals: List[TradingSignal],
        market_contexts: Dict[Symbol, MarketContext],
        portfolio_context: PortfolioContext,
        max_concurrent: int = 10
    ) -> Dict[Symbol, PositionSizeResult]:
        """
        Calcule les tailles pour plusieurs signaux en parallèle
        
        Args:
            signals: Liste des signaux à traiter
            market_contexts: Contextes de marché par symbole
            portfolio_context: Contexte du portefeuille
            max_concurrent: Nombre max de calculs simultanés
            
        Returns:
            Dict des résultats par symbole
        """
        results = {}
        
        # Créer les tâches
        tasks = []
        for signal in signals:
            if signal.symbol in market_contexts:
                task = self.sizer.calculate_position_size(
                    signal,
                    market_contexts[signal.symbol],
                    portfolio_context
                )
                tasks.append((signal.symbol, task))
        
        # Exécuter par batches
        for i in range(0, len(tasks), max_concurrent):
            batch = tasks[i:i + max_concurrent]
            batch_results = await asyncio.gather(
                *[task for _, task in batch],
                return_exceptions=True
            )
            
            for (symbol, _), result in zip(batch, batch_results):
                if isinstance(result, Exception):
                    # En cas d'erreur, utiliser fallback
                    result = self.sizer._get_safe_fallback_size(
                        next(s for s in signals if s.symbol == symbol),
                        market_contexts[symbol],
                        portfolio_context
                    )
                results[symbol] = result
        
        return results