"""
Portfolio Manager - Composant central de gestion du portefeuille
Gère l'allocation du capital, le sizing des positions, le rééquilibrage et le tracking P&L

Architecture moderne avec async/await, typing strict et patterns event-driven
Compatible avec les stratégies HFT et l'apprentissage par renforcement
"""

import asyncio
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Protocol, NewType
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import defaultdict
from enum import Enum
import json
import logging
from abc import ABC, abstractmethod

# Types sémantiques pour la sécurité
Symbol = NewType('Symbol', str)
StrategyId = NewType('StrategyId', str)
OrderId = NewType('OrderId', str)
Price = NewType('Price', Decimal)
Quantity = NewType('Quantity', Decimal)
Capital = NewType('Capital', Decimal)

# Configuration des constantes
MAX_POSITION_SIZE_PCT = 0.10  # 10% max par position
MAX_STRATEGY_ALLOCATION_PCT = 0.25  # 25% max par stratégie
DEFAULT_RISK_PER_TRADE_PCT = 0.02  # 2% risque par trade
TARGET_VOLATILITY_ANNUAL = 0.12  # 12% volatilité cible annuelle
MIN_REBALANCE_THRESHOLD = 0.05  # 5% dérive avant rééquilibrage
CORRELATION_WARNING_THRESHOLD = 0.80  # Alerte si corrélation > 80%


class AllocationMethod(Enum):
    """Méthodes d'allocation du capital"""
    EQUAL_WEIGHT = "equal_weight"
    RISK_PARITY = "risk_parity"
    KELLY_CRITERION = "kelly_criterion"
    VOLATILITY_TARGET = "volatility_target"
    DYNAMIC_ML = "dynamic_ml"


class OrderType(Enum):
    """Types d'ordres supportés"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class Position:
    """Représentation d'une position dans le portefeuille"""
    symbol: Symbol
    strategy_id: StrategyId
    quantity: Quantity
    avg_entry_price: Price
    current_price: Price
    opened_at: datetime
    last_updated: datetime
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    fees_paid: Decimal = Decimal("0")
    
    @property
    def market_value(self) -> Decimal:
        """Valeur marchande de la position"""
        return self.quantity * self.current_price
    
    @property
    def total_pnl(self) -> Decimal:
        """P&L total (réalisé + non réalisé)"""
        return self.realized_pnl + self.unrealized_pnl
    
    def update_price(self, new_price: Price) -> None:
        """Met à jour le prix et recalcule le P&L non réalisé"""
        self.current_price = new_price
        self.unrealized_pnl = (new_price - self.avg_entry_price) * self.quantity
        self.last_updated = datetime.now(timezone.utc)


@dataclass
class PortfolioSnapshot:
    """État instantané du portefeuille"""
    timestamp: datetime
    total_equity: Decimal
    cash_balance: Decimal
    margin_used: Decimal
    positions_value: Decimal
    total_pnl: Decimal
    daily_pnl: Decimal
    positions_count: int
    sharpe_ratio: float
    max_drawdown: float
    var_95: Decimal
    var_99: Decimal


@dataclass
class AllocationTarget:
    """Cible d'allocation pour une stratégie/symbole"""
    strategy_id: StrategyId
    symbol: Symbol
    target_weight: float
    target_value: Decimal
    current_value: Decimal
    deviation: float
    action_required: str  # "BUY", "SELL", "HOLD"
    quantity_change: Quantity


class RiskCalculator(Protocol):
    """Interface pour les calculs de risque"""
    async def calculate_var(self, positions: List[Position], confidence: float) -> Decimal:
        ...
    
    async def calculate_expected_shortfall(self, positions: List[Position], confidence: float) -> Decimal:
        ...
    
    async def calculate_correlations(self, positions: List[Position]) -> np.ndarray:
        ...


class PositionSizer(Protocol):
    """Interface pour le calcul de taille de position"""
    async def calculate_position_size(
        self,
        signal_strength: float,
        volatility: float,
        capital: Decimal,
        existing_positions: List[Position]
    ) -> Quantity:
        ...


class PortfolioManager:
    """
    Gestionnaire principal du portefeuille
    Coordonne l'allocation, le sizing, le risque et le P&L
    """
    
    def __init__(
        self,
        initial_capital: Decimal,
        risk_calculator: RiskCalculator,
        position_sizer: PositionSizer,
        allocation_method: AllocationMethod = AllocationMethod.RISK_PARITY,
        config: Optional[Dict[str, Any]] = None
    ):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_calculator = risk_calculator
        self.position_sizer = position_sizer
        self.allocation_method = allocation_method
        self.config = config or {}
        
        # État du portefeuille
        self.positions: Dict[Tuple[Symbol, StrategyId], Position] = {}
        self.cash_balance = initial_capital
        self.margin_used = Decimal("0")
        
        # Historique et métriques
        self.portfolio_history: List[PortfolioSnapshot] = []
        self.daily_returns: List[float] = []
        self.high_water_mark = initial_capital
        
        # Allocation par stratégie
        self.strategy_allocations: Dict[StrategyId, Decimal] = defaultdict(Decimal)
        self.strategy_performance: Dict[StrategyId, Dict[str, float]] = defaultdict(dict)
        
        # Event tracking
        self.pending_orders: Dict[OrderId, Dict[str, Any]] = {}
        self.last_rebalance = datetime.now(timezone.utc)
        
        # Configuration des limites
        self.max_position_size_pct = self.config.get('max_position_size_pct', MAX_POSITION_SIZE_PCT)
        self.risk_per_trade_pct = self.config.get('risk_per_trade_pct', DEFAULT_RISK_PER_TRADE_PCT)
        self.target_volatility = self.config.get('target_volatility', TARGET_VOLATILITY_ANNUAL)
        
        self.logger = logging.getLogger(__name__)
        self._lock = asyncio.Lock()  # Thread-safety pour les opérations critiques
    
    async def validate_trade(
        self,
        symbol: Symbol,
        strategy_id: StrategyId,
        quantity: Quantity,
        price: Price,
        order_type: OrderType
    ) -> Tuple[bool, Optional[str]]:
        """
        Validation pré-trade avec contrôles de risque
        Retourne (is_valid, error_message)
        """
        async with self._lock:
            # Vérifier le capital disponible
            required_capital = quantity * price
            if required_capital > self.cash_balance:
                return False, f"Capital insuffisant: requis={required_capital}, disponible={self.cash_balance}"
            
            # Vérifier la limite de position
            position_value = required_capital
            position_pct = float(position_value / self.current_capital)
            if position_pct > self.max_position_size_pct:
                return False, f"Position trop grande: {position_pct:.1%} > {self.max_position_size_pct:.1%}"
            
            # Vérifier l'allocation de la stratégie
            strategy_total = self.strategy_allocations.get(strategy_id, Decimal("0"))
            new_strategy_total = strategy_total + position_value
            strategy_pct = float(new_strategy_total / self.current_capital)
            if strategy_pct > MAX_STRATEGY_ALLOCATION_PCT:
                return False, f"Allocation stratégie dépassée: {strategy_pct:.1%} > {MAX_STRATEGY_ALLOCATION_PCT:.1%}"
            
            # Vérifier les corrélations
            if len(self.positions) > 0:
                correlations = await self.risk_calculator.calculate_correlations(list(self.positions.values()))
                max_correlation = np.max(correlations[correlations < 1.0]) if correlations.size > 0 else 0
                if max_correlation > CORRELATION_WARNING_THRESHOLD:
                    self.logger.warning(f"Corrélation élevée détectée: {max_correlation:.2f}")
            
            # Vérifier la VaR
            temp_positions = list(self.positions.values())
            var_99 = await self.risk_calculator.calculate_var(temp_positions, 0.99)
            var_pct = float(var_99 / self.current_capital)
            if var_pct > 0.05:  # VaR 99% ne doit pas dépasser 5%
                return False, f"VaR trop élevée: {var_pct:.1%} > 5%"
            
            return True, None
    
    async def open_position(
        self,
        symbol: Symbol,
        strategy_id: StrategyId,
        quantity: Quantity,
        price: Price,
        order_id: OrderId,
        fees: Decimal = Decimal("0")
    ) -> Position:
        """Ouvre une nouvelle position ou ajoute à une existante"""
        async with self._lock:
            key = (symbol, strategy_id)
            
            if key in self.positions:
                # Ajouter à la position existante
                position = self.positions[key]
                total_value = (position.quantity * position.avg_entry_price) + (quantity * price)
                new_quantity = position.quantity + quantity
                position.avg_entry_price = Price(total_value / new_quantity)
                position.quantity = new_quantity
                position.fees_paid += fees
            else:
                # Créer nouvelle position
                position = Position(
                    symbol=symbol,
                    strategy_id=strategy_id,
                    quantity=quantity,
                    avg_entry_price=price,
                    current_price=price,
                    opened_at=datetime.now(timezone.utc),
                    last_updated=datetime.now(timezone.utc),
                    fees_paid=fees
                )
                self.positions[key] = position
            
            # Mettre à jour le capital et les allocations
            cost = quantity * price + fees
            self.cash_balance -= cost
            self.strategy_allocations[strategy_id] += cost
            
            await self._update_portfolio_metrics()
            return position
    
    async def close_position(
        self,
        symbol: Symbol,
        strategy_id: StrategyId,
        quantity: Quantity,
        price: Price,
        order_id: OrderId,
        fees: Decimal = Decimal("0")
    ) -> Optional[Decimal]:
        """
        Ferme une position (totalement ou partiellement)
        Retourne le P&L réalisé
        """
        async with self._lock:
            key = (symbol, strategy_id)
            if key not in self.positions:
                self.logger.error(f"Position non trouvée: {symbol}/{strategy_id}")
                return None
            
            position = self.positions[key]
            
            # Calculer le P&L réalisé
            realized_pnl = (price - position.avg_entry_price) * quantity - fees
            position.realized_pnl += realized_pnl
            position.fees_paid += fees
            
            # Mettre à jour la quantité
            if quantity >= position.quantity:
                # Position complètement fermée
                proceeds = position.quantity * price - fees
                self.cash_balance += proceeds
                self.strategy_allocations[strategy_id] -= position.market_value
                del self.positions[key]
            else:
                # Position partiellement fermée
                position.quantity -= quantity
                proceeds = quantity * price - fees
                self.cash_balance += proceeds
                self.strategy_allocations[strategy_id] -= (quantity * position.avg_entry_price)
            
            await self._update_portfolio_metrics()
            return realized_pnl
    
    async def update_prices(self, price_updates: Dict[Symbol, Price]) -> None:
        """Met à jour les prix et recalcule les métriques"""
        async with self._lock:
            for (symbol, strategy_id), position in self.positions.items():
                if symbol in price_updates:
                    position.update_price(price_updates[symbol])
            
            await self._update_portfolio_metrics()
    
    async def calculate_optimal_allocation(
        self,
        strategy_signals: Dict[StrategyId, Dict[Symbol, float]]
    ) -> List[AllocationTarget]:
        """
        Calcule l'allocation optimale selon la méthode configurée
        strategy_signals: {strategy_id: {symbol: signal_strength}}
        """
        allocations = []
        
        if self.allocation_method == AllocationMethod.RISK_PARITY:
            allocations = await self._calculate_risk_parity_allocation(strategy_signals)
        elif self.allocation_method == AllocationMethod.KELLY_CRITERION:
            allocations = await self._calculate_kelly_allocation(strategy_signals)
        elif self.allocation_method == AllocationMethod.VOLATILITY_TARGET:
            allocations = await self._calculate_volatility_target_allocation(strategy_signals)
        elif self.allocation_method == AllocationMethod.EQUAL_WEIGHT:
            allocations = await self._calculate_equal_weight_allocation(strategy_signals)
        else:
            # Dynamic ML allocation
            allocations = await self._calculate_ml_allocation(strategy_signals)
        
        return allocations
    
    async def _calculate_risk_parity_allocation(
        self,
        strategy_signals: Dict[StrategyId, Dict[Symbol, float]]
    ) -> List[AllocationTarget]:
        """Allocation basée sur la parité des risques"""
        targets = []
        
        # Calculer les volatilités historiques
        volatilities = {}
        for strategy_id, signals in strategy_signals.items():
            for symbol, signal in signals.items():
                # Simuler la volatilité (en production, utiliser les données historiques)
                vol = np.random.uniform(0.1, 0.3)  # Placeholder
                volatilities[(strategy_id, symbol)] = vol
        
        # Calculer les poids inversement proportionnels à la volatilité
        total_inv_vol = sum(1/v for v in volatilities.values())
        
        for (strategy_id, symbol), vol in volatilities.items():
            weight = (1/vol) / total_inv_vol
            target_value = Decimal(str(weight)) * self.current_capital
            
            # Obtenir la valeur actuelle
            key = (symbol, strategy_id)
            current_value = self.positions[key].market_value if key in self.positions else Decimal("0")
            
            deviation = float((target_value - current_value) / target_value) if target_value > 0 else 0
            
            if abs(deviation) > MIN_REBALANCE_THRESHOLD:
                action = "BUY" if deviation > 0 else "SELL"
                quantity_change = Quantity((target_value - current_value) / Decimal("100"))  # Prix placeholder
                
                targets.append(AllocationTarget(
                    strategy_id=strategy_id,
                    symbol=symbol,
                    target_weight=weight,
                    target_value=target_value,
                    current_value=current_value,
                    deviation=deviation,
                    action_required=action,
                    quantity_change=quantity_change
                ))
        
        return targets
    
    async def _calculate_kelly_allocation(
        self,
        strategy_signals: Dict[StrategyId, Dict[Symbol, float]]
    ) -> List[AllocationTarget]:
        """Allocation basée sur le critère de Kelly"""
        targets = []
        
        for strategy_id, signals in strategy_signals.items():
            # Obtenir les statistiques de performance de la stratégie
            perf = self.strategy_performance.get(strategy_id, {})
            win_rate = perf.get('win_rate', 0.5)
            avg_win = perf.get('avg_win', 0.02)
            avg_loss = perf.get('avg_loss', 0.01)
            
            # Calculer la fraction de Kelly
            if avg_loss > 0:
                b = avg_win / avg_loss
                p = win_rate
                q = 1 - p
                kelly_fraction = (p * b - q) / b if b > 0 else 0
                
                # Appliquer un facteur de sécurité (Kelly fractionnaire)
                kelly_fraction *= 0.25  # 25% du Kelly complet
                
                for symbol, signal in signals.items():
                    if signal > 0:
                        weight = kelly_fraction * signal
                        target_value = Decimal(str(weight)) * self.current_capital
                        
                        # Limiter par position max
                        max_value = self.current_capital * Decimal(str(self.max_position_size_pct))
                        target_value = min(target_value, max_value)
                        
                        key = (symbol, strategy_id)
                        current_value = self.positions[key].market_value if key in self.positions else Decimal("0")
                        
                        targets.append(AllocationTarget(
                            strategy_id=strategy_id,
                            symbol=symbol,
                            target_weight=float(weight),
                            target_value=target_value,
                            current_value=current_value,
                            deviation=float((target_value - current_value) / target_value) if target_value > 0 else 0,
                            action_required="BUY" if target_value > current_value else "SELL",
                            quantity_change=Quantity((target_value - current_value) / Decimal("100"))
                        ))
        
        return targets
    
    async def _update_portfolio_metrics(self) -> None:
        """Met à jour toutes les métriques du portefeuille"""
        # Calculer la valeur totale des positions
        positions_value = sum(pos.market_value for pos in self.positions.values())
        
        # Capital total
        self.current_capital = self.cash_balance + positions_value
        
        # P&L total
        total_pnl = sum(pos.total_pnl for pos in self.positions.values())
        
        # P&L journalier
        if self.portfolio_history:
            daily_pnl = total_pnl - self.portfolio_history[-1].total_pnl
        else:
            daily_pnl = total_pnl
        
        # Calculer les métriques de risque
        var_95 = await self.risk_calculator.calculate_var(list(self.positions.values()), 0.95)
        var_99 = await self.risk_calculator.calculate_var(list(self.positions.values()), 0.99)
        
        # Sharpe ratio (simplifié)
        if len(self.daily_returns) > 30:
            returns_array = np.array(self.daily_returns[-252:])  # 1 an de données
            sharpe = np.sqrt(252) * np.mean(returns_array) / (np.std(returns_array) + 1e-8)
        else:
            sharpe = 0.0
        
        # Drawdown
        drawdown = float((self.high_water_mark - self.current_capital) / self.high_water_mark)
        if self.current_capital > self.high_water_mark:
            self.high_water_mark = self.current_capital
        
        # Créer le snapshot
        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_equity=self.current_capital,
            cash_balance=self.cash_balance,
            margin_used=self.margin_used,
            positions_value=positions_value,
            total_pnl=total_pnl,
            daily_pnl=daily_pnl,
            positions_count=len(self.positions),
            sharpe_ratio=sharpe,
            max_drawdown=drawdown,
            var_95=var_95,
            var_99=var_99
        )
        
        self.portfolio_history.append(snapshot)
        
        # Mettre à jour les rendements journaliers
        if self.portfolio_history and len(self.portfolio_history) > 1:
            prev_equity = self.portfolio_history[-2].total_equity
            daily_return = float((self.current_capital - prev_equity) / prev_equity)
            self.daily_returns.append(daily_return)
    
    async def rebalance_portfolio(self, force: bool = False) -> List[AllocationTarget]:
        """
        Rééquilibre le portefeuille selon les cibles d'allocation
        force: forcer le rééquilibrage même si les seuils ne sont pas atteints
        """
        # Vérifier si le rééquilibrage est nécessaire
        time_since_last = (datetime.now(timezone.utc) - self.last_rebalance).total_seconds() / 3600
        if not force and time_since_last < 24:  # Pas plus d'une fois par jour
            return []
        
        # Collecter les signaux actuels (placeholder - en production, obtenir des stratégies)
        strategy_signals = {}
        for strategy_id in self.strategy_allocations.keys():
            strategy_signals[strategy_id] = {}
        
        # Calculer les allocations optimales
        targets = await self.calculate_optimal_allocation(strategy_signals)
        
        # Filtrer les cibles nécessitant une action
        actions_required = [t for t in targets if abs(t.deviation) > MIN_REBALANCE_THRESHOLD]
        
        if actions_required or force:
            self.last_rebalance = datetime.now(timezone.utc)
            self.logger.info(f"Rééquilibrage: {len(actions_required)} ajustements nécessaires")
        
        return actions_required
    
    def get_position(self, symbol: Symbol, strategy_id: StrategyId) -> Optional[Position]:
        """Obtient une position spécifique"""
        return self.positions.get((symbol, strategy_id))
    
    def get_strategy_positions(self, strategy_id: StrategyId) -> List[Position]:
        """Obtient toutes les positions d'une stratégie"""
        return [pos for (sym, strat), pos in self.positions.items() if strat == strategy_id]
    
    def get_symbol_positions(self, symbol: Symbol) -> List[Position]:
        """Obtient toutes les positions sur un symbole"""
        return [pos for (sym, strat), pos in self.positions.items() if sym == symbol]
    
    async def get_portfolio_snapshot(self) -> PortfolioSnapshot:
        """Obtient l'état actuel du portefeuille"""
        await self._update_portfolio_metrics()
        return self.portfolio_history[-1] if self.portfolio_history else None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de performance complètes"""
        if not self.portfolio_history:
            return {}
        
        latest = self.portfolio_history[-1]
        
        return {
            "total_equity": float(latest.total_equity),
            "total_pnl": float(latest.total_pnl),
            "total_return_pct": float((self.current_capital - self.initial_capital) / self.initial_capital * 100),
            "daily_pnl": float(latest.daily_pnl),
            "positions_count": latest.positions_count,
            "sharpe_ratio": latest.sharpe_ratio,
            "max_drawdown_pct": latest.max_drawdown * 100,
            "var_95": float(latest.var_95),
            "var_99": float(latest.var_99),
            "cash_balance": float(self.cash_balance),
            "margin_used": float(self.margin_used),
            "positions_value": float(latest.positions_value)
        }
    
    async def emergency_liquidation(self, reason: str) -> Dict[str, Any]:
        """
        Liquidation d'urgence de toutes les positions
        Utilisé par les circuit breakers en cas de crise
        """
        self.logger.critical(f"LIQUIDATION D'URGENCE DÉCLENCHÉE: {reason}")
        
        liquidation_report = {
            "timestamp": datetime.now(timezone.utc),
            "reason": reason,
            "positions_closed": 0,
            "total_value_liquidated": Decimal("0"),
            "total_loss": Decimal("0")
        }
        
        async with self._lock:
            positions_to_close = list(self.positions.items())
            
            for (symbol, strategy_id), position in positions_to_close:
                # Simuler la fermeture au prix actuel (en production, utiliser les prix réels)
                close_price = position.current_price * Decimal("0.995")  # 0.5% de slippage d'urgence
                
                pnl = await self.close_position(
                    symbol=symbol,
                    strategy_id=strategy_id,
                    quantity=position.quantity,
                    price=Price(close_price),
                    order_id=OrderId(f"EMERGENCY_{symbol}_{strategy_id}"),
                    fees=position.quantity * close_price * Decimal("0.002")  # 0.2% de frais
                )
                
                liquidation_report["positions_closed"] += 1
                liquidation_report["total_value_liquidated"] += position.market_value
                if pnl and pnl < 0:
                    liquidation_report["total_loss"] += abs(pnl)
        
        return liquidation_report
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialise l'état du portefeuille en dictionnaire"""
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "initial_capital": float(self.initial_capital),
            "current_capital": float(self.current_capital),
            "cash_balance": float(self.cash_balance),
            "positions": [
                {
                    "symbol": pos.symbol,
                    "strategy_id": pos.strategy_id,
                    "quantity": float(pos.quantity),
                    "avg_entry_price": float(pos.avg_entry_price),
                    "current_price": float(pos.current_price),
                    "unrealized_pnl": float(pos.unrealized_pnl),
                    "realized_pnl": float(pos.realized_pnl)
                }
                for pos in self.positions.values()
            ],
            "performance_metrics": self.get_performance_metrics()
        }


# Implémentations concrètes des interfaces

class DefaultRiskCalculator:
    """Calculateur de risque par défaut"""
    
    async def calculate_var(self, positions: List[Position], confidence: float) -> Decimal:
        """Calcul simplifié de la VaR"""
        if not positions:
            return Decimal("0")
        
        # Simuler un calcul de VaR (en production, utiliser des modèles sophistiqués)
        total_value = sum(pos.market_value for pos in positions)
        # VaR approximative basée sur la volatilité historique
        volatility = 0.02  # 2% de volatilité journalière
        z_score = 2.33 if confidence == 0.99 else 1.645  # 99% ou 95%
        
        return total_value * Decimal(str(volatility * z_score))
    
    async def calculate_expected_shortfall(self, positions: List[Position], confidence: float) -> Decimal:
        """Calcul de l'Expected Shortfall (CVaR)"""
        var = await self.calculate_var(positions, confidence)
        # ES approximatif = 1.25 * VaR pour une distribution normale
        return var * Decimal("1.25")
    
    async def calculate_correlations(self, positions: List[Position]) -> np.ndarray:
        """Calcul des corrélations entre positions"""
        if len(positions) < 2:
            return np.array([[1.0]])
        
        # Simuler une matrice de corrélation (en production, utiliser les données historiques)
        n = len(positions)
        corr_matrix = np.random.rand(n, n) * 0.6 + 0.2
        np.fill_diagonal(corr_matrix, 1.0)
        # Rendre symétrique
        corr_matrix = (corr_matrix + corr_matrix.T) / 2
        
        return corr_matrix


class KellyCriterionSizer:
    """Position sizer basé sur le critère de Kelly"""
    
    def __init__(self, kelly_fraction: float = 0.25):
        self.kelly_fraction = kelly_fraction  # Fraction du Kelly complet à utiliser
    
    async def calculate_position_size(
        self,
        signal_strength: float,
        volatility: float,
        capital: Decimal,
        existing_positions: List[Position]
    ) -> Quantity:
        """Calcule la taille optimale selon Kelly"""
        # Paramètres Kelly simplifiés
        win_rate = 0.55  # Taux de réussite estimé
        win_loss_ratio = 1.5  # Ratio gain/perte moyen
        
        # Formule de Kelly: f = (p*b - q)/b
        # où p = probabilité de gain, b = ratio gain/perte, q = 1-p
        p = win_rate * signal_strength  # Ajuster par la force du signal
        q = 1 - p
        b = win_loss_ratio
        
        kelly_full = (p * b - q) / b if b > 0 else 0
        kelly_adjusted = kelly_full * self.kelly_fraction
        
        # Convertir en taille de position
        position_value = capital * Decimal(str(max(0, min(kelly_adjusted, 0.1))))  # Max 10%
        
        # Ajuster pour la volatilité
        vol_adjustment = 1 / (1 + volatility)
        position_value *= Decimal(str(vol_adjustment))
        
        # Simuler la conversion en quantité (en production, utiliser le prix réel)
        price_estimate = Decimal("100")  # Placeholder
        quantity = Quantity(position_value / price_estimate)
        
        return quantity


# Exemple d'utilisation
if __name__ == "__main__":
    async def main():
        # Initialiser les composants
        risk_calc = DefaultRiskCalculator()
        position_sizer = KellyCriterionSizer()
        
        # Créer le portfolio manager
        portfolio = PortfolioManager(
            initial_capital=Decimal("1000000"),  # 1M USD
            risk_calculator=risk_calc,
            position_sizer=position_sizer,
            allocation_method=AllocationMethod.RISK_PARITY
        )
        
        # Exemple d'ouverture de position
        is_valid, error = await portfolio.validate_trade(
            symbol=Symbol("BTC-USD"),
            strategy_id=StrategyId("ARBITRAGE_001"),
            quantity=Quantity(Decimal("10")),
            price=Price(Decimal("45000")),
            order_type=OrderType.LIMIT
        )
        
        if is_valid:
            position = await portfolio.open_position(
                symbol=Symbol("BTC-USD"),
                strategy_id=StrategyId("ARBITRAGE_001"),
                quantity=Quantity(Decimal("10")),
                price=Price(Decimal("45000")),
                order_id=OrderId("ORDER_001"),
                fees=Decimal("45")  # 0.01% de frais
            )
            print(f"Position ouverte: {position}")
        else:
            print(f"Trade rejeté: {error}")
        
        # Obtenir les métriques
        metrics = portfolio.get_performance_metrics()
        print(f"Métriques du portefeuille: {json.dumps(metrics, indent=2)}")
    
    # Lancer l'exemple
    asyncio.run(main())