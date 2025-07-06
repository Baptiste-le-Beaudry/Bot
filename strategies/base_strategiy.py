"""
Base Strategy - Classe abstraite définissant l'interface commune pour toutes les stratégies
Fournit l'infrastructure de base pour le traitement des données, la génération de signaux,
la gestion des positions et l'intégration avec le système de trading

Architecture moderne avec support pour l'async, le ML/DRL et le hot-swapping
"""

import asyncio
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Protocol, NewType, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque, defaultdict
from enum import Enum
import json
import logging
import uuid

# Import des types du portfolio_manager
from core.portfolio_manager import Symbol, Price, Quantity, OrderId, OrderType

# Types spécifiques aux stratégies
StrategyId = NewType('StrategyId', str)
SignalStrength = NewType('SignalStrength', float)
Confidence = NewType('Confidence', float)


class SignalType(Enum):
    """Types de signaux de trading"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE_LONG = "CLOSE_LONG"
    CLOSE_SHORT = "CLOSE_SHORT"
    SCALE_IN = "SCALE_IN"
    SCALE_OUT = "SCALE_OUT"


class StrategyState(Enum):
    """États possibles d'une stratégie"""
    INITIALIZING = "INITIALIZING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class MarketRegime(Enum):
    """Régimes de marché détectés"""
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    BREAKOUT = "BREAKOUT"
    MEAN_REVERTING = "MEAN_REVERTING"


@dataclass
class MarketData:
    """Structure pour les données de marché"""
    symbol: Symbol
    timestamp: datetime
    price: Price
    volume: Decimal
    bid: Price
    ask: Price
    bid_size: Quantity
    ask_size: Quantity
    high_24h: Optional[Price] = None
    low_24h: Optional[Price] = None
    vwap: Optional[Price] = None
    
    @property
    def spread(self) -> Decimal:
        """Calcul du spread bid-ask"""
        return self.ask - self.bid
    
    @property
    def mid_price(self) -> Price:
        """Prix milieu"""
        return Price((self.bid + self.ask) / 2)


@dataclass
class OrderBookSnapshot:
    """Snapshot du carnet d'ordres"""
    symbol: Symbol
    timestamp: datetime
    bids: List[Tuple[Price, Quantity]]  # [(price, quantity), ...]
    asks: List[Tuple[Price, Quantity]]
    
    def get_imbalance(self, levels: int = 5) -> float:
        """Calcule l'imbalance du carnet"""
        bid_volume = sum(q for p, q in self.bids[:levels])
        ask_volume = sum(q for p, q in self.asks[:levels])
        total_volume = bid_volume + ask_volume
        
        if total_volume > 0:
            return float((bid_volume - ask_volume) / total_volume)
        return 0.0


@dataclass
class TradingSignal:
    """Signal de trading généré par une stratégie"""
    strategy_id: StrategyId
    symbol: Symbol
    signal_type: SignalType
    strength: SignalStrength  # Force du signal [0, 1]
    confidence: Confidence    # Niveau de confiance [0, 1]
    quantity: Optional[Quantity] = None
    price: Optional[Price] = None
    stop_loss: Optional[Price] = None
    take_profit: Optional[Price] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    
    @property
    def is_expired(self) -> bool:
        """Vérifie si le signal a expiré"""
        if self.expires_at:
            return datetime.now(timezone.utc) > self.expires_at
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Sérialise le signal"""
        return {
            "strategy_id": self.strategy_id,
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "strength": float(self.strength),
            "confidence": float(self.confidence),
            "quantity": float(self.quantity) if self.quantity else None,
            "price": float(self.price) if self.price else None,
            "stop_loss": float(self.stop_loss) if self.stop_loss else None,
            "take_profit": float(self.take_profit) if self.take_profit else None,
            "time_in_force": self.time_in_force,
            "metadata": self.metadata,
            "generated_at": self.generated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None
        }


@dataclass
class StrategyMetrics:
    """Métriques de performance d'une stratégie"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = Decimal("0")
    gross_profit: Decimal = Decimal("0")
    gross_loss: Decimal = Decimal("0")
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    average_win: Decimal = Decimal("0")
    average_loss: Decimal = Decimal("0")
    largest_win: Decimal = Decimal("0")
    largest_loss: Decimal = Decimal("0")
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    recovery_factor: float = 0.0
    profit_factor: float = 0.0
    expectancy: Decimal = Decimal("0")
    
    @property
    def win_rate(self) -> float:
        """Taux de réussite"""
        if self.total_trades > 0:
            return self.winning_trades / self.total_trades
        return 0.0
    
    @property
    def risk_reward_ratio(self) -> float:
        """Ratio risque/récompense moyen"""
        if self.average_loss != 0:
            return float(self.average_win / abs(self.average_loss))
        return 0.0


class DataProvider(Protocol):
    """Interface pour fournir les données de marché"""
    async def get_latest_price(self, symbol: Symbol) -> MarketData:
        ...
    
    async def get_order_book(self, symbol: Symbol) -> OrderBookSnapshot:
        ...
    
    async def get_historical_data(
        self, 
        symbol: Symbol, 
        start: datetime, 
        end: datetime,
        interval: str
    ) -> pd.DataFrame:
        ...


class RiskManager(Protocol):
    """Interface pour la gestion des risques"""
    async def validate_signal(self, signal: TradingSignal) -> Tuple[bool, Optional[str]]:
        ...
    
    async def calculate_position_size(
        self,
        signal: TradingSignal,
        capital: Decimal,
        risk_per_trade: Decimal
    ) -> Quantity:
        ...


class BaseStrategy(ABC):
    """
    Classe de base abstraite pour toutes les stratégies de trading
    Fournit l'infrastructure commune et définit l'interface standard
    """
    
    def __init__(
        self,
        strategy_id: StrategyId,
        symbols: List[Symbol],
        data_provider: DataProvider,
        risk_manager: RiskManager,
        config: Optional[Dict[str, Any]] = None
    ):
        self.strategy_id = strategy_id
        self.symbols = symbols
        self.data_provider = data_provider
        self.risk_manager = risk_manager
        self.config = config or {}
        
        # État et contrôle
        self.state = StrategyState.INITIALIZING
        self._running = False
        self._tasks: List[asyncio.Task] = []
        
        # Buffers de données
        self.market_data_buffer: Dict[Symbol, deque] = {
            symbol: deque(maxlen=self.config.get('buffer_size', 1000))
            for symbol in symbols
        }
        self.order_book_buffer: Dict[Symbol, deque] = {
            symbol: deque(maxlen=self.config.get('orderbook_buffer_size', 100))
            for symbol in symbols
        }
        
        # Signaux et positions
        self.active_signals: Dict[Symbol, TradingSignal] = {}
        self.signal_history: deque = deque(maxlen=10000)
        self.position_tracker: Dict[Symbol, Dict[str, Any]] = defaultdict(dict)
        
        # Métriques
        self.metrics = StrategyMetrics()
        self.daily_metrics: deque = deque(maxlen=365)
        
        # Configuration des paramètres
        self.min_signal_strength = self.config.get('min_signal_strength', 0.5)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.max_positions = self.config.get('max_positions', 10)
        self.enable_ml = self.config.get('enable_ml', False)
        
        # Callbacks et hooks
        self._on_signal_callbacks: List[Callable] = []
        self._on_error_callbacks: List[Callable] = []
        
        # Logging
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{strategy_id}")
        
        # État spécifique à la stratégie
        self._strategy_state: Dict[str, Any] = {}
    
    # Méthodes abstraites à implémenter par les stratégies concrètes
    
    @abstractmethod
    async def analyze_market(self, symbol: Symbol) -> Optional[TradingSignal]:
        """
        Analyse le marché et génère un signal de trading
        Doit être implémenté par chaque stratégie
        """
        pass
    
    @abstractmethod
    async def calculate_indicators(self, symbol: Symbol) -> Dict[str, Any]:
        """
        Calcule les indicateurs techniques spécifiques à la stratégie
        """
        pass
    
    @abstractmethod
    def get_required_history_size(self) -> int:
        """
        Retourne la taille d'historique nécessaire pour la stratégie
        """
        pass
    
    # Méthodes du cycle de vie
    
    async def initialize(self) -> None:
        """Initialise la stratégie"""
        self.logger.info(f"Initialisation de la stratégie {self.strategy_id}")
        
        try:
            # Charger l'historique initial
            await self._load_initial_history()
            
            # Initialisation spécifique à la stratégie
            await self._on_initialize()
            
            self.state = StrategyState.RUNNING
            self.logger.info(f"Stratégie {self.strategy_id} initialisée avec succès")
            
        except Exception as e:
            self.state = StrategyState.ERROR
            self.logger.error(f"Erreur lors de l'initialisation: {str(e)}")
            raise
    
    async def start(self) -> None:
        """Démarre la stratégie"""
        if self.state != StrategyState.RUNNING:
            await self.initialize()
        
        self._running = True
        self.logger.info(f"Démarrage de la stratégie {self.strategy_id}")
        
        # Démarrer les tâches principales
        self._tasks.append(asyncio.create_task(self._main_loop()))
        self._tasks.append(asyncio.create_task(self._metrics_loop()))
        
        if self.enable_ml:
            self._tasks.append(asyncio.create_task(self._ml_update_loop()))
    
    async def stop(self) -> None:
        """Arrête la stratégie proprement"""
        self.logger.info(f"Arrêt de la stratégie {self.strategy_id}")
        self.state = StrategyState.STOPPING
        self._running = False
        
        # Annuler toutes les tâches
        for task in self._tasks:
            task.cancel()
        
        # Attendre la fin des tâches
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Nettoyage final
        await self._on_stop()
        
        self.state = StrategyState.STOPPED
        self.logger.info(f"Stratégie {self.strategy_id} arrêtée")
    
    async def pause(self) -> None:
        """Met en pause la stratégie"""
        self.state = StrategyState.PAUSED
        self.logger.info(f"Stratégie {self.strategy_id} mise en pause")
    
    async def resume(self) -> None:
        """Reprend l'exécution de la stratégie"""
        self.state = StrategyState.RUNNING
        self.logger.info(f"Stratégie {self.strategy_id} reprise")
    
    # Méthodes principales de traitement
    
    async def _main_loop(self) -> None:
        """Boucle principale de la stratégie"""
        while self._running:
            try:
                if self.state == StrategyState.RUNNING:
                    # Analyser chaque symbole
                    for symbol in self.symbols:
                        # Mettre à jour les données
                        await self._update_market_data(symbol)
                        
                        # Vérifier si nous avons assez d'historique
                        if len(self.market_data_buffer[symbol]) >= self.get_required_history_size():
                            # Analyser et générer un signal
                            signal = await self.analyze_market(symbol)
                            
                            if signal:
                                await self._process_signal(signal)
                
                # Attendre avant la prochaine itération
                await asyncio.sleep(self.config.get('update_interval', 1.0))
                
            except Exception as e:
                self.logger.error(f"Erreur dans la boucle principale: {str(e)}")
                await self._handle_error(e)
    
    async def _update_market_data(self, symbol: Symbol) -> None:
        """Met à jour les données de marché pour un symbole"""
        try:
            # Obtenir les dernières données
            market_data = await self.data_provider.get_latest_price(symbol)
            self.market_data_buffer[symbol].append(market_data)
            
            # Mettre à jour le carnet d'ordres si nécessaire
            if self.config.get('use_orderbook', False):
                order_book = await self.data_provider.get_order_book(symbol)
                self.order_book_buffer[symbol].append(order_book)
                
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour des données pour {symbol}: {str(e)}")
    
    async def _process_signal(self, signal: TradingSignal) -> None:
        """Traite un signal de trading"""
        # Vérifier la force et la confiance du signal
        if signal.strength < self.min_signal_strength:
            self.logger.debug(f"Signal trop faible ignoré: {signal.strength}")
            return
        
        if signal.confidence < self.min_confidence:
            self.logger.debug(f"Confiance insuffisante: {signal.confidence}")
            return
        
        # Valider avec le gestionnaire de risque
        is_valid, error = await self.risk_manager.validate_signal(signal)
        if not is_valid:
            self.logger.warning(f"Signal rejeté par le risk manager: {error}")
            return
        
        # Calculer la taille de position si non spécifiée
        if not signal.quantity:
            signal.quantity = await self.risk_manager.calculate_position_size(
                signal,
                Decimal("100000"),  # Capital placeholder
                Decimal(str(self.config.get('risk_per_trade', 0.02)))
            )
        
        # Enregistrer le signal
        self.active_signals[signal.symbol] = signal
        self.signal_history.append(signal)
        
        # Notifier les callbacks
        await self._notify_signal(signal)
        
        self.logger.info(f"Signal généré: {signal.signal_type.value} {signal.symbol} "
                        f"(force: {signal.strength:.2f}, confiance: {signal.confidence:.2f})")
    
    # Gestion des métriques
    
    async def _metrics_loop(self) -> None:
        """Boucle de mise à jour des métriques"""
        while self._running:
            try:
                await self._update_metrics()
                await asyncio.sleep(60)  # Mise à jour toutes les minutes
            except Exception as e:
                self.logger.error(f"Erreur dans la mise à jour des métriques: {str(e)}")
    
    async def _update_metrics(self) -> None:
        """Met à jour les métriques de performance"""
        # Calculer les métriques basées sur l'historique des signaux
        if not self.signal_history:
            return
        
        # Métriques de base (placeholder - à implémenter avec les données réelles)
        self.metrics.total_trades = len(self.signal_history)
        
        # Calculer le Sharpe ratio si nous avons des rendements
        if self.daily_metrics:
            returns = [m.get('return', 0) for m in self.daily_metrics]
            if len(returns) > 30:
                returns_array = np.array(returns)
                self.metrics.sharpe_ratio = np.sqrt(252) * np.mean(returns_array) / (np.std(returns_array) + 1e-8)
    
    def update_trade_result(self, symbol: Symbol, pnl: Decimal, is_win: bool) -> None:
        """Met à jour les métriques avec le résultat d'un trade"""
        self.metrics.total_trades += 1
        
        if is_win:
            self.metrics.winning_trades += 1
            self.metrics.gross_profit += pnl
            if pnl > self.metrics.largest_win:
                self.metrics.largest_win = pnl
        else:
            self.metrics.losing_trades += 1
            self.metrics.gross_loss += abs(pnl)
            if pnl < -self.metrics.largest_loss:
                self.metrics.largest_loss = abs(pnl)
        
        self.metrics.total_pnl += pnl
        
        # Recalculer les moyennes
        if self.metrics.winning_trades > 0:
            self.metrics.average_win = self.metrics.gross_profit / self.metrics.winning_trades
        if self.metrics.losing_trades > 0:
            self.metrics.average_loss = self.metrics.gross_loss / self.metrics.losing_trades
        
        # Profit factor
        if self.metrics.gross_loss > 0:
            self.metrics.profit_factor = float(self.metrics.gross_profit / self.metrics.gross_loss)
        
        # Expectancy
        if self.metrics.total_trades > 0:
            self.metrics.expectancy = self.metrics.total_pnl / self.metrics.total_trades
    
    # Hot-swapping et état
    
    async def save_state(self) -> Dict[str, Any]:
        """Sauvegarde l'état de la stratégie pour le hot-swapping"""
        state = {
            "strategy_id": self.strategy_id,
            "symbols": self.symbols,
            "metrics": {
                "total_trades": self.metrics.total_trades,
                "winning_trades": self.metrics.winning_trades,
                "total_pnl": float(self.metrics.total_pnl),
                "sharpe_ratio": self.metrics.sharpe_ratio
            },
            "active_signals": {
                symbol: signal.to_dict() 
                for symbol, signal in self.active_signals.items()
            },
            "position_tracker": dict(self.position_tracker),
            "strategy_specific": self._strategy_state,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Permettre aux stratégies d'ajouter leur état spécifique
        custom_state = await self._save_custom_state()
        if custom_state:
            state["custom"] = custom_state
        
        return state
    
    async def load_state(self, state: Dict[str, Any]) -> None:
        """Charge l'état sauvegardé"""
        # Restaurer les métriques
        metrics_data = state.get("metrics", {})
        self.metrics.total_trades = metrics_data.get("total_trades", 0)
        self.metrics.winning_trades = metrics_data.get("winning_trades", 0)
        self.metrics.total_pnl = Decimal(str(metrics_data.get("total_pnl", 0)))
        self.metrics.sharpe_ratio = metrics_data.get("sharpe_ratio", 0.0)
        
        # Restaurer les positions
        self.position_tracker = defaultdict(dict, state.get("position_tracker", {}))
        
        # Restaurer l'état spécifique
        self._strategy_state = state.get("strategy_specific", {})
        
        # Permettre aux stratégies de restaurer leur état custom
        if "custom" in state:
            await self._load_custom_state(state["custom"])
        
        self.logger.info(f"État restauré depuis {state.get('timestamp')}")
    
    # Méthodes utilitaires
    
    def get_latest_data(self, symbol: Symbol) -> Optional[MarketData]:
        """Obtient les dernières données pour un symbole"""
        buffer = self.market_data_buffer.get(symbol)
        if buffer and len(buffer) > 0:
            return buffer[-1]
        return None
    
    def get_data_series(self, symbol: Symbol, field: str, length: int) -> np.ndarray:
        """Obtient une série de données pour les calculs"""
        buffer = self.market_data_buffer.get(symbol)
        if not buffer or len(buffer) < length:
            return np.array([])
        
        data = []
        for i in range(-length, 0):
            market_data = buffer[i]
            value = getattr(market_data, field, None)
            if value is not None:
                data.append(float(value))
        
        return np.array(data)
    
    def detect_market_regime(self, symbol: Symbol) -> MarketRegime:
        """Détecte le régime de marché actuel"""
        # Implémentation simple basée sur la volatilité et la tendance
        prices = self.get_data_series(symbol, 'price', 50)
        
        if len(prices) < 50:
            return MarketRegime.RANGING
        
        # Calculer la tendance
        sma_short = np.mean(prices[-10:])
        sma_long = np.mean(prices[-50:])
        
        # Calculer la volatilité
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        
        # Déterminer le régime
        if volatility > 0.03:  # Seuil de haute volatilité
            return MarketRegime.HIGH_VOLATILITY
        elif volatility < 0.01:  # Seuil de basse volatilité
            return MarketRegime.LOW_VOLATILITY
        elif sma_short > sma_long * 1.02:  # Tendance haussière
            return MarketRegime.TRENDING_UP
        elif sma_short < sma_long * 0.98:  # Tendance baissière
            return MarketRegime.TRENDING_DOWN
        else:
            return MarketRegime.RANGING
    
    # Callbacks et notifications
    
    def register_signal_callback(self, callback: Callable) -> None:
        """Enregistre un callback pour les signaux"""
        self._on_signal_callbacks.append(callback)
    
    def register_error_callback(self, callback: Callable) -> None:
        """Enregistre un callback pour les erreurs"""
        self._on_error_callbacks.append(callback)
    
    async def _notify_signal(self, signal: TradingSignal) -> None:
        """Notifie tous les callbacks d'un nouveau signal"""
        for callback in self._on_signal_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            except Exception as e:
                self.logger.error(f"Erreur dans le callback de signal: {str(e)}")
    
    async def _handle_error(self, error: Exception) -> None:
        """Gère les erreurs et notifie les callbacks"""
        for callback in self._on_error_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(self.strategy_id, error)
                else:
                    callback(self.strategy_id, error)
            except Exception as e:
                self.logger.error(f"Erreur dans le callback d'erreur: {str(e)}")
    
    # Hooks pour les stratégies dérivées
    
    async def _on_initialize(self) -> None:
        """Hook appelé lors de l'initialisation"""
        pass
    
    async def _on_stop(self) -> None:
        """Hook appelé lors de l'arrêt"""
        pass
    
    async def _save_custom_state(self) -> Optional[Dict[str, Any]]:
        """Hook pour sauvegarder l'état custom de la stratégie"""
        return None
    
    async def _load_custom_state(self, state: Dict[str, Any]) -> None:
        """Hook pour charger l'état custom de la stratégie"""
        pass
    
    async def _load_initial_history(self) -> None:
        """Charge l'historique initial nécessaire"""
        required_size = self.get_required_history_size()
        
        for symbol in self.symbols:
            end_time = datetime.now(timezone.utc)
            start_time = end_time - pd.Timedelta(days=required_size // 1440)  # Approximation
            
            try:
                historical_data = await self.data_provider.get_historical_data(
                    symbol, start_time, end_time, "1m"
                )
                
                # Convertir et stocker dans le buffer
                for _, row in historical_data.iterrows():
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=row['timestamp'],
                        price=Price(Decimal(str(row['close']))),
                        volume=Decimal(str(row['volume'])),
                        bid=Price(Decimal(str(row.get('bid', row['close'])))),
                        ask=Price(Decimal(str(row.get('ask', row['close'])))),
                        bid_size=Quantity(Decimal("0")),
                        ask_size=Quantity(Decimal("0"))
                    )
                    self.market_data_buffer[symbol].append(market_data)
                    
            except Exception as e:
                self.logger.error(f"Erreur lors du chargement de l'historique pour {symbol}: {str(e)}")
    
    async def _ml_update_loop(self) -> None:
        """Boucle de mise à jour ML (si activé)"""
        while self._running:
            try:
                # Placeholder pour l'intégration ML
                # Les stratégies peuvent override cette méthode
                await asyncio.sleep(300)  # Mise à jour toutes les 5 minutes
            except Exception as e:
                self.logger.error(f"Erreur dans la mise à jour ML: {str(e)}")
    
    # Méthodes de diagnostic et debugging
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Retourne des informations de diagnostic"""
        return {
            "strategy_id": self.strategy_id,
            "state": self.state.value,
            "symbols": self.symbols,
            "buffer_sizes": {
                symbol: len(buffer) 
                for symbol, buffer in self.market_data_buffer.items()
            },
            "active_signals": len(self.active_signals),
            "total_signals_generated": len(self.signal_history),
            "metrics": {
                "total_trades": self.metrics.total_trades,
                "win_rate": self.metrics.win_rate,
                "sharpe_ratio": self.metrics.sharpe_ratio,
                "total_pnl": float(self.metrics.total_pnl)
            }
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.strategy_id}, symbols={self.symbols}, state={self.state.value})"


# Exemple d'implémentation concrète
class ExampleStrategy(BaseStrategy):
    """Exemple de stratégie concrète pour démonstration"""
    
    async def analyze_market(self, symbol: Symbol) -> Optional[TradingSignal]:
        """Implémentation simple d'analyse de marché"""
        # Obtenir les données récentes
        prices = self.get_data_series(symbol, 'price', 20)
        
        if len(prices) < 20:
            return None
        
        # Calcul simple de moyenne mobile
        sma_fast = np.mean(prices[-5:])
        sma_slow = np.mean(prices[-20:])
        
        # Générer un signal basique
        if sma_fast > sma_slow:
            return TradingSignal(
                strategy_id=self.strategy_id,
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=SignalStrength(0.7),
                confidence=Confidence(0.8),
                metadata={"sma_fast": float(sma_fast), "sma_slow": float(sma_slow)}
            )
        
        return None
    
    async def calculate_indicators(self, symbol: Symbol) -> Dict[str, Any]:
        """Calcule les indicateurs techniques"""
        prices = self.get_data_series(symbol, 'price', 50)
        
        if len(prices) < 50:
            return {}
        
        return {
            "sma_5": np.mean(prices[-5:]),
            "sma_20": np.mean(prices[-20:]),
            "sma_50": np.mean(prices),
            "volatility": np.std(prices[-20:])
        }
    
    def get_required_history_size(self) -> int:
        """Taille d'historique requise"""
        return 50  # 50 périodes pour calculer la SMA 50