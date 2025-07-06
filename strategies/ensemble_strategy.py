"""
Ensemble Strategy - Combinaison intelligente de multiples stratégies
Optimise l'allocation et la sélection des stratégies selon les conditions de marché

Caractéristiques principales:
- Sélection dynamique basée sur le régime de marché
- Pondération adaptative selon les performances
- Corrélation et diversification des signaux
- Meta-learning pour l'optimisation continue
- Gestion unifiée du risque multi-stratégies
Performance cible: Sharpe > 2.5, Drawdown < 10%
"""

import asyncio
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict
from enum import Enum
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from strategies.base_strategy import (
    BaseStrategy, TradingSignal, SignalType, MarketData,
    Symbol, StrategyId, SignalStrength, Confidence,
    Price, Quantity, MarketRegime
)
from strategies.statistical_arbitrage import StatisticalArbitrageStrategy
from strategies.market_making import MarketMakingStrategy
from strategies.scalping import ScalpingStrategy


class StrategyAllocation(Enum):
    """Modes d'allocation entre stratégies"""
    EQUAL_WEIGHT = "EQUAL_WEIGHT"
    PERFORMANCE_BASED = "PERFORMANCE_BASED"
    RISK_PARITY = "RISK_PARITY"
    REGIME_BASED = "REGIME_BASED"
    ML_OPTIMIZED = "ML_OPTIMIZED"


@dataclass
class StrategyPerformance:
    """Métriques de performance pour une stratégie"""
    strategy_id: StrategyId
    total_signals: int = 0
    successful_signals: int = 0
    total_pnl: Decimal = Decimal("0")
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    avg_return: float = 0.0
    volatility: float = 0.0
    correlation_to_market: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    rolling_performance: deque = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def success_rate(self) -> float:
        """Taux de succès des signaux"""
        if self.total_signals > 0:
            return self.successful_signals / self.total_signals
        return 0.0
    
    @property
    def risk_adjusted_return(self) -> float:
        """Rendement ajusté au risque"""
        if self.volatility > 0:
            return self.avg_return / self.volatility
        return 0.0


@dataclass
class EnsembleSignal:
    """Signal composite provenant de plusieurs stratégies"""
    primary_signal: TradingSignal
    supporting_signals: List[TradingSignal]
    consensus_score: float  # 0-1, niveau de consensus
    diversity_score: float  # 0-1, diversité des sources
    confidence_boost: float  # Multiplicateur de confiance
    conflict_resolution: str  # Méthode utilisée si conflit
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_strength(self) -> float:
        """Force totale du signal ensemble"""
        weights = [0.6]  # Poids du signal principal
        strengths = [float(self.primary_signal.strength)]
        
        # Ajouter les signaux de support avec poids décroissants
        for i, signal in enumerate(self.supporting_signals):
            weight = 0.4 / (i + 2)  # Poids décroissant
            weights.append(weight)
            strengths.append(float(signal.strength))
        
        # Normaliser les poids
        total_weight = sum(weights)
        normalized_weights = [w/total_weight for w in weights]
        
        # Calculer la force pondérée
        weighted_strength = sum(w * s for w, s in zip(normalized_weights, strengths))
        
        # Appliquer le boost de consensus
        return min(weighted_strength * (1 + self.consensus_score * 0.5), 1.0)


@dataclass
class MarketContext:
    """Contexte de marché pour la sélection de stratégies"""
    regime: MarketRegime
    volatility_percentile: float  # 0-100
    volume_percentile: float
    trend_strength: float  # -1 à 1
    correlation_matrix: np.ndarray
    liquidity_score: float  # 0-1
    event_risk: float  # 0-1
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_regime_scores(self) -> Dict[str, float]:
        """Scores de pertinence pour chaque stratégie selon le contexte"""
        scores = {}
        
        # Arbitrage statistique - bon en marché latéral, moyenne volatilité
        if self.regime == MarketRegime.RANGING:
            scores['statistical_arbitrage'] = 0.9
        elif self.regime == MarketRegime.MEAN_REVERTING:
            scores['statistical_arbitrage'] = 0.95
        elif self.regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            scores['statistical_arbitrage'] = 0.4
        else:
            scores['statistical_arbitrage'] = 0.6
        
        # Market making - bon avec haute liquidité, faible volatilité
        mm_score = 0.5
        if self.liquidity_score > 0.7:
            mm_score += 0.3
        if self.volatility_percentile < 30:
            mm_score += 0.2
        elif self.volatility_percentile > 70:
            mm_score -= 0.3
        scores['market_making'] = max(0.2, min(1.0, mm_score))
        
        # Scalping - bon avec haute liquidité et momentum
        scalp_score = 0.5
        if self.liquidity_score > 0.8:
            scalp_score += 0.2
        if abs(self.trend_strength) > 0.5:
            scalp_score += 0.3
        if self.volatility_percentile > 50:
            scalp_score += 0.1
        scores['scalping'] = max(0.3, min(1.0, scalp_score))
        
        return scores


class EnsembleStrategy(BaseStrategy):
    """
    Stratégie ensemble combinant plusieurs sous-stratégies
    Optimise dynamiquement l'allocation et la sélection
    """
    
    def __init__(
        self,
        strategy_id: StrategyId,
        symbols: List[Symbol],
        data_provider,
        risk_manager,
        config: Optional[Dict[str, Any]] = None
    ):
        # Configuration par défaut
        default_config = {
            # Stratégies à inclure
            'enabled_strategies': [
                'statistical_arbitrage',
                'market_making',
                'scalping'
            ],
            
            # Allocation
            'allocation_method': StrategyAllocation.ML_OPTIMIZED,
            'min_allocation': 0.1,  # 10% minimum par stratégie active
            'max_allocation': 0.5,  # 50% maximum par stratégie
            'rebalance_frequency': 3600,  # Secondes
            
            # Sélection de signaux
            'min_consensus': 0.3,  # Consensus minimum pour agir
            'require_confirmation': True,  # Exiger confirmation d'au moins 2 stratégies
            'conflict_resolution': 'weighted_vote',  # ou 'primary_only', 'ml_arbiter'
            
            # Risk management
            'correlation_limit': 0.7,  # Limite de corrélation entre positions
            'max_strategies_per_symbol': 2,  # Stratégies simultanées max par symbole
            'position_overlap_limit': 0.5,  # Chevauchement max des positions
            
            # Machine Learning
            'use_ml_selection': True,
            'ml_retrain_frequency': 86400,  # 24h
            'ml_features': [
                'regime', 'volatility', 'volume', 'spread',
                'past_performance', 'correlation'
            ],
            
            # Performance tracking
            'performance_window': 1000,  # Trades pour calculer les stats
            'adaptation_speed': 0.1,  # Vitesse d'adaptation des poids
            
            'buffer_size': 1000,
            'update_interval': 1.0
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(strategy_id, symbols, data_provider, risk_manager, default_config)
        
        # Initialiser les sous-stratégies
        self.sub_strategies: Dict[str, BaseStrategy] = {}
        self._initialize_sub_strategies()
        
        # Performance tracking
        self.strategy_performance: Dict[str, StrategyPerformance] = {
            name: StrategyPerformance(strategy_id=StrategyId(name))
            for name in self.config['enabled_strategies']
        }
        
        # Allocation et poids
        self.strategy_weights: Dict[str, float] = {
            name: 1.0 / len(self.config['enabled_strategies'])
            for name in self.config['enabled_strategies']
        }
        self.last_rebalance = datetime.now(timezone.utc)
        
        # Contexte de marché
        self.market_context: Dict[Symbol, MarketContext] = {}
        self.regime_history: deque = deque(maxlen=100)
        
        # Machine Learning
        self.strategy_selector = None  # Modèle de sélection
        self.signal_combiner = None    # Modèle de combinaison
        self.performance_predictor = None  # Prédiction de performance
        self.scaler = StandardScaler()
        
        # Historique des signaux
        self.signal_history: deque = deque(maxlen=5000)
        self.ensemble_signals: Dict[Symbol, EnsembleSignal] = {}
        
        # Métriques ensemble
        self.total_signals_generated = 0
        self.consensus_signals = 0
        self.conflict_resolutions = defaultdict(int)
    
    def _initialize_sub_strategies(self) -> None:
        """Initialise les sous-stratégies selon la configuration"""
        for strategy_name in self.config['enabled_strategies']:
            if strategy_name == 'statistical_arbitrage':
                self.sub_strategies[strategy_name] = StatisticalArbitrageStrategy(
                    strategy_id=StrategyId(f"{self.strategy_id}_stat_arb"),
                    symbols=self.symbols,
                    data_provider=self.data_provider,
                    risk_manager=self.risk_manager,
                    config={
                        'max_pairs': 5,
                        'use_ml': True,
                        'buffer_size': self.config['buffer_size']
                    }
                )
            
            elif strategy_name == 'market_making':
                self.sub_strategies[strategy_name] = MarketMakingStrategy(
                    strategy_id=StrategyId(f"{self.strategy_id}_mm"),
                    symbols=self.symbols,
                    data_provider=self.data_provider,
                    risk_manager=self.risk_manager,
                    config={
                        'quote_levels': 2,
                        'use_ml_quotes': True,
                        'buffer_size': self.config['buffer_size']
                    }
                )
            
            elif strategy_name == 'scalping':
                self.sub_strategies[strategy_name] = ScalpingStrategy(
                    strategy_id=StrategyId(f"{self.strategy_id}_scalp"),
                    symbols=self.symbols,
                    data_provider=self.data_provider,
                    risk_manager=self.risk_manager,
                    config={
                        'max_daily_trades': 200,
                        'use_ml_filter': True,
                        'buffer_size': self.config['buffer_size']
                    }
                )
    
    async def analyze_market(self, symbol: Symbol) -> Optional[TradingSignal]:
        """
        Analyse ensemble combinant plusieurs stratégies
        """
        # Mettre à jour le contexte de marché
        await self._update_market_context(symbol)
        
        # Vérifier s'il est temps de rééquilibrer
        await self._check_rebalance()
        
        # Collecter les signaux de toutes les stratégies actives
        strategy_signals = await self._collect_strategy_signals(symbol)
        
        if not strategy_signals:
            return None
        
        # Créer un signal ensemble
        ensemble_signal = await self._create_ensemble_signal(symbol, strategy_signals)
        
        if not ensemble_signal:
            return None
        
        # Valider le signal ensemble
        if not self._validate_ensemble_signal(ensemble_signal):
            return None
        
        # Ajuster la taille de position selon l'allocation
        final_signal = self._adjust_position_size(ensemble_signal)
        
        # Enregistrer pour l'apprentissage
        self._record_signal(final_signal, ensemble_signal)
        
        return final_signal
    
    async def _update_market_context(self, symbol: Symbol) -> None:
        """Met à jour le contexte de marché pour un symbole"""
        # Détecter le régime de marché
        regime = self.detect_market_regime(symbol)
        
        # Calculer la volatilité
        prices = self.get_data_series(symbol, 'price', 100)
        if len(prices) >= 20:
            returns = np.diff(prices) / prices[:-1]
            current_vol = np.std(returns[-20:]) * np.sqrt(252)
            
            # Calculer le percentile de volatilité
            all_vols = [np.std(returns[i:i+20]) for i in range(len(returns)-20)]
            if all_vols:
                vol_percentile = float(np.percentile(all_vols, 
                    [i for i in range(101) if current_vol >= np.percentile(all_vols, i)][-1]))
            else:
                vol_percentile = 50.0
        else:
            current_vol = 0.2
            vol_percentile = 50.0
        
        # Calculer le volume percentile
        volumes = self.get_data_series(symbol, 'volume', 100)
        if len(volumes) >= 20:
            current_volume = np.mean(volumes[-5:])
            volume_percentile = float(
                (volumes < current_volume).sum() / len(volumes) * 100
            )
        else:
            volume_percentile = 50.0
        
        # Calculer la force de tendance
        if len(prices) >= 50:
            sma_short = np.mean(prices[-10:])
            sma_long = np.mean(prices[-50:])
            price_now = prices[-1]
            
            trend_strength = (sma_short - sma_long) / sma_long
            trend_strength = max(-1, min(1, trend_strength * 10))  # Normaliser
        else:
            trend_strength = 0.0
        
        # Score de liquidité
        latest_data = self.get_latest_data(symbol)
        if latest_data:
            spread_bps = float((latest_data.ask - latest_data.bid) / latest_data.mid_price * 10000)
            liquidity_score = 1.0 / (1.0 + spread_bps / 10)  # Plus le spread est faible, meilleure est la liquidité
        else:
            liquidity_score = 0.5
        
        # Créer le contexte
        self.market_context[symbol] = MarketContext(
            regime=regime,
            volatility_percentile=vol_percentile,
            volume_percentile=volume_percentile,
            trend_strength=trend_strength,
            correlation_matrix=np.eye(len(self.symbols)),  # Placeholder
            liquidity_score=liquidity_score,
            event_risk=0.0  # Placeholder - pourrait intégrer un calendrier économique
        )
    
    async def _collect_strategy_signals(
        self, 
        symbol: Symbol
    ) -> Dict[str, Optional[TradingSignal]]:
        """Collecte les signaux de toutes les stratégies actives"""
        strategy_signals = {}
        
        # Obtenir les scores de pertinence selon le contexte
        context = self.market_context.get(symbol)
        if context:
            regime_scores = context.get_regime_scores()
        else:
            regime_scores = {name: 0.5 for name in self.config['enabled_strategies']}
        
        # Collecter les signaux en parallèle
        tasks = []
        active_strategies = []
        
        for name, strategy in self.sub_strategies.items():
            # Ne pas interroger les stratégies avec un score trop faible
            if regime_scores.get(name, 0) >= 0.3:
                tasks.append(strategy.analyze_market(symbol))
                active_strategies.append(name)
        
        if tasks:
            signals = await asyncio.gather(*tasks, return_exceptions=True)
            
            for name, signal in zip(active_strategies, signals):
                if isinstance(signal, Exception):
                    self.logger.error(f"Erreur dans {name}: {str(signal)}")
                    strategy_signals[name] = None
                else:
                    strategy_signals[name] = signal
        
        return strategy_signals
    
    async def _create_ensemble_signal(
        self,
        symbol: Symbol,
        strategy_signals: Dict[str, Optional[TradingSignal]]
    ) -> Optional[EnsembleSignal]:
        """Crée un signal ensemble à partir des signaux individuels"""
        # Filtrer les signaux valides
        valid_signals = [(name, sig) for name, sig in strategy_signals.items() if sig is not None]
        
        if not valid_signals:
            return None
        
        # Grouper par direction
        long_signals = [(n, s) for n, s in valid_signals if s.signal_type in [SignalType.BUY]]
        short_signals = [(n, s) for n, s in valid_signals if s.signal_type in [SignalType.SELL]]
        
        # Déterminer la direction dominante
        if len(long_signals) > len(short_signals):
            primary_direction = 'long'
            primary_signals = long_signals
            conflicting_signals = short_signals
        elif len(short_signals) > len(long_signals):
            primary_direction = 'short'
            primary_signals = short_signals
            conflicting_signals = long_signals
        else:
            # Égalité - utiliser la force des signaux
            long_strength = sum(s.strength * s.confidence for _, s in long_signals)
            short_strength = sum(s.strength * s.confidence for _, s in short_signals)
            
            if long_strength > short_strength:
                primary_direction = 'long'
                primary_signals = long_signals
                conflicting_signals = short_signals
            else:
                primary_direction = 'short'
                primary_signals = short_signals
                conflicting_signals = long_signals
        
        # Sélectionner le signal principal (le plus fort)
        primary_name, primary_signal = max(
            primary_signals,
            key=lambda x: float(x[1].strength * x[1].confidence * self.strategy_weights.get(x[0], 1.0))
        )
        
        # Signaux de support
        supporting_signals = [s for n, s in primary_signals if n != primary_name]
        
        # Calculer le consensus
        total_strategies = len(valid_signals)
        agreeing_strategies = len(primary_signals)
        consensus_score = agreeing_strategies / total_strategies if total_strategies > 0 else 0
        
        # Calculer la diversité
        unique_strategies = len(set(n for n, _ in primary_signals))
        diversity_score = unique_strategies / len(self.config['enabled_strategies'])
        
        # Résolution de conflit
        conflict_resolution = "none"
        if conflicting_signals:
            conflict_resolution = await self._resolve_signal_conflict(
                primary_signals, conflicting_signals, symbol
            )
            self.conflict_resolutions[conflict_resolution] += 1
        
        # Boost de confiance basé sur le consensus
        confidence_boost = 1.0
        if consensus_score > 0.7:
            confidence_boost = 1.2
        elif consensus_score > 0.5:
            confidence_boost = 1.1
        
        # Métadonnées
        metadata = {
            'primary_strategy': primary_name,
            'supporting_strategies': [n for n, _ in primary_signals if n != primary_name],
            'conflicting_strategies': [n for n, _ in conflicting_signals],
            'regime': self.market_context[symbol].regime.value if symbol in self.market_context else None,
            'strategy_weights': dict(self.strategy_weights)
        }
        
        return EnsembleSignal(
            primary_signal=primary_signal,
            supporting_signals=supporting_signals,
            consensus_score=consensus_score,
            diversity_score=diversity_score,
            confidence_boost=confidence_boost,
            conflict_resolution=conflict_resolution,
            metadata=metadata
        )
    
    async def _resolve_signal_conflict(
        self,
        primary_signals: List[Tuple[str, TradingSignal]],
        conflicting_signals: List[Tuple[str, TradingSignal]],
        symbol: Symbol
    ) -> str:
        """Résout les conflits entre signaux"""
        method = self.config['conflict_resolution']
        
        if method == 'weighted_vote':
            # Vote pondéré par les poids des stratégies
            primary_weight = sum(
                self.strategy_weights.get(name, 1.0) * signal.strength * signal.confidence
                for name, signal in primary_signals
            )
            conflict_weight = sum(
                self.strategy_weights.get(name, 1.0) * signal.strength * signal.confidence
                for name, signal in conflicting_signals
            )
            
            if primary_weight < conflict_weight * 1.5:  # Pas assez de marge
                return "weighted_vote_insufficient"
            return "weighted_vote_resolved"
        
        elif method == 'primary_only':
            # Ignorer simplement les signaux conflictuels
            return "primary_only"
        
        elif method == 'ml_arbiter' and self.signal_combiner:
            # Utiliser le ML pour arbitrer
            features = self._extract_conflict_features(
                primary_signals, conflicting_signals, symbol
            )
            if features is not None:
                decision = self.signal_combiner.predict([features])[0]
                return f"ml_arbiter_{decision}"
        
        return "default_resolution"
    
    def _validate_ensemble_signal(self, ensemble_signal: EnsembleSignal) -> bool:
        """Valide un signal ensemble selon les critères configurés"""
        # Vérifier le consensus minimum
        if ensemble_signal.consensus_score < self.config['min_consensus']:
            return False
        
        # Vérifier la confirmation si requise
        if self.config['require_confirmation']:
            total_signals = 1 + len(ensemble_signal.supporting_signals)
            if total_signals < 2:
                return False
        
        # Vérifier la force totale
        if ensemble_signal.total_strength < 0.5:
            return False
        
        return True
    
    def _adjust_position_size(self, ensemble_signal: EnsembleSignal) -> TradingSignal:
        """Ajuste la taille de position selon l'allocation de la stratégie"""
        signal = ensemble_signal.primary_signal
        primary_strategy = ensemble_signal.metadata['primary_strategy']
        
        # Obtenir le poids de la stratégie
        strategy_weight = self.strategy_weights.get(primary_strategy, 1.0)
        
        # Ajuster la quantité
        if signal.quantity:
            adjusted_quantity = Quantity(signal.quantity * Decimal(str(strategy_weight)))
        else:
            # Calculer basé sur l'allocation
            risk_amount = self.current_capital * Decimal(str(0.02 * strategy_weight))
            adjusted_quantity = Quantity(risk_amount / Decimal("100"))  # Placeholder
        
        # Créer le signal ajusté
        adjusted_signal = TradingSignal(
            strategy_id=self.strategy_id,
            symbol=signal.symbol,
            signal_type=signal.signal_type,
            strength=SignalStrength(ensemble_signal.total_strength),
            confidence=Confidence(
                min(float(signal.confidence) * ensemble_signal.confidence_boost, 0.95)
            ),
            quantity=adjusted_quantity,
            price=signal.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            time_in_force=signal.time_in_force,
            metadata={
                **signal.metadata,
                'ensemble_metadata': {
                    'consensus_score': ensemble_signal.consensus_score,
                    'diversity_score': ensemble_signal.diversity_score,
                    'primary_strategy': primary_strategy,
                    'supporting_count': len(ensemble_signal.supporting_signals)
                }
            }
        )
        
        return adjusted_signal
    
    async def _check_rebalance(self) -> None:
        """Vérifie et effectue le rééquilibrage si nécessaire"""
        now = datetime.now(timezone.utc)
        time_since_rebalance = (now - self.last_rebalance).total_seconds()
        
        if time_since_rebalance < self.config['rebalance_frequency']:
            return
        
        self.logger.info("Début du rééquilibrage des stratégies")
        
        # Calculer les nouveaux poids
        if self.config['allocation_method'] == StrategyAllocation.ML_OPTIMIZED:
            new_weights = await self._calculate_ml_weights()
        elif self.config['allocation_method'] == StrategyAllocation.PERFORMANCE_BASED:
            new_weights = self._calculate_performance_weights()
        elif self.config['allocation_method'] == StrategyAllocation.RISK_PARITY:
            new_weights = self._calculate_risk_parity_weights()
        elif self.config['allocation_method'] == StrategyAllocation.REGIME_BASED:
            new_weights = self._calculate_regime_weights()
        else:
            # Equal weight
            new_weights = {
                name: 1.0 / len(self.config['enabled_strategies'])
                for name in self.config['enabled_strategies']
            }
        
        # Appliquer les limites
        for name in new_weights:
            new_weights[name] = max(
                self.config['min_allocation'],
                min(self.config['max_allocation'], new_weights[name])
            )
        
        # Normaliser
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            for name in new_weights:
                new_weights[name] /= total_weight
        
        # Mise à jour avec lissage
        alpha = self.config['adaptation_speed']
        for name in self.strategy_weights:
            old_weight = self.strategy_weights[name]
            new_weight = new_weights.get(name, old_weight)
            self.strategy_weights[name] = old_weight * (1 - alpha) + new_weight * alpha
        
        self.last_rebalance = now
        self.logger.info(f"Nouveaux poids: {self.strategy_weights}")
    
    def _calculate_performance_weights(self) -> Dict[str, float]:
        """Calcule les poids basés sur la performance historique"""
        weights = {}
        
        # Collecter les métriques de performance
        performances = []
        for name, perf in self.strategy_performance.items():
            if perf.total_signals > 10:  # Minimum de signaux
                score = (
                    perf.sharpe_ratio * 0.4 +
                    perf.success_rate * 0.3 +
                    (1 - perf.max_drawdown) * 0.3
                )
                performances.append((name, max(0, score)))
            else:
                performances.append((name, 0.5))  # Score neutre
        
        # Convertir en poids
        total_score = sum(score for _, score in performances)
        if total_score > 0:
            for name, score in performances:
                weights[name] = score / total_score
        else:
            # Equal weight si pas de données
            for name, _ in performances:
                weights[name] = 1.0 / len(performances)
        
        return weights
    
    def _calculate_risk_parity_weights(self) -> Dict[str, float]:
        """Calcule les poids pour égaliser la contribution au risque"""
        weights = {}
        
        # Collecter les volatilités
        volatilities = {}
        for name, perf in self.strategy_performance.items():
            if perf.volatility > 0:
                volatilities[name] = perf.volatility
            else:
                volatilities[name] = 0.2  # Défaut 20%
        
        # Risk parity simple: poids inversement proportionnels à la volatilité
        total_inv_vol = sum(1/v for v in volatilities.values())
        
        for name, vol in volatilities.items():
            weights[name] = (1/vol) / total_inv_vol
        
        return weights
    
    def _calculate_regime_weights(self) -> Dict[str, float]:
        """Calcule les poids basés sur le régime de marché actuel"""
        weights = defaultdict(float)
        
        # Moyenne des scores de régime pour tous les symboles
        for symbol in self.symbols:
            if symbol in self.market_context:
                context = self.market_context[symbol]
                regime_scores = context.get_regime_scores()
                
                for name, score in regime_scores.items():
                    weights[name] += score / len(self.symbols)
        
        # Normaliser
        total = sum(weights.values())
        if total > 0:
            for name in weights:
                weights[name] /= total
        else:
            # Equal weight par défaut
            for name in self.config['enabled_strategies']:
                weights[name] = 1.0 / len(self.config['enabled_strategies'])
        
        return dict(weights)
    
    async def _calculate_ml_weights(self) -> Dict[str, float]:
        """Calcule les poids optimaux avec le machine learning"""
        if not self.performance_predictor:
            # Fallback to performance-based
            return self._calculate_performance_weights()
        
        # Préparer les features
        features = []
        
        # Features globales
        avg_volatility = np.mean([
            ctx.volatility_percentile 
            for ctx in self.market_context.values()
        ]) if self.market_context else 50.0
        
        avg_volume = np.mean([
            ctx.volume_percentile 
            for ctx in self.market_context.values()
        ]) if self.market_context else 50.0
        
        features.extend([avg_volatility, avg_volume])
        
        # Features par stratégie
        for name in self.config['enabled_strategies']:
            perf = self.strategy_performance[name]
            features.extend([
                perf.sharpe_ratio,
                perf.success_rate,
                perf.volatility,
                perf.max_drawdown
            ])
        
        # Prédire les poids optimaux
        try:
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            predicted_weights = self.performance_predictor.predict(features_scaled)[0]
            
            # Convertir en dictionnaire
            weights = {}
            for i, name in enumerate(self.config['enabled_strategies']):
                weights[name] = max(0, predicted_weights[i])
            
            # Normaliser
            total = sum(weights.values())
            if total > 0:
                for name in weights:
                    weights[name] /= total
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Erreur ML weights: {str(e)}")
            return self._calculate_performance_weights()
    
    def _record_signal(
        self, 
        signal: TradingSignal, 
        ensemble_signal: EnsembleSignal
    ) -> None:
        """Enregistre un signal pour l'apprentissage"""
        self.signal_history.append({
            'timestamp': datetime.now(timezone.utc),
            'signal': signal,
            'ensemble': ensemble_signal,
            'market_context': self.market_context.get(signal.symbol),
            'weights': dict(self.strategy_weights)
        })
        
        self.total_signals_generated += 1
        if ensemble_signal.consensus_score > 0.5:
            self.consensus_signals += 1
    
    def _extract_conflict_features(
        self,
        primary_signals: List[Tuple[str, TradingSignal]],
        conflicting_signals: List[Tuple[str, TradingSignal]],
        symbol: Symbol
    ) -> Optional[np.ndarray]:
        """Extrait les features pour la résolution de conflits ML"""
        try:
            features = []
            
            # Force relative des signaux
            primary_strength = np.mean([s.strength * s.confidence for _, s in primary_signals])
            conflict_strength = np.mean([s.strength * s.confidence for _, s in conflicting_signals])
            features.extend([primary_strength, conflict_strength])
            
            # Nombre de signaux de chaque côté
            features.extend([len(primary_signals), len(conflicting_signals)])
            
            # Performance historique des stratégies
            for signals in [primary_signals, conflicting_signals]:
                avg_sharpe = np.mean([
                    self.strategy_performance[name].sharpe_ratio 
                    for name, _ in signals
                ])
                features.append(avg_sharpe)
            
            # Contexte de marché
            if symbol in self.market_context:
                ctx = self.market_context[symbol]
                features.extend([
                    ctx.volatility_percentile / 100,
                    ctx.volume_percentile / 100,
                    ctx.trend_strength,
                    ctx.liquidity_score
                ])
            else:
                features.extend([0.5, 0.5, 0.0, 0.5])
            
            return np.array(features)
            
        except Exception as e:
            self.logger.error(f"Erreur extraction features: {str(e)}")
            return None
    
    async def calculate_indicators(self, symbol: Symbol) -> Dict[str, Any]:
        """Calcule les indicateurs de l'ensemble"""
        indicators = {}
        
        # Contexte de marché
        if symbol in self.market_context:
            ctx = self.market_context[symbol]
            indicators['market_regime'] = ctx.regime.value
            indicators['volatility_percentile'] = ctx.volatility_percentile
            indicators['trend_strength'] = ctx.trend_strength
            indicators['liquidity_score'] = ctx.liquidity_score
        
        # Poids des stratégies
        indicators['strategy_weights'] = dict(self.strategy_weights)
        
        # Métriques ensemble
        indicators['total_signals'] = self.total_signals_generated
        indicators['consensus_rate'] = (
            self.consensus_signals / self.total_signals_generated 
            if self.total_signals_generated > 0 else 0
        )
        
        # Performance par stratégie
        for name, perf in self.strategy_performance.items():
            indicators[f'{name}_sharpe'] = perf.sharpe_ratio
            indicators[f'{name}_success_rate'] = perf.success_rate
        
        return indicators
    
    def get_required_history_size(self) -> int:
        """Retourne la taille d'historique requise"""
        # Maximum des besoins des sous-stratégies
        return max(
            strategy.get_required_history_size() 
            for strategy in self.sub_strategies.values()
        )
    
    async def _on_initialize(self) -> None:
        """Initialisation de l'ensemble"""
        self.logger.info(f"Initialisation de la stratégie ensemble avec {len(self.sub_strategies)} sous-stratégies")
        
        # Initialiser toutes les sous-stratégies
        init_tasks = [
            strategy.initialize() 
            for strategy in self.sub_strategies.values()
        ]
        await asyncio.gather(*init_tasks)
        
        # Initialiser les modèles ML si configuré
        if self.config['use_ml_selection']:
            await self._initialize_ml_models()
    
    async def _initialize_ml_models(self) -> None:
        """Initialise les modèles de machine learning"""
        # Placeholder pour l'initialisation des modèles
        # En production, charger des modèles pré-entraînés
        
        # Modèle de sélection de stratégies
        self.strategy_selector = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Modèle de prédiction de performance
        self.performance_predictor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        self.logger.info("Modèles ML initialisés")
    
    async def _save_custom_state(self) -> Dict[str, Any]:
        """Sauvegarde l'état de l'ensemble"""
        # États des sous-stratégies
        sub_states = {}
        for name, strategy in self.sub_strategies.items():
            sub_states[name] = await strategy.save_state()
        
        return {
            'strategy_weights': dict(self.strategy_weights),
            'performance': {
                name: {
                    'total_signals': perf.total_signals,
                    'success_rate': perf.success_rate,
                    'sharpe_ratio': perf.sharpe_ratio
                }
                for name, perf in self.strategy_performance.items()
            },
            'sub_strategy_states': sub_states,
            'total_signals': self.total_signals_generated,
            'consensus_signals': self.consensus_signals
        }
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Diagnostics étendus de l'ensemble"""
        base_diagnostics = super().get_diagnostics()
        
        # Diagnostics des sous-stratégies
        sub_diagnostics = {}
        for name, strategy in self.sub_strategies.items():
            sub_diagnostics[name] = strategy.get_diagnostics()
        
        base_diagnostics.update({
            'allocation_method': self.config['allocation_method'].value,
            'strategy_weights': dict(self.strategy_weights),
            'sub_strategy_diagnostics': sub_diagnostics,
            'total_ensemble_signals': self.total_signals_generated,
            'consensus_rate': (
                self.consensus_signals / self.total_signals_generated 
                if self.total_signals_generated > 0 else 0
            ),
            'conflict_resolutions': dict(self.conflict_resolutions),
            'performance_summary': {
                name: {
                    'sharpe': perf.sharpe_ratio,
                    'success_rate': perf.success_rate,
                    'signals': perf.total_signals
                }
                for name, perf in self.strategy_performance.items()
            }
        })
        
        return base_diagnostics


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration de l'ensemble
    config = {
        'enabled_strategies': ['statistical_arbitrage', 'market_making', 'scalping'],
        'allocation_method': StrategyAllocation.ML_OPTIMIZED,
        'min_consensus': 0.3,
        'require_confirmation': True,
        'use_ml_selection': True,
        'rebalance_frequency': 3600  # 1 heure
    }
    
    # Symboles diversifiés
    symbols = [
        Symbol("BTC-USD"),
        Symbol("ETH-USD"),
        Symbol("EUR/USD"),
        Symbol("SPY"),
        Symbol("GLD")
    ]
    
    print("Configuration de la Stratégie Ensemble:")
    print(f"- Sous-stratégies: {config['enabled_strategies']}")
    print(f"- Méthode d'allocation: {config['allocation_method'].value}")
    print(f"- Consensus minimum: {config['min_consensus']}")
    print(f"- Confirmation requise: {config['require_confirmation']}")
    print(f"- Symboles: {len(symbols)}")