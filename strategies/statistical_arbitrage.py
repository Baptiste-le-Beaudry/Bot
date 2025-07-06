"""
Statistical Arbitrage Strategy - Stratégie d'arbitrage statistique avancée
Exploite les déviations temporaires des relations statistiques entre actifs

Caractéristiques principales:
- Détection de cointégration (Johansen, Engle-Granger)
- Calcul de Z-score pour les signaux d'entrée/sortie
- Machine learning pour la sélection de paires
- Gestion dynamique du risque et des positions
Performance cible: Sharpe ratio 1.5-3.0
"""

import asyncio
import numpy as np
import pandas as pd
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.johansen import coint_johansen
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from strategies.base_strategy import (
    BaseStrategy, TradingSignal, SignalType, MarketData,
    Symbol, StrategyId, SignalStrength, Confidence,
    Price, Quantity, MarketRegime
)


@dataclass
class PairStats:
    """Statistiques pour une paire d'actifs"""
    symbol1: Symbol
    symbol2: Symbol
    cointegration_pvalue: float
    half_life: float
    hurst_exponent: float
    spread_mean: float
    spread_std: float
    current_zscore: float
    correlation: float
    beta: float  # Hedge ratio
    last_updated: datetime
    ml_score: float = 0.0  # Score du modèle ML
    
    @property
    def is_tradeable(self) -> bool:
        """Vérifie si la paire est tradeable"""
        return (
            self.cointegration_pvalue < 0.05 and  # Cointégrée à 95%
            5 <= self.half_life <= 60 and         # Half-life raisonnable (5-60 périodes)
            self.hurst_exponent < 0.5 and         # Mean-reverting
            abs(self.correlation) > 0.7            # Forte corrélation
        )


@dataclass
class ActivePairTrade:
    """Trade actif sur une paire"""
    pair_stats: PairStats
    entry_zscore: float
    entry_time: datetime
    position_size_1: Quantity
    position_size_2: Quantity
    entry_price_1: Price
    entry_price_2: Price
    direction: str  # "long_spread" ou "short_spread"
    unrealized_pnl: Decimal = Decimal("0")
    max_profit: Decimal = Decimal("0")
    time_in_trade: timedelta = timedelta()
    
    def update_pnl(self, current_price_1: Price, current_price_2: Price) -> None:
        """Met à jour le P&L non réalisé"""
        if self.direction == "long_spread":
            # Long symbol1, short symbol2
            pnl_1 = (current_price_1 - self.entry_price_1) * self.position_size_1
            pnl_2 = (self.entry_price_2 - current_price_2) * self.position_size_2
        else:
            # Short symbol1, long symbol2
            pnl_1 = (self.entry_price_1 - current_price_1) * self.position_size_1
            pnl_2 = (current_price_2 - self.entry_price_2) * self.position_size_2
        
        self.unrealized_pnl = pnl_1 + pnl_2
        self.max_profit = max(self.max_profit, self.unrealized_pnl)
        self.time_in_trade = datetime.now(timezone.utc) - self.entry_time


class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Stratégie d'arbitrage statistique sophistiquée
    Identifie et trade les divergences dans les paires cointégrées
    """
    
    def __init__(
        self,
        strategy_id: StrategyId,
        symbols: List[Symbol],
        data_provider,
        risk_manager,
        config: Optional[Dict[str, Any]] = None
    ):
        # Configuration par défaut optimisée
        default_config = {
            'lookback_period': 200,          # Période pour les calculs de cointégration
            'entry_zscore': 2.0,             # Z-score pour entrer en position
            'exit_zscore': 0.0,              # Z-score pour sortir (retour à la moyenne)
            'stop_loss_zscore': 3.5,         # Z-score pour stop loss
            'max_pairs': 10,                 # Nombre max de paires à trader
            'min_correlation': 0.7,          # Corrélation minimale
            'max_half_life': 60,             # Half-life maximale en périodes
            'position_size_pct': 0.02,       # 2% du capital par paire
            'use_ml': True,                  # Utiliser ML pour la sélection
            'rebalance_frequency': 24,       # Heures entre recalculs
            'enable_dynamic_hedging': True,   # Ajustement dynamique du hedge ratio
            'max_pair_exposure': 0.05,       # Exposition max par paire (5%)
            'use_johansen': True,            # Utiliser le test de Johansen
            'ml_features': ['spread_vol', 'correlation', 'volume_ratio'],
            'buffer_size': 500,              # Taille du buffer de données
            'min_trades_per_day': 100        # Volume minimum pour considérer un symbole
        }
        
        if config:
            default_config.update(config)
        
        super().__init__(strategy_id, symbols, data_provider, risk_manager, default_config)
        
        # Structures de données spécifiques
        self.pair_stats: Dict[Tuple[Symbol, Symbol], PairStats] = {}
        self.active_trades: Dict[Tuple[Symbol, Symbol], ActivePairTrade] = {}
        self.price_series: Dict[Symbol, pd.Series] = {}
        
        # Machine Learning
        self.ml_model: Optional[RandomForestRegressor] = None
        self.scaler = StandardScaler()
        self.ml_training_data: deque = deque(maxlen=1000)
        
        # Métriques spécifiques
        self.pairs_analyzed = 0
        self.pairs_trading = 0
        self.last_rebalance = datetime.now(timezone.utc)
        
        # Cache pour optimisation
        self._spread_cache: Dict[Tuple[Symbol, Symbol], deque] = {}
        self._zscore_cache: Dict[Tuple[Symbol, Symbol], float] = {}
    
    async def analyze_market(self, symbol: Symbol) -> Optional[TradingSignal]:
        """
        Analyse principale - vérifie toutes les paires impliquant ce symbole
        """
        # Vérifier si nous avons assez de données
        if len(self.market_data_buffer[symbol]) < self.config['lookback_period']:
            return None
        
        # Mettre à jour les séries de prix
        await self._update_price_series(symbol)
        
        # Vérifier s'il est temps de recalculer les paires
        hours_since_rebalance = (datetime.now(timezone.utc) - self.last_rebalance).total_seconds() / 3600
        if hours_since_rebalance >= self.config['rebalance_frequency']:
            await self._rebalance_pairs()
        
        # Analyser toutes les paires actives impliquant ce symbole
        signals = []
        for (sym1, sym2), pair_stats in self.pair_stats.items():
            if sym1 == symbol or sym2 == symbol:
                signal = await self._analyze_pair(pair_stats)
                if signal:
                    signals.append(signal)
        
        # Retourner le signal le plus fort
        if signals:
            return max(signals, key=lambda s: s.strength * s.confidence)
        
        return None
    
    async def _analyze_pair(self, pair_stats: PairStats) -> Optional[TradingSignal]:
        """Analyse une paire spécifique pour des opportunités"""
        if not pair_stats.is_tradeable:
            return None
        
        # Calculer le Z-score actuel
        current_zscore = await self._calculate_current_zscore(pair_stats)
        pair_stats.current_zscore = current_zscore
        
        # Vérifier si nous avons déjà une position sur cette paire
        pair_key = (pair_stats.symbol1, pair_stats.symbol2)
        
        if pair_key in self.active_trades:
            # Gérer la position existante
            return await self._manage_existing_position(pair_stats, current_zscore)
        else:
            # Chercher une nouvelle entrée
            return await self._check_entry_signal(pair_stats, current_zscore)
    
    async def _check_entry_signal(self, pair_stats: PairStats, current_zscore: float) -> Optional[TradingSignal]:
        """Vérifie les conditions d'entrée pour une paire"""
        entry_threshold = self.config['entry_zscore']
        
        # Signal d'achat du spread (long symbol1, short symbol2)
        if current_zscore <= -entry_threshold:
            # Le spread est trop bas, on s'attend à ce qu'il remonte
            signal_strength = min(abs(current_zscore) / 3.0, 1.0)  # Normaliser entre 0 et 1
            confidence = self._calculate_signal_confidence(pair_stats, current_zscore)
            
            return TradingSignal(
                strategy_id=self.strategy_id,
                symbol=pair_stats.symbol1,  # Symbol principal
                signal_type=SignalType.BUY,
                strength=SignalStrength(signal_strength),
                confidence=Confidence(confidence),
                metadata={
                    'pair': (pair_stats.symbol1, pair_stats.symbol2),
                    'direction': 'long_spread',
                    'zscore': current_zscore,
                    'hedge_ratio': pair_stats.beta,
                    'half_life': pair_stats.half_life,
                    'pair_action': {
                        pair_stats.symbol1: 'BUY',
                        pair_stats.symbol2: 'SELL'
                    }
                }
            )
        
        # Signal de vente du spread (short symbol1, long symbol2)
        elif current_zscore >= entry_threshold:
            # Le spread est trop haut, on s'attend à ce qu'il baisse
            signal_strength = min(current_zscore / 3.0, 1.0)
            confidence = self._calculate_signal_confidence(pair_stats, current_zscore)
            
            return TradingSignal(
                strategy_id=self.strategy_id,
                symbol=pair_stats.symbol1,
                signal_type=SignalType.SELL,
                strength=SignalStrength(signal_strength),
                confidence=Confidence(confidence),
                metadata={
                    'pair': (pair_stats.symbol1, pair_stats.symbol2),
                    'direction': 'short_spread',
                    'zscore': current_zscore,
                    'hedge_ratio': pair_stats.beta,
                    'half_life': pair_stats.half_life,
                    'pair_action': {
                        pair_stats.symbol1: 'SELL',
                        pair_stats.symbol2: 'BUY'
                    }
                }
            )
        
        return None
    
    async def _manage_existing_position(
        self, 
        pair_stats: PairStats, 
        current_zscore: float
    ) -> Optional[TradingSignal]:
        """Gère une position existante"""
        pair_key = (pair_stats.symbol1, pair_stats.symbol2)
        trade = self.active_trades[pair_key]
        
        # Mettre à jour le P&L
        latest_price_1 = self.get_latest_data(pair_stats.symbol1).price
        latest_price_2 = self.get_latest_data(pair_stats.symbol2).price
        trade.update_pnl(latest_price_1, latest_price_2)
        
        # Conditions de sortie
        exit_zscore = self.config['exit_zscore']
        stop_loss_zscore = self.config['stop_loss_zscore']
        
        should_exit = False
        exit_reason = ""
        
        # 1. Retour à la moyenne (take profit)
        if trade.direction == "long_spread" and current_zscore >= exit_zscore:
            should_exit = True
            exit_reason = "mean_reversion"
        elif trade.direction == "short_spread" and current_zscore <= exit_zscore:
            should_exit = True
            exit_reason = "mean_reversion"
        
        # 2. Stop loss si le spread continue dans la mauvaise direction
        elif abs(current_zscore) >= stop_loss_zscore:
            should_exit = True
            exit_reason = "stop_loss"
        
        # 3. Temps maximum dans le trade (basé sur half-life)
        elif trade.time_in_trade.total_seconds() / 3600 > pair_stats.half_life * 2:
            should_exit = True
            exit_reason = "time_exit"
        
        # 4. Trailing stop si nous avons un profit significatif
        elif trade.max_profit > Decimal("0") and trade.unrealized_pnl < trade.max_profit * Decimal("0.5"):
            should_exit = True
            exit_reason = "trailing_stop"
        
        if should_exit:
            # Générer un signal de fermeture
            return TradingSignal(
                strategy_id=self.strategy_id,
                symbol=pair_stats.symbol1,
                signal_type=SignalType.CLOSE_LONG if trade.direction == "long_spread" else SignalType.CLOSE_SHORT,
                strength=SignalStrength(0.9),  # Forte conviction pour les sorties
                confidence=Confidence(0.95),
                metadata={
                    'pair': pair_key,
                    'exit_reason': exit_reason,
                    'zscore': current_zscore,
                    'pnl': float(trade.unrealized_pnl),
                    'time_in_trade': str(trade.time_in_trade),
                    'pair_action': {
                        pair_stats.symbol1: 'CLOSE',
                        pair_stats.symbol2: 'CLOSE'
                    }
                }
            )
        
        # Possibilité d'ajuster la position (scaling)
        elif self.config.get('enable_dynamic_hedging', False):
            return await self._check_position_adjustment(pair_stats, trade, current_zscore)
        
        return None
    
    async def _calculate_current_zscore(self, pair_stats: PairStats) -> float:
        """Calcule le Z-score actuel du spread"""
        # Obtenir les prix actuels
        price1 = float(self.get_latest_data(pair_stats.symbol1).price)
        price2 = float(self.get_latest_data(pair_stats.symbol2).price)
        
        # Calculer le spread actuel
        spread = price1 - pair_stats.beta * price2
        
        # Calculer le Z-score
        zscore = (spread - pair_stats.spread_mean) / pair_stats.spread_std if pair_stats.spread_std > 0 else 0
        
        # Mettre en cache
        self._zscore_cache[(pair_stats.symbol1, pair_stats.symbol2)] = zscore
        
        return zscore
    
    def _calculate_signal_confidence(self, pair_stats: PairStats, zscore: float) -> float:
        """Calcule la confiance dans le signal"""
        confidence = 0.5  # Base
        
        # Facteurs augmentant la confiance
        if pair_stats.cointegration_pvalue < 0.01:  # Très forte cointégration
            confidence += 0.15
        
        if 10 <= pair_stats.half_life <= 30:  # Half-life idéale
            confidence += 0.1
        
        if abs(pair_stats.correlation) > 0.85:  # Très forte corrélation
            confidence += 0.1
        
        if pair_stats.ml_score > 0.7:  # Le ML est confiant
            confidence += 0.15
        
        # Ajuster selon le régime de marché
        regime = self.detect_market_regime(pair_stats.symbol1)
        if regime == MarketRegime.MEAN_REVERTING:
            confidence += 0.1
        elif regime in [MarketRegime.HIGH_VOLATILITY, MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            confidence -= 0.1
        
        return min(confidence, 0.95)  # Plafonner à 95%
    
    async def _rebalance_pairs(self) -> None:
        """Recalcule toutes les paires et leurs statistiques"""
        self.logger.info("Début du rebalancing des paires")
        self.last_rebalance = datetime.now(timezone.utc)
        
        # Créer toutes les combinaisons de paires possibles
        all_pairs = []
        for i, sym1 in enumerate(self.symbols):
            for sym2 in self.symbols[i+1:]:
                if self._is_valid_pair(sym1, sym2):
                    all_pairs.append((sym1, sym2))
        
        # Analyser chaque paire
        pair_scores = []
        for sym1, sym2 in all_pairs:
            stats = await self._analyze_pair_statistics(sym1, sym2)
            if stats and stats.is_tradeable:
                # Calculer un score composite
                score = self._calculate_pair_score(stats)
                pair_scores.append((score, stats))
        
        # Sélectionner les meilleures paires
        pair_scores.sort(key=lambda x: x[0], reverse=True)
        self.pair_stats.clear()
        
        for score, stats in pair_scores[:self.config['max_pairs']]:
            self.pair_stats[(stats.symbol1, stats.symbol2)] = stats
            self.logger.info(f"Paire sélectionnée: {stats.symbol1}/{stats.symbol2} "
                           f"(score: {score:.3f}, p-value: {stats.cointegration_pvalue:.4f}, "
                           f"half-life: {stats.half_life:.1f})")
        
        self.pairs_analyzed = len(all_pairs)
        self.pairs_trading = len(self.pair_stats)
        
        # Entraîner/mettre à jour le modèle ML si activé
        if self.config['use_ml'] and len(self.ml_training_data) > 100:
            await self._train_ml_model()
    
    async def _analyze_pair_statistics(self, symbol1: Symbol, symbol2: Symbol) -> Optional[PairStats]:
        """Analyse complète des statistiques d'une paire"""
        try:
            # Obtenir les séries de prix
            if symbol1 not in self.price_series or symbol2 not in self.price_series:
                return None
            
            prices1 = self.price_series[symbol1]
            prices2 = self.price_series[symbol2]
            
            if len(prices1) < self.config['lookback_period'] or len(prices2) < self.config['lookback_period']:
                return None
            
            # Aligner les séries
            prices1 = prices1[-self.config['lookback_period']:]
            prices2 = prices2[-self.config['lookback_period']:]
            
            # Test de cointégration d'Engle-Granger
            coint_result = coint(prices1, prices2)
            p_value = coint_result[1]
            
            # Si pas cointégré selon Engle-Granger, essayer Johansen si activé
            if p_value > 0.05 and self.config.get('use_johansen', False):
                p_value = self._johansen_test(prices1, prices2)
            
            # Calculer le hedge ratio (beta) par régression
            model = sm.OLS(prices1, sm.add_constant(prices2))
            results = model.fit()
            beta = results.params[1]
            
            # Calculer le spread
            spread = prices1 - beta * prices2
            spread_mean = float(spread.mean())
            spread_std = float(spread.std())
            
            # Calculer la half-life du spread
            half_life = self._calculate_half_life(spread)
            
            # Calculer l'exposant de Hurst
            hurst = self._calculate_hurst_exponent(spread)
            
            # Corrélation
            correlation = float(prices1.corr(prices2))
            
            # Z-score actuel
            current_spread = float(prices1.iloc[-1] - beta * prices2.iloc[-1])
            current_zscore = (current_spread - spread_mean) / spread_std if spread_std > 0 else 0
            
            return PairStats(
                symbol1=symbol1,
                symbol2=symbol2,
                cointegration_pvalue=p_value,
                half_life=half_life,
                hurst_exponent=hurst,
                spread_mean=spread_mean,
                spread_std=spread_std,
                current_zscore=current_zscore,
                correlation=correlation,
                beta=beta,
                last_updated=datetime.now(timezone.utc)
            )
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse de {symbol1}/{symbol2}: {str(e)}")
            return None
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calcule la half-life du spread (temps de retour à la moyenne)"""
        try:
            # Modèle AR(1) pour estimer la vitesse de mean reversion
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            
            # Aligner les séries
            spread_lag = spread_lag[spread_diff.index]
            
            # Régression pour obtenir le coefficient de mean reversion
            model = sm.OLS(spread_diff, spread_lag)
            results = model.fit()
            
            # Calculer la half-life
            if results.params[0] < 0:
                half_life = -np.log(2) / results.params[0]
                return float(half_life) if half_life > 0 else 999
            else:
                return 999  # Pas de mean reversion
                
        except:
            return 999
    
    def _calculate_hurst_exponent(self, series: pd.Series, max_lag: int = 20) -> float:
        """Calcule l'exposant de Hurst (< 0.5 = mean reverting)"""
        try:
            # Convertir en array numpy
            ts = series.values
            
            # Calculer les écarts types pour différents lags
            lags = range(2, min(max_lag, len(ts) // 2))
            tau = []
            
            for lag in lags:
                # Calculer les différences
                differences = ts[lag:] - ts[:-lag]
                # Calculer l'écart type
                tau.append(np.std(differences))
            
            # Régression log-log
            if len(tau) > 0:
                reg = np.polyfit(np.log(list(lags)), np.log(tau), 1)
                hurst = reg[0]
                return float(hurst)
            else:
                return 0.5
                
        except:
            return 0.5
    
    def _johansen_test(self, prices1: pd.Series, prices2: pd.Series) -> float:
        """Test de cointégration de Johansen"""
        try:
            # Préparer les données
            data = pd.DataFrame({'y1': prices1, 'y2': prices2})
            
            # Test de Johansen
            result = coint_johansen(data, det_order=0, k_ar_diff=1)
            
            # Utiliser la statistique de trace
            trace_stat = result.lr1[0]
            critical_value = result.cvt[0, 1]  # Valeur critique à 95%
            
            # Convertir en p-value approximative
            if trace_stat > critical_value:
                return 0.01  # Forte évidence de cointégration
            else:
                return 0.10  # Pas de cointégration
                
        except:
            return 1.0
    
    def _calculate_pair_score(self, stats: PairStats) -> float:
        """Calcule un score composite pour classer les paires"""
        score = 0.0
        
        # Pénaliser les p-values élevées
        score += (1 - stats.cointegration_pvalue) * 30
        
        # Favoriser les half-lives moyennes
        if 10 <= stats.half_life <= 30:
            score += 20
        elif 5 <= stats.half_life <= 60:
            score += 10
        
        # Favoriser les exposants de Hurst bas (mean reverting)
        score += (0.5 - stats.hurst_exponent) * 40
        
        # Favoriser les fortes corrélations
        score += abs(stats.correlation) * 20
        
        # Ajouter le score ML si disponible
        score += stats.ml_score * 30
        
        # Pénaliser les Z-scores extrêmes actuels (déjà étendus)
        score -= min(abs(stats.current_zscore), 3) * 5
        
        return score
    
    def _is_valid_pair(self, sym1: Symbol, sym2: Symbol) -> bool:
        """Vérifie si une paire est valide pour l'analyse"""
        # Éviter les paires identiques
        if sym1 == sym2:
            return False
        
        # Vérifier le volume minimum
        data1 = self.get_latest_data(sym1)
        data2 = self.get_latest_data(sym2)
        
        if data1 and data2:
            min_volume = self.config.get('min_trades_per_day', 100)
            # Estimation simplifiée du volume journalier
            if data1.volume < min_volume or data2.volume < min_volume:
                return False
        
        # Autres critères possibles (même secteur, même devise de base, etc.)
        # À personnaliser selon les besoins
        
        return True
    
    async def _update_price_series(self, symbol: Symbol) -> None:
        """Met à jour les séries de prix pour un symbole"""
        # Extraire les prix du buffer
        prices = []
        for data in self.market_data_buffer[symbol]:
            prices.append(float(data.price))
        
        if len(prices) >= self.config['lookback_period']:
            self.price_series[symbol] = pd.Series(prices[-self.config['lookback_period']:])
    
    async def _train_ml_model(self) -> None:
        """Entraîne le modèle ML pour la sélection de paires"""
        if len(self.ml_training_data) < 100:
            return
        
        try:
            # Préparer les données d'entraînement
            X = []
            y = []
            
            for data in self.ml_training_data:
                features = data['features']
                target = data['target']  # Rendement de la paire
                X.append(features)
                y.append(target)
            
            X = np.array(X)
            y = np.array(y)
            
            # Normaliser les features
            X_scaled = self.scaler.fit_transform(X)
            
            # Entraîner le modèle
            self.ml_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.ml_model.fit(X_scaled, y)
            
            # Mettre à jour les scores ML des paires actuelles
            for pair_stats in self.pair_stats.values():
                features = self._extract_ml_features(pair_stats)
                if features is not None:
                    features_scaled = self.scaler.transform([features])
                    pair_stats.ml_score = float(self.ml_model.predict(features_scaled)[0])
            
            self.logger.info("Modèle ML mis à jour avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement ML: {str(e)}")
    
    def _extract_ml_features(self, pair_stats: PairStats) -> Optional[np.ndarray]:
        """Extrait les features ML d'une paire"""
        try:
            features = [
                pair_stats.cointegration_pvalue,
                pair_stats.half_life,
                pair_stats.hurst_exponent,
                pair_stats.spread_std,
                abs(pair_stats.correlation),
                abs(pair_stats.current_zscore),
                pair_stats.beta
            ]
            
            # Ajouter des features supplémentaires si configurées
            # Volume ratio, volatilité, etc.
            
            return np.array(features)
            
        except:
            return None
    
    async def _check_position_adjustment(
        self,
        pair_stats: PairStats,
        trade: ActivePairTrade,
        current_zscore: float
    ) -> Optional[TradingSignal]:
        """Vérifie si la position doit être ajustée (scaling in/out)"""
        # Logique de scaling basée sur le Z-score et le P&L
        
        # Scaling out si on approche de la cible
        if abs(current_zscore) < 0.5 and trade.unrealized_pnl > Decimal("0"):
            return TradingSignal(
                strategy_id=self.strategy_id,
                symbol=pair_stats.symbol1,
                signal_type=SignalType.SCALE_OUT,
                strength=SignalStrength(0.5),
                confidence=Confidence(0.8),
                metadata={
                    'pair': (pair_stats.symbol1, pair_stats.symbol2),
                    'scale_factor': 0.5,  # Réduire de 50%
                    'reason': 'approaching_target'
                }
            )
        
        # Scaling in si l'opportunité s'améliore (dans la limite du risque)
        elif abs(current_zscore) > abs(trade.entry_zscore) * 1.2:
            if abs(current_zscore) < self.config['stop_loss_zscore']:
                return TradingSignal(
                    strategy_id=self.strategy_id,
                    symbol=pair_stats.symbol1,
                    signal_type=SignalType.SCALE_IN,
                    strength=SignalStrength(0.3),
                    confidence=Confidence(0.7),
                    metadata={
                        'pair': (pair_stats.symbol1, pair_stats.symbol2),
                        'scale_factor': 0.3,  # Ajouter 30%
                        'reason': 'better_entry'
                    }
                )
        
        return None
    
    async def calculate_indicators(self, symbol: Symbol) -> Dict[str, Any]:
        """Calcule les indicateurs spécifiques à l'arbitrage statistique"""
        indicators = {}
        
        # Indicateurs pour chaque paire impliquant ce symbole
        for (sym1, sym2), stats in self.pair_stats.items():
            if sym1 == symbol or sym2 == symbol:
                pair_key = f"{sym1}_{sym2}"
                indicators[f"{pair_key}_zscore"] = stats.current_zscore
                indicators[f"{pair_key}_half_life"] = stats.half_life
                indicators[f"{pair_key}_correlation"] = stats.correlation
                indicators[f"{pair_key}_ml_score"] = stats.ml_score
        
        # Métriques globales
        indicators['active_pairs'] = len(self.pair_stats)
        indicators['active_trades'] = len(self.active_trades)
        
        return indicators
    
    def get_required_history_size(self) -> int:
        """Retourne la taille d'historique requise"""
        return self.config['lookback_period'] + 50  # Marge de sécurité
    
    async def _on_initialize(self) -> None:
        """Hook d'initialisation spécifique"""
        self.logger.info(f"Initialisation de l'arbitrage statistique avec {len(self.symbols)} symboles")
        
        # Initialiser les caches
        for i, sym1 in enumerate(self.symbols):
            for sym2 in self.symbols[i+1:]:
                self._spread_cache[(sym1, sym2)] = deque(maxlen=self.config['lookback_period'])
    
    async def _save_custom_state(self) -> Dict[str, Any]:
        """Sauvegarde l'état spécifique à la stratégie"""
        return {
            'pair_stats': {
                f"{s1}_{s2}": {
                    'p_value': stats.cointegration_pvalue,
                    'half_life': stats.half_life,
                    'beta': stats.beta,
                    'ml_score': stats.ml_score
                }
                for (s1, s2), stats in self.pair_stats.items()
            },
            'active_trades': {
                f"{s1}_{s2}": {
                    'entry_zscore': trade.entry_zscore,
                    'direction': trade.direction,
                    'unrealized_pnl': float(trade.unrealized_pnl)
                }
                for (s1, s2), trade in self.active_trades.items()
            }
        }
    
    async def _load_custom_state(self, state: Dict[str, Any]) -> None:
        """Charge l'état spécifique sauvegardé"""
        # Restaurer les statistiques de paires
        # Note: nécessiterait une reconstruction complète des objets PairStats
        self.logger.info("État custom chargé")
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Retourne des diagnostics étendus"""
        base_diagnostics = super().get_diagnostics()
        
        base_diagnostics.update({
            'pairs_analyzed': self.pairs_analyzed,
            'pairs_trading': self.pairs_trading,
            'active_trades': len(self.active_trades),
            'total_pair_stats': len(self.pair_stats),
            'ml_model_trained': self.ml_model is not None,
            'top_pairs': [
                {
                    'pair': f"{s1}/{s2}",
                    'p_value': stats.cointegration_pvalue,
                    'zscore': stats.current_zscore,
                    'half_life': stats.half_life
                }
                for (s1, s2), stats in list(self.pair_stats.items())[:5]
            ]
        })
        
        return base_diagnostics


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration pour test
    config = {
        'lookback_period': 200,
        'entry_zscore': 2.0,
        'exit_zscore': 0.0,
        'stop_loss_zscore': 3.5,
        'max_pairs': 5,
        'use_ml': True
    }
    
    # Symboles pour l'arbitrage (ex: paires crypto corrélées)
    symbols = [
        Symbol("BTC-USD"),
        Symbol("ETH-USD"),
        Symbol("BNB-USD"),
        Symbol("SOL-USD"),
        Symbol("ADA-USD")
    ]
    
    print("Configuration de la stratégie d'arbitrage statistique:")
    print(f"- Symboles: {symbols}")
    print(f"- Lookback: {config['lookback_period']} périodes")
    print(f"- Z-score entrée: ±{config['entry_zscore']}")
    print(f"- Max paires: {config['max_pairs']}")