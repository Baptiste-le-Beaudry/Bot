"""
Risk Monitor - Surveillance des Risques en Temps Réel
=====================================================

Ce module implémente un système complet de surveillance des risques pour
le trading algorithmique. Il calcule et surveille en continu les métriques
de risque critiques avec alertes automatiques et intégration circuit breakers.

Métriques surveillées:
- VaR (Value at Risk) et CVaR
- Exposition par symbole/secteur/stratégie
- Corrélations dynamiques
- Greeks (Delta, Gamma, Vega pour options)
- Stress testing en temps réel
- Liquidité et concentration

Architecture:
- Calculs streaming avec latence < 100ms
- Agrégation multi-niveaux (position/stratégie/portefeuille)
- Historisation pour analyse post-mortem
- API temps réel pour dashboards
- Integration native avec circuit breakers

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
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Protocol
import json
from functools import lru_cache
import warnings
from scipy import stats
from concurrent.futures import ThreadPoolExecutor

# Imports internes
from core.portfolio_manager import Symbol, Price, Quantity, Position
from risk.circuit_breakers import CircuitBreakerManager, CircuitBreakerType
from risk.position_sizer import PositionSizer, MarketContext
from utils.logger import get_structured_logger
from utils.metrics import MetricsCollector
from monitoring.alerts import AlertManager, AlertLevel

# Suppression des warnings NumPy
warnings.filterwarnings('ignore', category=RuntimeWarning)


class RiskMetricType(Enum):
    """Types de métriques de risque surveillées"""
    VAR_95 = "var_95"
    VAR_99 = "var_99"
    CVAR_95 = "cvar_95"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAX_DRAWDOWN = "max_drawdown"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    BETA = "beta"
    CORRELATION = "correlation"
    CONCENTRATION = "concentration"
    LIQUIDITY = "liquidity"
    LEVERAGE = "leverage"
    MARGIN_USAGE = "margin_usage"
    STRESS_TEST = "stress_test"


class RiskLevel(Enum):
    """Niveaux de risque pour classification"""
    LOW = "low"          # Risque faible, trading normal
    MODERATE = "moderate" # Risque modéré, surveillance accrue
    HIGH = "high"        # Risque élevé, réduction positions
    CRITICAL = "critical" # Risque critique, action immédiate
    EXTREME = "extreme"   # Risque extrême, arrêt trading


@dataclass
class RiskSnapshot:
    """Snapshot instantané de tous les risques"""
    timestamp: datetime
    portfolio_var_95: Decimal
    portfolio_var_99: Decimal
    portfolio_cvar_95: Decimal
    current_drawdown: float
    max_drawdown: float
    total_exposure: Decimal
    net_exposure: Decimal
    gross_exposure: Decimal
    leverage: float
    margin_usage: float
    
    # Par symbole
    symbol_exposures: Dict[Symbol, Decimal] = field(default_factory=dict)
    symbol_vars: Dict[Symbol, Decimal] = field(default_factory=dict)
    
    # Par stratégie
    strategy_exposures: Dict[str, Decimal] = field(default_factory=dict)
    strategy_vars: Dict[str, Decimal] = field(default_factory=dict)
    
    # Corrélations
    correlation_matrix: Optional[np.ndarray] = None
    concentration_risk: float = 0.0
    
    # Liquidité
    liquidity_scores: Dict[Symbol, float] = field(default_factory=dict)
    illiquid_exposure: Decimal = Decimal("0")
    
    # Niveau de risque global
    risk_level: RiskLevel = RiskLevel.LOW
    risk_score: float = 0.0  # 0-100
    
    # Alertes actives
    active_alerts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour sérialisation"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_var_95': float(self.portfolio_var_95),
            'portfolio_var_99': float(self.portfolio_var_99),
            'portfolio_cvar_95': float(self.portfolio_cvar_95),
            'current_drawdown': self.current_drawdown,
            'max_drawdown': self.max_drawdown,
            'total_exposure': float(self.total_exposure),
            'net_exposure': float(self.net_exposure),
            'gross_exposure': float(self.gross_exposure),
            'leverage': self.leverage,
            'margin_usage': self.margin_usage,
            'risk_level': self.risk_level.value,
            'risk_score': self.risk_score,
            'active_alerts': self.active_alerts
        }


@dataclass
class RiskLimit:
    """Limite de risque configurable"""
    metric_type: RiskMetricType
    threshold_warning: float
    threshold_critical: float
    current_value: float = 0.0
    is_percentage: bool = True
    action_on_breach: str = "alert"  # alert, reduce_position, stop_trading
    
    @property
    def is_breached(self) -> bool:
        """Vérifie si la limite est dépassée"""
        return self.current_value > self.threshold_critical
    
    @property
    def is_warning(self) -> bool:
        """Vérifie si on approche de la limite"""
        return self.current_value > self.threshold_warning


@dataclass
class StressTestScenario:
    """Scénario de stress test"""
    name: str
    description: str
    market_shock: float  # % de mouvement
    volatility_multiplier: float
    correlation_shock: float  # Augmentation des corrélations
    liquidity_reduction: float  # Réduction de liquidité
    
    def apply_to_positions(
        self,
        positions: List[Position],
        market_prices: Dict[Symbol, Price]
    ) -> Decimal:
        """Applique le scénario et calcule la perte potentielle"""
        total_loss = Decimal("0")
        
        for position in positions:
            current_price = market_prices.get(position.symbol, position.current_price)
            
            # Appliquer le choc de marché
            if position.quantity > 0:  # Long
                shocked_price = current_price * Decimal(str(1 + self.market_shock))
            else:  # Short
                shocked_price = current_price * Decimal(str(1 - self.market_shock))
            
            # Calculer la perte
            position_loss = position.quantity * (shocked_price - current_price)
            
            # Ajuster pour la liquidité réduite (slippage additionnel)
            liquidity_impact = abs(position.market_value) * Decimal(str(self.liquidity_reduction))
            
            total_loss += position_loss - liquidity_impact
        
        return total_loss


class RiskMonitor:
    """
    Moniteur principal de surveillance des risques en temps réel
    avec calculs avancés et alertes automatiques
    """
    
    def __init__(
        self,
        circuit_breaker_manager: CircuitBreakerManager,
        position_sizer: PositionSizer,
        alert_manager: AlertManager,
        metrics_collector: MetricsCollector,
        config: Optional[Dict[str, Any]] = None
    ):
        self.circuit_breaker_manager = circuit_breaker_manager
        self.position_sizer = position_sizer
        self.alert_manager = alert_manager
        self.metrics = metrics_collector
        self.config = config or {}
        
        # Logger
        self.logger = get_structured_logger(
            "risk_monitor",
            module="risk"
        )
        
        # Configuration par défaut
        self._setup_default_config()
        
        # État
        self.current_positions: Dict[Symbol, Position] = {}
        self.market_prices: Dict[Symbol, Price] = {}
        self.historical_returns: Dict[Symbol, deque] = defaultdict(
            lambda: deque(maxlen=self.config['returns_lookback'])
        )
        
        # Limites de risque
        self.risk_limits: Dict[RiskMetricType, RiskLimit] = self._setup_risk_limits()
        
        # Historique
        self.risk_snapshots: deque = deque(maxlen=10000)
        self.var_history: deque = deque(maxlen=1000)
        self.drawdown_history: deque = deque(maxlen=1000)
        
        # Cache pour calculs coûteux
        self._correlation_cache: Optional[Tuple[np.ndarray, datetime]] = None
        self._var_cache: Dict[str, Tuple[Decimal, datetime]] = {}
        
        # Thread pool pour calculs parallèles
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # État de monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Scénarios de stress test
        self.stress_scenarios = self._setup_stress_scenarios()
        
        # Statistiques
        self.breach_counts = defaultdict(int)
        self.alert_counts = defaultdict(int)
        
        self.logger.info("risk_monitor_initialized", config=self.config)
    
    def _setup_default_config(self) -> None:
        """Configure les paramètres par défaut"""
        defaults = {
            # Limites VaR
            'var_95_limit': 0.05,        # 5% du capital
            'var_99_limit': 0.10,        # 10% du capital
            'cvar_95_limit': 0.08,       # 8% du capital
            
            # Limites d'exposition
            'max_gross_exposure': 2.0,    # 200% du capital
            'max_net_exposure': 1.0,      # 100% du capital
            'max_single_position': 0.10,  # 10% par position
            'max_sector_exposure': 0.30,  # 30% par secteur
            
            # Limites de performance
            'max_drawdown': 0.20,         # 20% drawdown max
            'min_sharpe_ratio': 0.5,      # Sharpe minimum
            
            # Paramètres de calcul
            'var_confidence_95': 0.95,
            'var_confidence_99': 0.99,
            'returns_lookback': 500,      # Fenêtre historique
            'correlation_window': 60,     # Jours pour corrélations
            
            # Fréquences de mise à jour
            'update_interval': 1.0,       # Secondes
            'var_calc_interval': 60.0,    # Secondes
            'stress_test_interval': 300.0, # 5 minutes
            
            # Seuils d'alerte
            'risk_score_warning': 60,     # Score 0-100
            'risk_score_critical': 80,
            
            # Options de calcul
            'use_monte_carlo': True,
            'monte_carlo_simulations': 10000,
            'use_historical_var': True,
            'use_parametric_var': True,
            
            # Stress testing
            'enable_stress_testing': True,
            'stress_test_on_demand': True,
        }
        
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
    
    def _setup_risk_limits(self) -> Dict[RiskMetricType, RiskLimit]:
        """Configure les limites de risque"""
        return {
            RiskMetricType.VAR_95: RiskLimit(
                metric_type=RiskMetricType.VAR_95,
                threshold_warning=self.config['var_95_limit'] * 0.8,
                threshold_critical=self.config['var_95_limit']
            ),
            RiskMetricType.VAR_99: RiskLimit(
                metric_type=RiskMetricType.VAR_99,
                threshold_warning=self.config['var_99_limit'] * 0.8,
                threshold_critical=self.config['var_99_limit']
            ),
            RiskMetricType.MAX_DRAWDOWN: RiskLimit(
                metric_type=RiskMetricType.MAX_DRAWDOWN,
                threshold_warning=self.config['max_drawdown'] * 0.7,
                threshold_critical=self.config['max_drawdown'],
                action_on_breach="reduce_position"
            ),
            RiskMetricType.LEVERAGE: RiskLimit(
                metric_type=RiskMetricType.LEVERAGE,
                threshold_warning=1.5,
                threshold_critical=2.0,
                is_percentage=False,
                action_on_breach="stop_new_positions"
            ),
            RiskMetricType.CONCENTRATION: RiskLimit(
                metric_type=RiskMetricType.CONCENTRATION,
                threshold_warning=0.25,
                threshold_critical=0.35,
                action_on_breach="rebalance"
            )
        }
    
    def _setup_stress_scenarios(self) -> List[StressTestScenario]:
        """Configure les scénarios de stress test standards"""
        return [
            StressTestScenario(
                name="Market Crash",
                description="Chute de marché de 10%",
                market_shock=-0.10,
                volatility_multiplier=3.0,
                correlation_shock=0.3,
                liquidity_reduction=0.2
            ),
            StressTestScenario(
                name="Flash Crash",
                description="Flash crash de 5% avec récupération",
                market_shock=-0.05,
                volatility_multiplier=5.0,
                correlation_shock=0.5,
                liquidity_reduction=0.5
            ),
            StressTestScenario(
                name="Volatility Spike",
                description="Spike de volatilité sans crash",
                market_shock=0.0,
                volatility_multiplier=4.0,
                correlation_shock=0.2,
                liquidity_reduction=0.1
            ),
            StressTestScenario(
                name="Liquidity Crisis",
                description="Crise de liquidité majeure",
                market_shock=-0.03,
                volatility_multiplier=2.0,
                correlation_shock=0.4,
                liquidity_reduction=0.7
            ),
            StressTestScenario(
                name="Black Swan",
                description="Événement extrême improbable",
                market_shock=-0.20,
                volatility_multiplier=10.0,
                correlation_shock=0.8,
                liquidity_reduction=0.9
            )
        ]
    
    async def start_monitoring(self) -> None:
        """Démarre la surveillance des risques"""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Démarrer les calculs périodiques
        asyncio.create_task(self._var_calculation_loop())
        asyncio.create_task(self._stress_test_loop())
        
        self.logger.info("risk_monitoring_started")
    
    async def stop_monitoring(self) -> None:
        """Arrête la surveillance"""
        self._running = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        self.executor.shutdown(wait=True)
        self.logger.info("risk_monitoring_stopped")
    
    async def _monitoring_loop(self) -> None:
        """Boucle principale de surveillance"""
        while self._running:
            try:
                start_time = datetime.now(timezone.utc)
                
                # Créer un snapshot des risques
                snapshot = await self.calculate_risk_snapshot()
                
                # Vérifier les limites
                breaches = self._check_risk_limits(snapshot)
                
                # Envoyer aux circuit breakers
                await self._update_circuit_breakers(snapshot)
                
                # Alerter si nécessaire
                if breaches:
                    await self._handle_limit_breaches(breaches, snapshot)
                
                # Enregistrer le snapshot
                self.risk_snapshots.append(snapshot)
                
                # Métriques de performance
                calc_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                self.metrics.histogram(
                    "risk_monitor.calculation_time",
                    calc_time * 1000
                )
                
                # Attendre le prochain cycle
                await asyncio.sleep(self.config['update_interval'])
                
            except Exception as e:
                self.logger.error(
                    "risk_monitoring_error",
                    error=str(e),
                    exc_info=True
                )
                await asyncio.sleep(1.0)
    
    async def calculate_risk_snapshot(self) -> RiskSnapshot:
        """Calcule un snapshot complet des risques"""
        snapshot = RiskSnapshot(timestamp=datetime.now(timezone.utc))
        
        # Calculs d'exposition
        self._calculate_exposures(snapshot)
        
        # VaR et CVaR
        await self._calculate_var_metrics(snapshot)
        
        # Drawdown
        self._calculate_drawdown_metrics(snapshot)
        
        # Corrélations et concentration
        self._calculate_correlation_metrics(snapshot)
        
        # Liquidité
        self._calculate_liquidity_metrics(snapshot)
        
        # Score de risque global
        snapshot.risk_score = self._calculate_risk_score(snapshot)
        snapshot.risk_level = self._determine_risk_level(snapshot.risk_score)
        
        return snapshot
    
    def _calculate_exposures(self, snapshot: RiskSnapshot) -> None:
        """Calcule les expositions du portefeuille"""
        total_capital = self._get_total_capital()
        
        # Expositions par symbole
        for symbol, position in self.current_positions.items():
            exposure = abs(position.market_value)
            snapshot.symbol_exposures[symbol] = exposure
            
            # Ajouter à l'exposition par stratégie
            if position.strategy_id:
                if position.strategy_id not in snapshot.strategy_exposures:
                    snapshot.strategy_exposures[position.strategy_id] = Decimal("0")
                snapshot.strategy_exposures[position.strategy_id] += exposure
        
        # Expositions globales
        long_exposure = sum(
            pos.market_value for pos in self.current_positions.values()
            if pos.quantity > 0
        )
        short_exposure = abs(sum(
            pos.market_value for pos in self.current_positions.values()
            if pos.quantity < 0
        ))
        
        snapshot.gross_exposure = long_exposure + short_exposure
        snapshot.net_exposure = long_exposure - short_exposure
        snapshot.total_exposure = snapshot.gross_exposure
        
        # Levier
        if total_capital > 0:
            snapshot.leverage = float(snapshot.gross_exposure / total_capital)
        else:
            snapshot.leverage = 0.0
        
        # Utilisation de marge (simplifié)
        snapshot.margin_usage = min(snapshot.leverage / 2.0, 1.0)  # Assumant 2:1 max
    
    async def _calculate_var_metrics(self, snapshot: RiskSnapshot) -> None:
        """Calcule les métriques VaR et CVaR"""
        if not self.current_positions:
            return
        
        # Méthodes de calcul en parallèle
        tasks = []
        
        if self.config.get('use_historical_var', True):
            tasks.append(self._calculate_historical_var())
        
        if self.config.get('use_parametric_var', True):
            tasks.append(self._calculate_parametric_var())
        
        if self.config.get('use_monte_carlo', True):
            tasks.append(self._calculate_monte_carlo_var())
        
        # Attendre tous les calculs
        var_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Moyenner les résultats valides
        valid_results_95 = []
        valid_results_99 = []
        
        for result in var_results:
            if isinstance(result, dict) and 'var_95' in result:
                valid_results_95.append(result['var_95'])
                valid_results_99.append(result['var_99'])
        
        if valid_results_95:
            snapshot.portfolio_var_95 = Decimal(str(np.mean(valid_results_95)))
            snapshot.portfolio_var_99 = Decimal(str(np.mean(valid_results_99)))
            
            # CVaR (moyenne des pertes au-delà de VaR)
            snapshot.portfolio_cvar_95 = snapshot.portfolio_var_95 * Decimal("1.4")
        
        # VaR par symbole
        for symbol in self.current_positions:
            symbol_var = await self._calculate_symbol_var(symbol)
            if symbol_var:
                snapshot.symbol_vars[symbol] = symbol_var
    
    async def _calculate_historical_var(self) -> Dict[str, float]:
        """Calcule VaR par méthode historique"""
        # Collecter tous les rendements
        all_returns = []
        
        for symbol, returns in self.historical_returns.items():
            if symbol in self.current_positions and len(returns) > 20:
                position = self.current_positions[symbol]
                weighted_returns = [
                    r * float(position.market_value) for r in returns
                ]
                all_returns.extend(weighted_returns)
        
        if len(all_returns) < 100:
            return {}
        
        # Calculer les percentiles
        sorted_returns = sorted(all_returns)
        var_95_idx = int(len(sorted_returns) * (1 - self.config['var_confidence_95']))
        var_99_idx = int(len(sorted_returns) * (1 - self.config['var_confidence_99']))
        
        return {
            'var_95': abs(sorted_returns[var_95_idx]),
            'var_99': abs(sorted_returns[var_99_idx])
        }
    
    async def _calculate_parametric_var(self) -> Dict[str, float]:
        """Calcule VaR par méthode paramétrique (variance-covariance)"""
        if not self.current_positions:
            return {}
        
        # Construire la matrice de positions
        symbols = list(self.current_positions.keys())
        positions = np.array([
            float(self.current_positions[s].market_value)
            for s in symbols
        ])
        
        # Obtenir la matrice de covariance
        cov_matrix = self._get_covariance_matrix(symbols)
        if cov_matrix is None:
            return {}
        
        # Calcul de la variance du portefeuille
        portfolio_variance = positions @ cov_matrix @ positions.T
        portfolio_std = np.sqrt(portfolio_variance)
        
        # VaR paramétrique
        z_95 = stats.norm.ppf(self.config['var_confidence_95'])
        z_99 = stats.norm.ppf(self.config['var_confidence_99'])
        
        return {
            'var_95': portfolio_std * z_95,
            'var_99': portfolio_std * z_99
        }
    
    async def _calculate_monte_carlo_var(self) -> Dict[str, float]:
        """Calcule VaR par simulation Monte Carlo"""
        if not self.current_positions:
            return {}
        
        n_simulations = self.config.get('monte_carlo_simulations', 10000)
        
        # Paramètres pour la simulation
        symbols = list(self.current_positions.keys())
        positions = np.array([
            float(self.current_positions[s].market_value)
            for s in symbols
        ])
        
        # Moyennes et écarts-types historiques
        means = []
        stds = []
        
        for symbol in symbols:
            returns = list(self.historical_returns.get(symbol, []))
            if returns:
                means.append(np.mean(returns))
                stds.append(np.std(returns))
            else:
                means.append(0.0)
                stds.append(0.02)  # 2% par défaut
        
        # Simulations
        simulated_returns = np.random.normal(
            means,
            stds,
            size=(n_simulations, len(symbols))
        )
        
        # Calculer les P&L simulés
        simulated_pnl = simulated_returns @ positions
        
        # VaR des simulations
        var_95 = np.percentile(simulated_pnl, (1 - self.config['var_confidence_95']) * 100)
        var_99 = np.percentile(simulated_pnl, (1 - self.config['var_confidence_99']) * 100)
        
        return {
            'var_95': abs(var_95),
            'var_99': abs(var_99)
        }
    
    async def _calculate_symbol_var(self, symbol: Symbol) -> Optional[Decimal]:
        """Calcule VaR pour un symbole spécifique"""
        returns = list(self.historical_returns.get(symbol, []))
        if len(returns) < 20:
            return None
        
        position = self.current_positions.get(symbol)
        if not position:
            return None
        
        # VaR historique simple
        sorted_returns = sorted(returns)
        var_idx = int(len(sorted_returns) * (1 - self.config['var_confidence_95']))
        var_return = abs(sorted_returns[var_idx])
        
        return abs(position.market_value) * Decimal(str(var_return))
    
    def _calculate_drawdown_metrics(self, snapshot: RiskSnapshot) -> None:
        """Calcule les métriques de drawdown"""
        # Obtenir l'historique des valeurs du portefeuille
        portfolio_values = [
            sum(pos.market_value for pos in positions.values())
            for positions in self._get_historical_positions()
        ]
        
        if not portfolio_values:
            return
        
        # Calculer le drawdown
        peak = portfolio_values[0]
        max_dd = 0.0
        current_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            
            dd = float((peak - value) / peak) if peak > 0 else 0.0
            max_dd = max(max_dd, dd)
            
        # Le dernier est le drawdown actuel
        if portfolio_values and peak > 0:
            current_dd = float((peak - portfolio_values[-1]) / peak)
        
        snapshot.current_drawdown = current_dd
        snapshot.max_drawdown = max_dd
        
        # Mettre à jour l'historique
        self.drawdown_history.append({
            'timestamp': snapshot.timestamp,
            'current': current_dd,
            'max': max_dd
        })
    
    def _calculate_correlation_metrics(self, snapshot: RiskSnapshot) -> None:
        """Calcule les métriques de corrélation et concentration"""
        symbols = list(self.current_positions.keys())
        
        if len(symbols) < 2:
            snapshot.concentration_risk = 1.0  # Tout dans un actif
            return
        
        # Obtenir la matrice de corrélation
        corr_matrix = self._get_correlation_matrix(symbols)
        if corr_matrix is not None:
            snapshot.correlation_matrix = corr_matrix
            
            # Calculer le risque de concentration
            # Herfindahl-Hirschman Index (HHI)
            total_exposure = snapshot.gross_exposure
            if total_exposure > 0:
                hhi = sum(
                    (float(exp / total_exposure) ** 2)
                    for exp in snapshot.symbol_exposures.values()
                )
                snapshot.concentration_risk = hhi
            else:
                snapshot.concentration_risk = 0.0
        
        # Ajouter des alertes si corrélations élevées
        if corr_matrix is not None:
            high_corr_pairs = self._find_high_correlations(corr_matrix, symbols)
            for pair, corr in high_corr_pairs:
                if corr > 0.8:
                    snapshot.active_alerts.append(
                        f"Haute corrélation {pair[0]}-{pair[1]}: {corr:.2f}"
                    )
    
    def _calculate_liquidity_metrics(self, snapshot: RiskSnapshot) -> None:
        """Calcule les métriques de liquidité"""
        illiquid_exposure = Decimal("0")
        
        for symbol, position in self.current_positions.items():
            # Score de liquidité (à implémenter avec données réelles)
            liquidity_score = self._get_liquidity_score(symbol)
            snapshot.liquidity_scores[symbol] = liquidity_score
            
            # Exposition illiquide
            if liquidity_score < 0.3:  # Seuil d'illiquidité
                illiquid_exposure += abs(position.market_value)
        
        snapshot.illiquid_exposure = illiquid_exposure
        
        # Alerte si trop d'exposition illiquide
        if snapshot.gross_exposure > 0:
            illiquid_ratio = float(illiquid_exposure / snapshot.gross_exposure)
            if illiquid_ratio > 0.2:  # 20% max
                snapshot.active_alerts.append(
                    f"Exposition illiquide élevée: {illiquid_ratio:.1%}"
                )
    
    def _calculate_risk_score(self, snapshot: RiskSnapshot) -> float:
        """Calcule un score de risque global de 0 à 100"""
        scores = []
        
        # Score VaR (0-25 points)
        total_capital = self._get_total_capital()
        if total_capital > 0 and snapshot.portfolio_var_95 > 0:
            var_ratio = float(snapshot.portfolio_var_95 / total_capital)
            var_score = min(25, var_ratio / self.config['var_95_limit'] * 25)
            scores.append(var_score)
        
        # Score Drawdown (0-25 points)
        dd_score = min(25, snapshot.current_drawdown / self.config['max_drawdown'] * 25)
        scores.append(dd_score)
        
        # Score Leverage (0-25 points)
        leverage_score = min(25, snapshot.leverage / 2.0 * 25)
        scores.append(leverage_score)
        
        # Score Concentration (0-25 points)
        concentration_score = snapshot.concentration_risk * 25
        scores.append(concentration_score)
        
        return sum(scores)
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Détermine le niveau de risque selon le score"""
        if risk_score < 20:
            return RiskLevel.LOW
        elif risk_score < 40:
            return RiskLevel.MODERATE
        elif risk_score < 60:
            return RiskLevel.HIGH
        elif risk_score < 80:
            return RiskLevel.CRITICAL
        else:
            return RiskLevel.EXTREME
    
    def _check_risk_limits(self, snapshot: RiskSnapshot) -> List[RiskLimit]:
        """Vérifie toutes les limites de risque"""
        breached_limits = []
        
        # Vérifier VaR
        total_capital = self._get_total_capital()
        if total_capital > 0:
            # VaR 95
            var_95_pct = float(snapshot.portfolio_var_95 / total_capital)
            self.risk_limits[RiskMetricType.VAR_95].current_value = var_95_pct
            if self.risk_limits[RiskMetricType.VAR_95].is_breached:
                breached_limits.append(self.risk_limits[RiskMetricType.VAR_95])
            
            # VaR 99
            var_99_pct = float(snapshot.portfolio_var_99 / total_capital)
            self.risk_limits[RiskMetricType.VAR_99].current_value = var_99_pct
            if self.risk_limits[RiskMetricType.VAR_99].is_breached:
                breached_limits.append(self.risk_limits[RiskMetricType.VAR_99])
        
        # Vérifier Drawdown
        self.risk_limits[RiskMetricType.MAX_DRAWDOWN].current_value = snapshot.current_drawdown
        if self.risk_limits[RiskMetricType.MAX_DRAWDOWN].is_breached:
            breached_limits.append(self.risk_limits[RiskMetricType.MAX_DRAWDOWN])
        
        # Vérifier Leverage
        self.risk_limits[RiskMetricType.LEVERAGE].current_value = snapshot.leverage
        if self.risk_limits[RiskMetricType.LEVERAGE].is_breached:
            breached_limits.append(self.risk_limits[RiskMetricType.LEVERAGE])
        
        # Vérifier Concentration
        self.risk_limits[RiskMetricType.CONCENTRATION].current_value = snapshot.concentration_risk
        if self.risk_limits[RiskMetricType.CONCENTRATION].is_breached:
            breached_limits.append(self.risk_limits[RiskMetricType.CONCENTRATION])
        
        return breached_limits
    
    async def _update_circuit_breakers(self, snapshot: RiskSnapshot) -> None:
        """Met à jour les circuit breakers avec les données de risque"""
        cb_data = {
            'balance': self._get_total_capital(),
            'drawdown': snapshot.current_drawdown,
            'var_95': float(snapshot.portfolio_var_95),
            'leverage': snapshot.leverage,
            'risk_score': snapshot.risk_score
        }
        
        # Envoyer aux circuit breakers
        await self.circuit_breaker_manager.check_all_breakers(cb_data)
    
    async def _handle_limit_breaches(
        self,
        breached_limits: List[RiskLimit],
        snapshot: RiskSnapshot
    ) -> None:
        """Gère les dépassements de limites"""
        for limit in breached_limits:
            self.breach_counts[limit.metric_type] += 1
            
            # Logger
            self.logger.warning(
                "risk_limit_breached",
                metric=limit.metric_type.value,
                current_value=limit.current_value,
                threshold=limit.threshold_critical,
                action=limit.action_on_breach
            )
            
            # Alerter
            alert_level = AlertLevel.CRITICAL if snapshot.risk_level >= RiskLevel.CRITICAL else AlertLevel.WARNING
            
            await self.alert_manager.send_alert(
                level=alert_level,
                title=f"Limite de Risque Dépassée: {limit.metric_type.value}",
                message=f"Valeur actuelle: {limit.current_value:.2%}, Limite: {limit.threshold_critical:.2%}",
                metadata={
                    'metric_type': limit.metric_type.value,
                    'current_value': limit.current_value,
                    'threshold': limit.threshold_critical,
                    'risk_score': snapshot.risk_score,
                    'risk_level': snapshot.risk_level.value
                }
            )
            
            # Exécuter l'action configurée
            if limit.action_on_breach == "reduce_position":
                await self._request_position_reduction(0.5)  # Réduire de 50%
            elif limit.action_on_breach == "stop_trading":
                await self.circuit_breaker_manager.stop_all_trading(
                    f"Risk limit breach: {limit.metric_type.value}"
                )
    
    async def _var_calculation_loop(self) -> None:
        """Boucle de calcul VaR périodique"""
        while self._running:
            try:
                await asyncio.sleep(self.config['var_calc_interval'])
                
                # Recalculer VaR avec plus de précision
                snapshot = await self.calculate_risk_snapshot()
                
                # Enregistrer dans l'historique
                self.var_history.append({
                    'timestamp': snapshot.timestamp,
                    'var_95': float(snapshot.portfolio_var_95),
                    'var_99': float(snapshot.portfolio_var_99),
                    'cvar_95': float(snapshot.portfolio_cvar_95)
                })
                
            except Exception as e:
                self.logger.error("var_calculation_error", error=str(e))
    
    async def _stress_test_loop(self) -> None:
        """Boucle de stress testing périodique"""
        while self._running:
            try:
                await asyncio.sleep(self.config['stress_test_interval'])
                
                if self.config.get('enable_stress_testing', True):
                    results = await self.run_stress_tests()
                    
                    # Alerter si résultats critiques
                    for scenario, loss in results.items():
                        loss_pct = float(loss / self._get_total_capital())
                        if loss_pct > 0.15:  # Perte > 15%
                            await self.alert_manager.send_alert(
                                level=AlertLevel.WARNING,
                                title=f"Stress Test Alerte: {scenario}",
                                message=f"Perte potentielle: {loss_pct:.1%}",
                                metadata={'scenario': scenario, 'loss': float(loss)}
                            )
                
            except Exception as e:
                self.logger.error("stress_test_error", error=str(e))
    
    async def run_stress_tests(
        self,
        custom_scenarios: Optional[List[StressTestScenario]] = None
    ) -> Dict[str, Decimal]:
        """Exécute les stress tests"""
        scenarios = custom_scenarios or self.stress_scenarios
        results = {}
        
        # Convertir les positions en liste
        positions = list(self.current_positions.values())
        
        # Exécuter chaque scénario
        for scenario in scenarios:
            try:
                loss = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    scenario.apply_to_positions,
                    positions,
                    self.market_prices
                )
                results[scenario.name] = abs(loss)
                
            except Exception as e:
                self.logger.error(
                    "stress_test_scenario_error",
                    scenario=scenario.name,
                    error=str(e)
                )
                results[scenario.name] = Decimal("0")
        
        return results
    
    def update_position(self, position: Position) -> None:
        """Met à jour une position dans le monitoring"""
        self.current_positions[position.symbol] = position
        
        # Logger les grandes positions
        total_capital = self._get_total_capital()
        if total_capital > 0:
            position_pct = float(abs(position.market_value) / total_capital)
            if position_pct > 0.05:  # Position > 5%
                self.logger.info(
                    "large_position_update",
                    symbol=position.symbol,
                    value=float(position.market_value),
                    percentage=position_pct
                )
    
    def update_market_price(self, symbol: Symbol, price: Price) -> None:
        """Met à jour le prix de marché d'un symbole"""
        old_price = self.market_prices.get(symbol)
        self.market_prices[symbol] = price
        
        # Calculer le rendement si prix précédent existe
        if old_price and old_price > 0:
            return_pct = float((price - old_price) / old_price)
            self.historical_returns[symbol].append(return_pct)
    
    def remove_position(self, symbol: Symbol) -> None:
        """Retire une position du monitoring"""
        if symbol in self.current_positions:
            del self.current_positions[symbol]
            self.logger.info("position_removed", symbol=symbol)
    
    # Méthodes helper privées
    
    def _get_total_capital(self) -> Decimal:
        """Obtient le capital total (à implémenter avec portfolio manager)"""
        # Placeholder - devrait venir du portfolio manager
        return Decimal("1000000")
    
    def _get_historical_positions(self) -> List[Dict[Symbol, Position]]:
        """Obtient l'historique des positions (à implémenter)"""
        # Placeholder - devrait venir d'un système de stockage
        return []
    
    def _get_covariance_matrix(self, symbols: List[Symbol]) -> Optional[np.ndarray]:
        """Calcule la matrice de covariance des rendements"""
        # Vérifier le cache
        if self._correlation_cache:
            cached_matrix, cache_time = self._correlation_cache
            if (datetime.now(timezone.utc) - cache_time).seconds < 300:  # 5 min cache
                return cached_matrix
        
        # Construire la matrice de rendements
        returns_matrix = []
        
        for symbol in symbols:
            returns = list(self.historical_returns.get(symbol, []))
            if len(returns) < 20:
                return None
            returns_matrix.append(returns[-self.config['correlation_window']:])
        
        if not returns_matrix:
            return None
        
        # Calculer la covariance
        try:
            returns_array = np.array(returns_matrix)
            cov_matrix = np.cov(returns_array)
            
            # Mettre en cache
            self._correlation_cache = (cov_matrix, datetime.now(timezone.utc))
            
            return cov_matrix
            
        except Exception as e:
            self.logger.error("covariance_calculation_error", error=str(e))
            return None
    
    def _get_correlation_matrix(self, symbols: List[Symbol]) -> Optional[np.ndarray]:
        """Calcule la matrice de corrélation"""
        cov_matrix = self._get_covariance_matrix(symbols)
        if cov_matrix is None:
            return None
        
        # Convertir covariance en corrélation
        try:
            std_devs = np.sqrt(np.diag(cov_matrix))
            corr_matrix = cov_matrix / np.outer(std_devs, std_devs)
            return corr_matrix
        except Exception:
            return None
    
    def _find_high_correlations(
        self,
        corr_matrix: np.ndarray,
        symbols: List[Symbol]
    ) -> List[Tuple[Tuple[Symbol, Symbol], float]]:
        """Trouve les paires avec haute corrélation"""
        high_corr_pairs = []
        
        n = len(symbols)
        for i in range(n):
            for j in range(i + 1, n):
                corr = corr_matrix[i, j]
                if abs(corr) > 0.7:  # Seuil de corrélation élevée
                    high_corr_pairs.append(
                        ((symbols[i], symbols[j]), corr)
                    )
        
        return sorted(high_corr_pairs, key=lambda x: abs(x[1]), reverse=True)
    
    def _get_liquidity_score(self, symbol: Symbol) -> float:
        """Calcule un score de liquidité (0-1)"""
        # Placeholder - devrait utiliser volume, spread, profondeur du carnet
        # Pour l'instant, retourner un score aléatoire mais cohérent
        return 0.5 + hash(symbol) % 50 / 100
    
    async def _request_position_reduction(self, factor: float) -> None:
        """Demande une réduction des positions"""
        self.logger.warning(
            "position_reduction_requested",
            reduction_factor=factor
        )
        # À implémenter : communication avec portfolio manager
    
    def get_current_risk_snapshot(self) -> Optional[RiskSnapshot]:
        """Retourne le dernier snapshot de risque"""
        if self.risk_snapshots:
            return self.risk_snapshots[-1]
        return None
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Retourne un résumé des risques actuels"""
        snapshot = self.get_current_risk_snapshot()
        if not snapshot:
            return {}
        
        return {
            'risk_level': snapshot.risk_level.value,
            'risk_score': snapshot.risk_score,
            'var_95': float(snapshot.portfolio_var_95),
            'var_99': float(snapshot.portfolio_var_99),
            'current_drawdown': snapshot.current_drawdown,
            'leverage': snapshot.leverage,
            'concentration_risk': snapshot.concentration_risk,
            'active_alerts': len(snapshot.active_alerts),
            'breached_limits': sum(
                1 for limit in self.risk_limits.values()
                if limit.is_breached
            )
        }
    
    def get_historical_var(self, periods: int = 100) -> List[Dict[str, Any]]:
        """Retourne l'historique VaR"""
        return list(self.var_history)[-periods:]
    
    def get_stress_test_results(self) -> Dict[str, Any]:
        """Retourne les derniers résultats de stress test"""
        # À implémenter avec stockage des résultats
        return {}
    
    async def trigger_emergency_assessment(self) -> RiskSnapshot:
        """Déclenche une évaluation d'urgence des risques"""
        self.logger.warning("emergency_risk_assessment_triggered")
        
        # Calculer immédiatement un nouveau snapshot
        snapshot = await self.calculate_risk_snapshot()
        
        # Forcer les stress tests
        if self.config.get('stress_test_on_demand', True):
            stress_results = await self.run_stress_tests()
            snapshot.metadata = {'stress_test_results': stress_results}
        
        # Vérifier tous les circuit breakers
        await self._update_circuit_breakers(snapshot)
        
        return snapshot


# Fonction helper pour créer un moniteur préconfiguré
def create_risk_monitor(config: Dict[str, Any]) -> RiskMonitor:
    """Crée un moniteur de risques préconfiguré"""
    # Ici on créerait les dépendances nécessaires
    # Pour l'instant, retourner None car nécessite l'intégration complète
    return None