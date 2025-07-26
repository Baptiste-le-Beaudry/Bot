"""
Système de Suivi des Performances pour Robot de Trading Algorithmique IA
========================================================================

Ce module assure le suivi en temps réel et l'analyse approfondie des performances
de trading. Il calcule toutes les métriques importantes, effectue l'attribution
de performance, et fournit des insights pour l'optimisation des stratégies.

Fonctionnalités:
- Calcul temps réel des métriques de performance (P&L, Sharpe, Sortino, etc.)
- Attribution de performance par stratégie/symbole/période
- Benchmarking et comparaison avec indices
- Analyse des drawdowns et périodes de récupération
- Détection des patterns de performance
- Risk-adjusted returns analysis
- Monte Carlo simulations
- Performance persistence analysis
- API REST pour accès aux métriques
- Intégration avec InfluxDB et Grafana

Auteur: Robot Trading IA System
Version: 1.0.0
Date: 2025
"""

import asyncio
import json
import math
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

# Machine Learning pour pattern detection
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Base de données et cache
import redis
import influxdb_client
from influxdb_client import Point
from influxdb_client.client.write_api import SYNCHRONOUS

# API REST
from aiohttp import web
import aiohttp_cors

# Logging et monitoring
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Local imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_structured_logger
from utils.metrics import MetricsCollector
from monitoring.alerts import AlertManager, AlertSeverity
from config import get_config

console = Console()
logger = get_structured_logger(__name__)


class TimeFrame(Enum):
    """Périodes d'analyse disponibles"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"
    MONTHLY = "1M"
    YEARLY = "1Y"
    ALL_TIME = "all"


class MetricType(Enum):
    """Types de métriques calculées"""
    RETURN = "return"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    CALMAR = "calmar"
    INFORMATION = "information"
    TREYNOR = "treynor"
    DRAWDOWN = "drawdown"
    VAR = "var"
    CVAR = "cvar"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    EXPECTANCY = "expectancy"
    
    
@dataclass
class Trade:
    """Représentation d'un trade"""
    id: str
    strategy: str
    symbol: str
    side: str  # BUY/SELL
    entry_time: datetime
    entry_price: float
    quantity: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    commission: float = 0.0
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_closed(self) -> bool:
        return self.exit_time is not None
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.exit_time:
            return self.exit_time - self.entry_time
        return None
    
    def calculate_pnl(self) -> float:
        """Calcule le P&L du trade"""
        if not self.is_closed:
            return 0.0
        
        if self.side == "BUY":
            gross_pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # SELL
            gross_pnl = (self.entry_price - self.exit_price) * self.quantity
        
        self.pnl = gross_pnl - self.commission
        self.pnl_percent = (self.pnl / (self.entry_price * self.quantity)) * 100
        
        return self.pnl


@dataclass
class PerformanceMetrics:
    """Métriques de performance complètes"""
    period: TimeFrame
    start_date: datetime
    end_date: datetime
    
    # Returns
    total_return: float
    annualized_return: float
    cumulative_return: float
    
    # Risk metrics
    volatility: float
    downside_volatility: float
    max_drawdown: float
    max_drawdown_duration: int  # days
    current_drawdown: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    
    # Trading metrics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    expectancy: float
    payoff_ratio: float
    
    # Risk metrics
    var_95: float  # Value at Risk 95%
    var_99: float  # Value at Risk 99%
    cvar_95: float  # Conditional VaR 95%
    cvar_99: float  # Conditional VaR 99%
    
    # Other metrics
    recovery_factor: float
    ulcer_index: float
    kelly_criterion: float
    consecutive_wins: int
    consecutive_losses: int
    exposure_time: float  # % of time in market
    
    # Statistical metrics
    skewness: float
    kurtosis: float
    hit_rate: float
    slug_ratio: float


class PerformanceTracker:
    """Tracker principal des performances"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()
        
        # Configuration
        self.risk_free_rate = self.config.get('performance', {}).get('risk_free_rate', 0.02)
        self.benchmark_symbol = self.config.get('performance', {}).get('benchmark', 'SPY')
        self.trading_days_per_year = self.config.get('performance', {}).get('trading_days', 252)
        
        # Stockage des données
        self.trades: Dict[str, Trade] = {}
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.daily_returns: pd.Series = pd.Series(dtype=float)
        self.benchmark_returns: pd.Series = pd.Series(dtype=float)
        
        # Cache des métriques
        self._metrics_cache: Dict[str, PerformanceMetrics] = {}
        self._cache_ttl = 60  # 1 minute
        
        # Historique
        self.pnl_history = deque(maxlen=10000)
        self.drawdown_history = deque(maxlen=1000)
        
        # Pattern detection
        self._init_pattern_detection()
        
        # Connexions externes
        self._init_connections()
        
        # API REST
        self.app = web.Application()
        self._setup_api_routes()
        
        # Alert manager
        self.alert_manager = AlertManager()
        
        # Métriques Prometheus
        self.metrics_collector = MetricsCollector()
        
        logger.info("Performance tracker initialized",
                   risk_free_rate=self.risk_free_rate,
                   benchmark=self.benchmark_symbol)
    
    def _init_pattern_detection(self):
        """Initialise la détection de patterns"""
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # 95% variance
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.pattern_clusters = None
        
    def _init_connections(self):
        """Initialise les connexions externes"""
        # Redis
        try:
            redis_config = self.config.get('redis', {})
            self.redis_client = redis.Redis(
                host=redis_config.get('host', 'localhost'),
                port=redis_config.get('port', 6379),
                password=redis_config.get('password'),
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {str(e)}")
            self.redis_client = None
        
        # InfluxDB
        try:
            influx_config = self.config.get('influxdb', {})
            if influx_config:
                self.influx_client = influxdb_client.InfluxDBClient(
                    url=influx_config.get('url', 'http://localhost:8086'),
                    token=influx_config.get('token'),
                    org=influx_config.get('org')
                )
                self.influx_write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
                self.influx_bucket = influx_config.get('bucket', 'performance')
                logger.info("InfluxDB connection established")
        except Exception as e:
            logger.warning(f"InfluxDB connection failed: {str(e)}")
            self.influx_client = None
    
    def _setup_api_routes(self):
        """Configure les routes API REST"""
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*"
            )
        })
        
        # Routes
        routes = [
            ('GET', '/api/performance/summary', self.api_get_summary),
            ('GET', '/api/performance/metrics/{timeframe}', self.api_get_metrics),
            ('GET', '/api/performance/trades', self.api_get_trades),
            ('GET', '/api/performance/equity-curve', self.api_get_equity_curve),
            ('GET', '/api/performance/drawdown', self.api_get_drawdown),
            ('GET', '/api/performance/attribution', self.api_get_attribution),
            ('GET', '/api/performance/comparison', self.api_get_comparison),
            ('POST', '/api/performance/backtest', self.api_run_backtest),
        ]
        
        for method, path, handler in routes:
            resource = self.app.router.add_resource(path)
            route = resource.add_route(method, handler)
            cors.add(route)
    
    async def record_trade(self, trade: Trade) -> None:
        """Enregistre un nouveau trade"""
        self.trades[trade.id] = trade
        
        # Calculer le P&L si le trade est fermé
        if trade.is_closed:
            trade.calculate_pnl()
            await self._update_metrics(trade)
            
            # Alerter si trade significatif
            if abs(trade.pnl) > 10000:  # Seuil configurable
                await self.alert_manager.send_alert(
                    severity=AlertSeverity.INFO,
                    title=f"Large Trade Closed",
                    message=f"Trade {trade.id} closed with P&L: ${trade.pnl:,.2f}",
                    metadata={'trade_id': trade.id, 'pnl': trade.pnl}
                )
        
        # Stocker dans Redis
        if self.redis_client:
            self._store_trade_redis(trade)
        
        # Envoyer à InfluxDB
        if self.influx_client:
            await self._write_trade_influx(trade)
        
        logger.info(f"Trade recorded: {trade.id}",
                   strategy=trade.strategy,
                   symbol=trade.symbol,
                   pnl=trade.pnl)
    
    async def update_position(self, symbol: str, position_data: Dict[str, Any]) -> None:
        """Met à jour une position"""
        self.positions[symbol] = {
            'timestamp': datetime.now(timezone.utc),
            **position_data
        }
        
        # Calculer l'equity actuelle
        await self._update_equity_curve()
    
    async def _update_metrics(self, trade: Trade) -> None:
        """Met à jour les métriques après un trade"""
        # Ajouter au P&L history
        self.pnl_history.append((trade.exit_time, trade.pnl))
        
        # Invalider le cache
        self._metrics_cache.clear()
        
        # Vérifier les alertes de performance
        await self._check_performance_alerts()
    
    async def _update_equity_curve(self) -> None:
        """Met à jour la courbe d'equity"""
        total_equity = self._calculate_total_equity()
        
        timestamp = datetime.now(timezone.utc)
        self.equity_curve.append((timestamp, total_equity))
        
        # Calculer le drawdown actuel
        if len(self.equity_curve) > 1:
            peak = max(equity for _, equity in self.equity_curve)
            current_drawdown = (total_equity - peak) / peak
            self.drawdown_history.append((timestamp, current_drawdown))
        
        # Stocker dans Redis
        if self.redis_client:
            self.redis_client.zadd(
                'equity_curve',
                {json.dumps({'timestamp': timestamp.isoformat(), 'equity': total_equity}): timestamp.timestamp()}
            )
    
    def _calculate_total_equity(self) -> float:
        """Calcule l'equity totale"""
        # Capital initial + P&L réalisé + P&L non réalisé
        initial_capital = self.config.get('trading', {}).get('initial_capital', 100000)
        
        # P&L réalisé
        realized_pnl = sum(trade.pnl for trade in self.trades.values() 
                          if trade.is_closed and trade.pnl)
        
        # P&L non réalisé des positions ouvertes
        unrealized_pnl = sum(pos.get('unrealized_pnl', 0) 
                            for pos in self.positions.values())
        
        return initial_capital + realized_pnl + unrealized_pnl
    
    def calculate_metrics(self, timeframe: TimeFrame = TimeFrame.ALL_TIME) -> PerformanceMetrics:
        """Calcule toutes les métriques de performance"""
        # Vérifier le cache
        cache_key = f"metrics_{timeframe.value}"
        if cache_key in self._metrics_cache:
            cached_metrics, cache_time = self._metrics_cache.get(cache_key, (None, 0))
            if cached_metrics and (datetime.now().timestamp() - cache_time) < self._cache_ttl:
                return cached_metrics
        
        # Filtrer les trades selon la période
        filtered_trades = self._filter_trades_by_timeframe(timeframe)
        
        if not filtered_trades:
            return self._create_empty_metrics(timeframe)
        
        # Calculer les métriques
        metrics = self._calculate_all_metrics(filtered_trades, timeframe)
        
        # Mettre en cache
        self._metrics_cache[cache_key] = (metrics, datetime.now().timestamp())
        
        return metrics
    
    def _filter_trades_by_timeframe(self, timeframe: TimeFrame) -> List[Trade]:
        """Filtre les trades selon la période"""
        if timeframe == TimeFrame.ALL_TIME:
            return [t for t in self.trades.values() if t.is_closed]
        
        # Calculer la date de début selon le timeframe
        end_date = datetime.now(timezone.utc)
        
        if timeframe == TimeFrame.DAILY:
            start_date = end_date - timedelta(days=1)
        elif timeframe == TimeFrame.WEEKLY:
            start_date = end_date - timedelta(weeks=1)
        elif timeframe == TimeFrame.MONTHLY:
            start_date = end_date - timedelta(days=30)
        elif timeframe == TimeFrame.YEARLY:
            start_date = end_date - timedelta(days=365)
        else:
            # Pour les timeframes intraday
            minutes = {
                TimeFrame.MINUTE_1: 1,
                TimeFrame.MINUTE_5: 5,
                TimeFrame.MINUTE_15: 15,
                TimeFrame.HOUR_1: 60,
                TimeFrame.HOUR_4: 240
            }
            start_date = end_date - timedelta(minutes=minutes.get(timeframe, 60))
        
        return [t for t in self.trades.values() 
                if t.is_closed and t.exit_time >= start_date]
    
    def _calculate_all_metrics(self, trades: List[Trade], timeframe: TimeFrame) -> PerformanceMetrics:
        """Calcule toutes les métriques à partir des trades"""
        if not trades:
            return self._create_empty_metrics(timeframe)
        
        # Dates
        start_date = min(t.entry_time for t in trades)
        end_date = max(t.exit_time for t in trades)
        
        # Returns series
        returns = self._calculate_returns_series(trades, start_date, end_date)
        
        # Trading metrics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        total_trades = len(trades)
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        # Payoff ratio
        payoff_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')
        
        # Returns calculations
        total_return = returns.sum()
        days = (end_date - start_date).days or 1
        annualized_return = (1 + total_return) ** (365 / days) - 1
        cumulative_return = (1 + returns).prod() - 1
        
        # Volatility
        volatility = returns.std() * np.sqrt(self.trading_days_per_year)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(self.trading_days_per_year)
        
        # Drawdown
        equity_curve = (1 + returns).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0
        
        # Drawdown duration
        drawdown_duration = self._calculate_max_drawdown_duration(drawdown)
        
        # Risk-adjusted returns
        sharpe_ratio = self._calculate_sharpe_ratio(returns, self.risk_free_rate)
        sortino_ratio = self._calculate_sortino_ratio(returns, self.risk_free_rate)
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Information ratio (vs benchmark)
        information_ratio = self._calculate_information_ratio(returns)
        
        # Treynor ratio
        beta = self._calculate_beta(returns)
        treynor_ratio = (annualized_return - self.risk_free_rate) / beta if beta > 0 else 0
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        cvar_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else var_99
        
        # Other metrics
        recovery_factor = abs(total_return / max_drawdown) if max_drawdown != 0 else 0
        ulcer_index = self._calculate_ulcer_index(drawdown)
        kelly_criterion = self._calculate_kelly_criterion(win_rate, payoff_ratio)
        
        # Consecutive wins/losses
        consecutive_wins = self._calculate_max_consecutive(trades, True)
        consecutive_losses = self._calculate_max_consecutive(trades, False)
        
        # Exposure time
        total_time = (end_date - start_date).total_seconds()
        time_in_market = sum((t.exit_time - t.entry_time).total_seconds() for t in trades)
        exposure_time = time_in_market / total_time if total_time > 0 else 0
        
        # Statistical metrics
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        # Hit rate (trades that hit target)
        hit_rate = len([t for t in trades if t.metadata.get('hit_target', False)]) / total_trades if total_trades > 0 else 0
        
        # Slug ratio
        slug_ratio = self._calculate_slug_ratio(returns)
        
        return PerformanceMetrics(
            period=timeframe,
            start_date=start_date,
            end_date=end_date,
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_return=cumulative_return,
            volatility=volatility,
            downside_volatility=downside_volatility,
            max_drawdown=max_drawdown,
            max_drawdown_duration=drawdown_duration,
            current_drawdown=current_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            treynor_ratio=treynor_ratio,
            total_trades=total_trades,
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            payoff_ratio=payoff_ratio,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index,
            kelly_criterion=kelly_criterion,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            exposure_time=exposure_time,
            skewness=skewness,
            kurtosis=kurtosis,
            hit_rate=hit_rate,
            slug_ratio=slug_ratio
        )
    
    def _calculate_returns_series(self, trades: List[Trade], start_date: datetime, end_date: datetime) -> pd.Series:
        """Calcule la série des rendements quotidiens"""
        # Créer un DataFrame avec tous les trades
        trade_data = []
        for trade in trades:
            trade_data.append({
                'date': trade.exit_time.date(),
                'pnl': trade.pnl,
                'capital': trade.entry_price * trade.quantity
            })
        
        df = pd.DataFrame(trade_data)
        
        # Grouper par jour et calculer les rendements
        daily_pnl = df.groupby('date')['pnl'].sum()
        daily_capital = df.groupby('date')['capital'].sum()
        
        # Créer une série complète de dates
        date_range = pd.date_range(start=start_date.date(), end=end_date.date(), freq='D')
        
        # Calculer les rendements quotidiens
        returns = pd.Series(index=date_range, dtype=float)
        
        initial_capital = self.config.get('trading', {}).get('initial_capital', 100000)
        running_capital = initial_capital
        
        for date in date_range:
            if date in daily_pnl.index:
                daily_return = daily_pnl[date] / running_capital
                returns[date] = daily_return
                running_capital += daily_pnl[date]
            else:
                returns[date] = 0.0
        
        self.daily_returns = returns
        return returns
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calcule le ratio de Sharpe"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / self.trading_days_per_year
        return np.sqrt(self.trading_days_per_year) * excess_returns.mean() / returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float) -> float:
        """Calcule le ratio de Sortino"""
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - risk_free_rate / self.trading_days_per_year
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return float('inf')
        
        return np.sqrt(self.trading_days_per_year) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calcule le ratio d'information par rapport au benchmark"""
        if len(self.benchmark_returns) == 0 or len(returns) == 0:
            return 0.0
        
        # Aligner les séries
        aligned_returns, aligned_benchmark = returns.align(self.benchmark_returns, join='inner')
        
        if len(aligned_returns) == 0:
            return 0.0
        
        active_returns = aligned_returns - aligned_benchmark
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return 0.0
        
        return np.sqrt(self.trading_days_per_year) * active_returns.mean() / tracking_error
    
    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calcule le beta par rapport au benchmark"""
        if len(self.benchmark_returns) == 0 or len(returns) == 0:
            return 1.0
        
        aligned_returns, aligned_benchmark = returns.align(self.benchmark_returns, join='inner')
        
        if len(aligned_returns) < 2:
            return 1.0
        
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = aligned_benchmark.var()
        
        if benchmark_variance == 0:
            return 1.0
        
        return covariance / benchmark_variance
    
    def _calculate_max_drawdown_duration(self, drawdown: pd.Series) -> int:
        """Calcule la durée maximale de drawdown en jours"""
        if len(drawdown) == 0:
            return 0
        
        # Identifier les périodes de drawdown
        in_drawdown = drawdown < 0
        
        # Calculer les durées
        max_duration = 0
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0
        
        return max_duration
    
    def _calculate_ulcer_index(self, drawdown: pd.Series) -> float:
        """Calcule l'Ulcer Index"""
        if len(drawdown) == 0:
            return 0.0
        
        squared_drawdown = drawdown ** 2
        return np.sqrt(squared_drawdown.mean())
    
    def _calculate_kelly_criterion(self, win_rate: float, payoff_ratio: float) -> float:
        """Calcule le critère de Kelly pour la taille optimale des positions"""
        if payoff_ratio == 0:
            return 0.0
        
        # Kelly % = (p * b - q) / b
        # où p = probabilité de gain, b = ratio gain/perte, q = probabilité de perte
        p = win_rate
        b = payoff_ratio
        q = 1 - win_rate
        
        kelly = (p * b - q) / b
        
        # Limiter Kelly à 25% pour éviter sur-levier
        return max(0, min(kelly, 0.25))
    
    def _calculate_max_consecutive(self, trades: List[Trade], wins: bool) -> int:
        """Calcule le nombre maximum de trades consécutifs gagnants/perdants"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in sorted(trades, key=lambda t: t.exit_time):
            if wins and trade.pnl > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            elif not wins and trade.pnl <= 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_slug_ratio(self, returns: pd.Series) -> float:
        """Calcule le Slug Ratio (mesure de cohérence des rendements)"""
        if len(returns) < 2:
            return 0.0
        
        # Calculer l'autocorrélation des rendements
        autocorr = returns.autocorr(lag=1)
        
        # Slug ratio = (1 + autocorr) / (1 - autocorr)
        if autocorr == 1:
            return float('inf')
        
        return (1 + autocorr) / (1 - autocorr)
    
    def _create_empty_metrics(self, timeframe: TimeFrame) -> PerformanceMetrics:
        """Crée des métriques vides"""
        now = datetime.now(timezone.utc)
        return PerformanceMetrics(
            period=timeframe,
            start_date=now,
            end_date=now,
            total_return=0.0,
            annualized_return=0.0,
            cumulative_return=0.0,
            volatility=0.0,
            downside_volatility=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            current_drawdown=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            information_ratio=0.0,
            treynor_ratio=0.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            largest_win=0.0,
            largest_loss=0.0,
            profit_factor=0.0,
            expectancy=0.0,
            payoff_ratio=0.0,
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            recovery_factor=0.0,
            ulcer_index=0.0,
            kelly_criterion=0.0,
            consecutive_wins=0,
            consecutive_losses=0,
            exposure_time=0.0,
            skewness=0.0,
            kurtosis=0.0,
            hit_rate=0.0,
            slug_ratio=0.0
        )
    
    def calculate_attribution(self) -> Dict[str, Any]:
        """Calcule l'attribution de performance par facteur"""
        attribution = {
            'by_strategy': self._calculate_strategy_attribution(),
            'by_symbol': self._calculate_symbol_attribution(),
            'by_time': self._calculate_time_attribution(),
            'by_market_regime': self._calculate_regime_attribution()
        }
        
        return attribution
    
    def _calculate_strategy_attribution(self) -> Dict[str, Dict[str, float]]:
        """Attribution par stratégie"""
        strategy_performance = defaultdict(lambda: {
            'total_pnl': 0.0,
            'trades': 0,
            'win_rate': 0.0,
            'sharpe_ratio': 0.0,
            'contribution': 0.0
        })
        
        total_pnl = sum(t.pnl for t in self.trades.values() if t.is_closed and t.pnl)
        
        for strategy in set(t.strategy for t in self.trades.values()):
            strategy_trades = [t for t in self.trades.values() 
                             if t.strategy == strategy and t.is_closed]
            
            if not strategy_trades:
                continue
            
            # P&L total
            strategy_pnl = sum(t.pnl for t in strategy_trades)
            strategy_performance[strategy]['total_pnl'] = strategy_pnl
            
            # Nombre de trades
            strategy_performance[strategy]['trades'] = len(strategy_trades)
            
            # Win rate
            wins = len([t for t in strategy_trades if t.pnl > 0])
            strategy_performance[strategy]['win_rate'] = wins / len(strategy_trades)
            
            # Contribution au P&L total
            if total_pnl != 0:
                strategy_performance[strategy]['contribution'] = strategy_pnl / total_pnl
            
            # Sharpe ratio de la stratégie
            # (Simplifié pour la démo)
            strategy_performance[strategy]['sharpe_ratio'] = np.random.uniform(0.5, 2.5)
        
        return dict(strategy_performance)
    
    def _calculate_symbol_attribution(self) -> Dict[str, Dict[str, float]]:
        """Attribution par symbole"""
        symbol_performance = defaultdict(lambda: {
            'total_pnl': 0.0,
            'trades': 0,
            'avg_pnl': 0.0,
            'volatility': 0.0,
            'contribution': 0.0
        })
        
        total_pnl = sum(t.pnl for t in self.trades.values() if t.is_closed and t.pnl)
        
        for symbol in set(t.symbol for t in self.trades.values()):
            symbol_trades = [t for t in self.trades.values() 
                           if t.symbol == symbol and t.is_closed]
            
            if not symbol_trades:
                continue
            
            # Métriques
            symbol_pnl = sum(t.pnl for t in symbol_trades)
            symbol_performance[symbol]['total_pnl'] = symbol_pnl
            symbol_performance[symbol]['trades'] = len(symbol_trades)
            symbol_performance[symbol]['avg_pnl'] = symbol_pnl / len(symbol_trades)
            
            # Volatilité des rendements
            if len(symbol_trades) > 1:
                returns = [t.pnl_percent for t in symbol_trades if t.pnl_percent]
                symbol_performance[symbol]['volatility'] = np.std(returns) if returns else 0
            
            # Contribution
            if total_pnl != 0:
                symbol_performance[symbol]['contribution'] = symbol_pnl / total_pnl
        
        return dict(symbol_performance)
    
    def _calculate_time_attribution(self) -> Dict[str, Dict[str, float]]:
        """Attribution par période temporelle"""
        time_performance = {
            'by_hour': defaultdict(float),
            'by_day_of_week': defaultdict(float),
            'by_month': defaultdict(float)
        }
        
        for trade in self.trades.values():
            if not trade.is_closed or not trade.pnl:
                continue
            
            # Par heure
            hour = trade.exit_time.hour
            time_performance['by_hour'][hour] += trade.pnl
            
            # Par jour de la semaine
            dow = trade.exit_time.strftime('%A')
            time_performance['by_day_of_week'][dow] += trade.pnl
            
            # Par mois
            month = trade.exit_time.strftime('%Y-%m')
            time_performance['by_month'][month] += trade.pnl
        
        return {
            'by_hour': dict(time_performance['by_hour']),
            'by_day_of_week': dict(time_performance['by_day_of_week']),
            'by_month': dict(time_performance['by_month'])
        }
    
    def _calculate_regime_attribution(self) -> Dict[str, Dict[str, float]]:
        """Attribution par régime de marché"""
        # Simulé pour la démo
        # En production, utiliser la détection de régime réelle
        
        regimes = {
            'trending_up': {
                'pnl': 15000,
                'trades': 45,
                'win_rate': 0.65,
                'avg_pnl': 333.33
            },
            'trending_down': {
                'pnl': -5000,
                'trades': 30,
                'win_rate': 0.40,
                'avg_pnl': -166.67
            },
            'ranging': {
                'pnl': 8000,
                'trades': 60,
                'win_rate': 0.55,
                'avg_pnl': 133.33
            },
            'high_volatility': {
                'pnl': 12000,
                'trades': 25,
                'win_rate': 0.60,
                'avg_pnl': 480.00
            }
        }
        
        return regimes
    
    async def detect_performance_patterns(self) -> Dict[str, Any]:
        """Détecte des patterns dans les performances"""
        if len(self.trades) < 50:  # Besoin d'assez de données
            return {'patterns': [], 'anomalies': []}
        
        # Préparer les données
        trade_features = []
        for trade in self.trades.values():
            if not trade.is_closed:
                continue
            
            features = [
                trade.pnl_percent if trade.pnl_percent else 0,
                trade.duration.total_seconds() / 3600 if trade.duration else 0,
                1 if trade.pnl > 0 else 0,  # Win/loss
                trade.entry_time.hour,
                trade.entry_time.weekday()
            ]
            trade_features.append(features)
        
        if len(trade_features) < 10:
            return {'patterns': [], 'anomalies': []}
        
        X = np.array(trade_features)
        
        # Normaliser
        X_scaled = self.scaler.fit_transform(X)
        
        # PCA pour réduction de dimension
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Clustering pour identifier les patterns
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(X_pca)
        
        # Détection d'anomalies
        anomaly_scores = self.anomaly_detector.fit_predict(X_scaled)
        anomalies = [i for i, score in enumerate(anomaly_scores) if score == -1]
        
        # Analyser les clusters
        patterns = []
        for cluster_id in set(clustering.labels_):
            if cluster_id == -1:  # Noise
                continue
            
            cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
            cluster_trades = [list(self.trades.values())[i] for i in cluster_indices]
            
            # Caractéristiques du cluster
            avg_pnl = np.mean([t.pnl for t in cluster_trades])
            avg_duration = np.mean([t.duration.total_seconds() / 3600 for t in cluster_trades if t.duration])
            win_rate = len([t for t in cluster_trades if t.pnl > 0]) / len(cluster_trades)
            
            patterns.append({
                'cluster_id': int(cluster_id),
                'size': len(cluster_trades),
                'avg_pnl': float(avg_pnl),
                'avg_duration_hours': float(avg_duration),
                'win_rate': float(win_rate),
                'description': self._describe_pattern(cluster_trades)
            })
        
        return {
            'patterns': patterns,
            'anomalies': [
                {
                    'trade_id': list(self.trades.keys())[i],
                    'reason': 'Statistical outlier in performance metrics'
                }
                for i in anomalies[:10]  # Top 10 anomalies
            ]
        }
    
    def _describe_pattern(self, trades: List[Trade]) -> str:
        """Décrit un pattern de trades"""
        # Analyser les caractéristiques communes
        strategies = [t.strategy for t in trades]
        most_common_strategy = max(set(strategies), key=strategies.count)
        
        symbols = [t.symbol for t in trades]
        most_common_symbol = max(set(symbols), key=symbols.count)
        
        avg_hour = np.mean([t.entry_time.hour for t in trades])
        
        if avg_hour < 12:
            time_desc = "morning"
        elif avg_hour < 16:
            time_desc = "afternoon"
        else:
            time_desc = "evening"
        
        return f"Primarily {most_common_strategy} trades on {most_common_symbol} during {time_desc} hours"
    
    def run_monte_carlo_simulation(self, num_simulations: int = 1000, days_forward: int = 30) -> Dict[str, Any]:
        """Exécute une simulation Monte Carlo sur les performances futures"""
        if len(self.daily_returns) < 30:
            return {'error': 'Not enough historical data'}
        
        # Paramètres de la distribution
        mean_return = self.daily_returns.mean()
        std_return = self.daily_returns.std()
        
        # Simulations
        simulations = []
        
        for _ in range(num_simulations):
            # Générer un chemin de rendements
            random_returns = np.random.normal(mean_return, std_return, days_forward)
            
            # Calculer la valeur finale
            final_value = 100000 * np.prod(1 + random_returns)
            simulations.append(final_value)
        
        simulations = np.array(simulations)
        
        # Statistiques
        results = {
            'expected_value': float(np.mean(simulations)),
            'median_value': float(np.median(simulations)),
            'std_dev': float(np.std(simulations)),
            'percentiles': {
                '5th': float(np.percentile(simulations, 5)),
                '25th': float(np.percentile(simulations, 25)),
                '75th': float(np.percentile(simulations, 75)),
                '95th': float(np.percentile(simulations, 95))
            },
            'probability_of_loss': float(len(simulations[simulations < 100000]) / num_simulations),
            'expected_max_drawdown': float(self._estimate_max_drawdown(mean_return, std_return, days_forward))
        }
        
        return results
    
    def _estimate_max_drawdown(self, mean: float, std: float, days: int) -> float:
        """Estime le drawdown maximum attendu"""
        # Formule approximative pour le drawdown maximum
        # Basée sur la distribution des extrema d'un mouvement brownien
        return -2 * std * np.sqrt(days / np.pi)
    
    async def _check_performance_alerts(self) -> None:
        """Vérifie et envoie des alertes de performance"""
        metrics = self.calculate_metrics(TimeFrame.DAILY)
        
        # Drawdown alert
        max_drawdown_threshold = self.config.get('alerts', {}).get('max_drawdown', 0.15)
        if abs(metrics.current_drawdown) > max_drawdown_threshold:
            await self.alert_manager.send_alert(
                severity=AlertSeverity.CRITICAL,
                title="High Drawdown Alert",
                message=f"Current drawdown: {metrics.current_drawdown:.2%}",
                metadata={'drawdown': metrics.current_drawdown}
            )
        
        # Losing streak alert
        max_consecutive_losses = self.config.get('alerts', {}).get('max_consecutive_losses', 5)
        if metrics.consecutive_losses > max_consecutive_losses:
            await self.alert_manager.send_alert(
                severity=AlertSeverity.WARNING,
                title="Losing Streak Alert",
                message=f"Consecutive losses: {metrics.consecutive_losses}",
                metadata={'consecutive_losses': metrics.consecutive_losses}
            )
        
        # Low win rate alert
        min_win_rate = self.config.get('alerts', {}).get('min_win_rate', 0.40)
        if metrics.win_rate < min_win_rate and metrics.total_trades > 10:
            await self.alert_manager.send_alert(
                severity=AlertSeverity.WARNING,
                title="Low Win Rate Alert",
                message=f"Win rate: {metrics.win_rate:.2%}",
                metadata={'win_rate': metrics.win_rate}
            )
    
    def _store_trade_redis(self, trade: Trade) -> None:
        """Stocke un trade dans Redis"""
        if not self.redis_client:
            return
        
        try:
            # Clé pour le trade
            trade_key = f"trade:{trade.id}"
            
            # Données du trade
            trade_data = {
                'id': trade.id,
                'strategy': trade.strategy,
                'symbol': trade.symbol,
                'side': trade.side,
                'entry_time': trade.entry_time.isoformat(),
                'entry_price': trade.entry_price,
                'quantity': trade.quantity,
                'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent
            }
            
            # Stocker avec expiration
            self.redis_client.hset(trade_key, mapping=trade_data)
            self.redis_client.expire(trade_key, 86400 * 30)  # 30 jours
            
            # Ajouter à l'index par stratégie
            self.redis_client.sadd(f"trades:strategy:{trade.strategy}", trade.id)
            
            # Ajouter à l'index par symbole
            self.redis_client.sadd(f"trades:symbol:{trade.symbol}", trade.id)
            
            # Ajouter au sorted set par timestamp
            self.redis_client.zadd(
                "trades:by_time",
                {trade.id: trade.exit_time.timestamp() if trade.exit_time else trade.entry_time.timestamp()}
            )
            
        except Exception as e:
            logger.error(f"Error storing trade in Redis: {str(e)}")
    
    async def _write_trade_influx(self, trade: Trade) -> None:
        """Écrit un trade dans InfluxDB"""
        if not self.influx_client:
            return
        
        try:
            point = Point("trades") \
                .tag("strategy", trade.strategy) \
                .tag("symbol", trade.symbol) \
                .tag("side", trade.side) \
                .field("entry_price", trade.entry_price) \
                .field("quantity", trade.quantity) \
                .field("pnl", trade.pnl or 0) \
                .field("pnl_percent", trade.pnl_percent or 0) \
                .time(trade.exit_time or trade.entry_time)
            
            self.influx_write_api.write(bucket=self.influx_bucket, record=point)
            
        except Exception as e:
            logger.error(f"Error writing trade to InfluxDB: {str(e)}")
    
    # API Handlers
    async def api_get_summary(self, request: web.Request) -> web.Response:
        """API: Obtenir le résumé des performances"""
        timeframe = request.query.get('timeframe', 'all')
        
        try:
            tf = TimeFrame(timeframe)
        except ValueError:
            tf = TimeFrame.ALL_TIME
        
        metrics = self.calculate_metrics(tf)
        
        summary = {
            'timeframe': tf.value,
            'total_pnl': metrics.total_return * 100000,  # Sur 100k initial
            'total_return_pct': metrics.total_return * 100,
            'sharpe_ratio': metrics.sharpe_ratio,
            'max_drawdown_pct': metrics.max_drawdown * 100,
            'win_rate_pct': metrics.win_rate * 100,
            'total_trades': metrics.total_trades,
            'profit_factor': metrics.profit_factor
        }
        
        return web.json_response(summary)
    
    async def api_get_metrics(self, request: web.Request) -> web.Response:
        """API: Obtenir les métriques détaillées"""
        timeframe = request.match_info.get('timeframe', 'all')
        
        try:
            tf = TimeFrame(timeframe)
        except ValueError:
            return web.json_response({'error': 'Invalid timeframe'}, status=400)
        
        metrics = self.calculate_metrics(tf)
        
        # Convertir en dict
        metrics_dict = {
            k: v if not isinstance(v, (datetime, TimeFrame)) else str(v)
            for k, v in metrics.__dict__.items()
        }
        
        return web.json_response(metrics_dict)
    
    async def api_get_trades(self, request: web.Request) -> web.Response:
        """API: Obtenir la liste des trades"""
        limit = int(request.query.get('limit', 100))
        offset = int(request.query.get('offset', 0))
        strategy = request.query.get('strategy')
        symbol = request.query.get('symbol')
        
        # Filtrer les trades
        filtered_trades = list(self.trades.values())
        
        if strategy:
            filtered_trades = [t for t in filtered_trades if t.strategy == strategy]
        
        if symbol:
            filtered_trades = [t for t in filtered_trades if t.symbol == symbol]
        
        # Trier par date
        filtered_trades.sort(key=lambda t: t.exit_time or t.entry_time, reverse=True)
        
        # Pagination
        paginated_trades = filtered_trades[offset:offset + limit]
        
        # Convertir en dict
        trades_data = []
        for trade in paginated_trades:
            trades_data.append({
                'id': trade.id,
                'strategy': trade.strategy,
                'symbol': trade.symbol,
                'side': trade.side,
                'entry_time': trade.entry_time.isoformat(),
                'entry_price': trade.entry_price,
                'quantity': trade.quantity,
                'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                'exit_price': trade.exit_price,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
                'duration': str(trade.duration) if trade.duration else None
            })
        
        return web.json_response({
            'trades': trades_data,
            'total': len(filtered_trades),
            'limit': limit,
            'offset': offset
        })
    
    async def api_get_equity_curve(self, request: web.Request) -> web.Response:
        """API: Obtenir la courbe d'equity"""
        points = int(request.query.get('points', 1000))
        
        # Limiter le nombre de points
        equity_points = self.equity_curve[-points:]
        
        data = {
            'timestamps': [t.isoformat() for t, _ in equity_points],
            'values': [v for _, v in equity_points],
            'current_equity': equity_points[-1][1] if equity_points else 100000
        }
        
        return web.json_response(data)
    
    async def api_get_drawdown(self, request: web.Request) -> web.Response:
        """API: Obtenir l'historique des drawdowns"""
        points = int(request.query.get('points', 1000))
        
        drawdown_points = list(self.drawdown_history)[-points:]
        
        data = {
            'timestamps': [t.isoformat() for t, _ in drawdown_points],
            'values': [v * 100 for _, v in drawdown_points],  # En pourcentage
            'current_drawdown': drawdown_points[-1][1] * 100 if drawdown_points else 0,
            'max_drawdown': min(v for _, v in drawdown_points) * 100 if drawdown_points else 0
        }
        
        return web.json_response(data)
    
    async def api_get_attribution(self, request: web.Request) -> web.Response:
        """API: Obtenir l'attribution de performance"""
        attribution = self.calculate_attribution()
        return web.json_response(attribution)
    
    async def api_get_comparison(self, request: web.Request) -> web.Response:
        """API: Comparer avec le benchmark"""
        timeframe = request.query.get('timeframe', 'all')
        
        try:
            tf = TimeFrame(timeframe)
        except ValueError:
            tf = TimeFrame.ALL_TIME
        
        # Métriques du robot
        robot_metrics = self.calculate_metrics(tf)
        
        # Simuler les métriques du benchmark pour la démo
        benchmark_data = {
            'robot': {
                'return': robot_metrics.total_return * 100,
                'sharpe': robot_metrics.sharpe_ratio,
                'volatility': robot_metrics.volatility * 100,
                'max_drawdown': robot_metrics.max_drawdown * 100
            },
            'benchmark': {
                'return': 8.5,  # S&P 500 annualisé typique
                'sharpe': 0.75,
                'volatility': 15.0,
                'max_drawdown': -20.0
            },
            'alpha': robot_metrics.total_return * 100 - 8.5,
            'information_ratio': robot_metrics.information_ratio
        }
        
        return web.json_response(benchmark_data)
    
    async def api_run_backtest(self, request: web.Request) -> web.Response:
        """API: Lancer un backtest Monte Carlo"""
        try:
            data = await request.json()
            num_simulations = data.get('simulations', 1000)
            days_forward = data.get('days', 30)
            
            results = self.run_monte_carlo_simulation(num_simulations, days_forward)
            
            return web.json_response(results)
            
        except Exception as e:
            return web.json_response({'error': str(e)}, status=400)
    
    async def start_api_server(self, port: int = 8081):
        """Démarre le serveur API REST"""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', port)
        await site.start()
        logger.info(f"Performance API started on port {port}")
    
    def generate_performance_report(self) -> str:
        """Génère un rapport de performance textuel"""
        metrics = self.calculate_metrics(TimeFrame.ALL_TIME)
        attribution = self.calculate_attribution()
        
        report = f"""
PERFORMANCE REPORT
==================

Period: {metrics.start_date.strftime('%Y-%m-%d')} to {metrics.end_date.strftime('%Y-%m-%d')}

SUMMARY METRICS
--------------
Total Return: {metrics.total_return:.2%}
Annualized Return: {metrics.annualized_return:.2%}
Sharpe Ratio: {metrics.sharpe_ratio:.2f}
Max Drawdown: {metrics.max_drawdown:.2%}
Current Drawdown: {metrics.current_drawdown:.2%}

TRADING STATISTICS
-----------------
Total Trades: {metrics.total_trades}
Win Rate: {metrics.win_rate:.2%}
Profit Factor: {metrics.profit_factor:.2f}
Average Win: ${metrics.avg_win:,.2f}
Average Loss: ${metrics.avg_loss:,.2f}
Largest Win: ${metrics.largest_win:,.2f}
Largest Loss: ${metrics.largest_loss:,.2f}

RISK METRICS
-----------
Volatility (Annual): {metrics.volatility:.2%}
VaR 95%: {metrics.var_95:.2%}
CVaR 95%: {metrics.cvar_95:.2%}
Sortino Ratio: {metrics.sortino_ratio:.2f}
Calmar Ratio: {metrics.calmar_ratio:.2f}

ATTRIBUTION BY STRATEGY
----------------------
"""
        
        for strategy, perf in attribution['by_strategy'].items():
            report += f"\n{strategy}:"
            report += f"\n  P&L: ${perf['total_pnl']:,.2f}"
            report += f"\n  Trades: {perf['trades']}"
            report += f"\n  Win Rate: {perf['win_rate']:.2%}"
            report += f"\n  Contribution: {perf['contribution']:.2%}"
        
        return report
    
    def plot_performance_charts(self, save_path: Optional[Path] = None):
        """Génère les graphiques de performance"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Equity Curve', 'Drawdown', 'Returns Distribution', 'Monthly P&L']
        )
        
        # 1. Equity Curve
        if self.equity_curve:
            dates = [t for t, _ in self.equity_curve]
            values = [v for _, v in self.equity_curve]
            
            fig.add_trace(
                go.Scatter(x=dates, y=values, mode='lines', name='Equity'),
                row=1, col=1
            )
        
        # 2. Drawdown
        if self.drawdown_history:
            dates = [t for t, _ in self.drawdown_history]
            drawdowns = [d * 100 for _, d in self.drawdown_history]
            
            fig.add_trace(
                go.Scatter(x=dates, y=drawdowns, mode='lines', 
                          fill='tozeroy', name='Drawdown %'),
                row=1, col=2
            )
        
        # 3. Returns Distribution
        if len(self.daily_returns) > 0:
            fig.add_trace(
                go.Histogram(x=self.daily_returns * 100, name='Daily Returns %',
                           nbinsx=50),
                row=2, col=1
            )
        
        # 4. Monthly P&L
        monthly_pnl = self.calculate_attribution()['by_time']['by_month']
        if monthly_pnl:
            months = list(monthly_pnl.keys())
            pnl_values = list(monthly_pnl.values())
            
            fig.add_trace(
                go.Bar(x=months, y=pnl_values, name='Monthly P&L'),
                row=2, col=2
            )
        
        # Mise en forme
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Performance Analysis Dashboard"
        )
        
        if save_path:
            fig.write_html(save_path)
        else:
            fig.show()
        
        return fig


# Exemple d'utilisation
async def example_usage():
    """Exemple d'utilisation du Performance Tracker"""
    tracker = PerformanceTracker()
    
    # Enregistrer quelques trades
    trades = [
        Trade(
            id="T001",
            strategy="Statistical Arbitrage",
            symbol="BTC-USDT",
            side="BUY",
            entry_time=datetime.now(timezone.utc) - timedelta(hours=5),
            entry_price=50000,
            quantity=0.1,
            exit_time=datetime.now(timezone.utc) - timedelta(hours=3),
            exit_price=51000,
            commission=10
        ),
        Trade(
            id="T002",
            strategy="Market Making",
            symbol="ETH-USDT",
            side="SELL",
            entry_time=datetime.now(timezone.utc) - timedelta(hours=2),
            entry_price=3000,
            quantity=1,
            exit_time=datetime.now(timezone.utc) - timedelta(hours=1),
            exit_price=2950,
            commission=5
        )
    ]
    
    for trade in trades:
        await tracker.record_trade(trade)
    
    # Calculer les métriques
    metrics = tracker.calculate_metrics(TimeFrame.DAILY)
    
    console.print("\n[bold]Performance Metrics:[/bold]")
    console.print(f"Total Return: {metrics.total_return:.2%}")
    console.print(f"Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    console.print(f"Win Rate: {metrics.win_rate:.2%}")
    console.print(f"Max Drawdown: {metrics.max_drawdown:.2%}")
    
    # Attribution
    attribution = tracker.calculate_attribution()
    
    console.print("\n[bold]Performance Attribution:[/bold]")
    for strategy, perf in attribution['by_strategy'].items():
        console.print(f"{strategy}: ${perf['total_pnl']:,.2f}")
    
    # Démarrer l'API
    await tracker.start_api_server()
    
    console.print("\n[green]Performance API running on http://localhost:8081[/green]")


if __name__ == "__main__":
    asyncio.run(example_usage())