"""
Advanced Backtesting Module
Système de backtesting rigoureux pour l'évaluation des stratégies de trading.
Inclut walk-forward analysis, Monte Carlo, et tests de stress.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Métriques de performance
from empyrical import (
    sharpe_ratio, sortino_ratio, calmar_ratio, omega_ratio,
    max_drawdown, annual_return, annual_volatility,
    tail_ratio, capture_ratio
)

# Import des modules internes
from ..environments.trading_env import TradingEnv
from ...strategies.base_strategy import BaseStrategy
from ...utils.metrics import calculate_all_metrics
from ...data.processors.data_normalizer import DataNormalizer

# Configuration du logger
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration pour le backtesting"""
    # Données
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    symbols: List[str] = field(default_factory=lambda: ['BTC/USD'])
    
    # Capital et risque
    initial_capital: float = 100000
    commission: float = 0.001
    slippage_model: str = 'fixed'  # fixed, linear, market_impact
    slippage_rate: float = 0.0005
    
    # Position sizing
    position_sizing: str = 'fixed'  # fixed, kelly, risk_parity
    max_position_size: float = 0.3
    leverage: float = 1.0
    
    # Walk-forward
    walk_forward: bool = True
    in_sample_periods: int = 252  # Jours de trading
    out_sample_periods: int = 63
    step_size: int = 21
    
    # Monte Carlo
    monte_carlo_simulations: int = 1000
    confidence_intervals: List[float] = field(default_factory=lambda: [0.05, 0.95])
    
    # Benchmark
    benchmark: Optional[str] = 'BTC/USD'
    risk_free_rate: float = 0.02
    
    # Analyse
    calculate_factor_exposure: bool = True
    stress_test_scenarios: List[str] = field(default_factory=lambda: ['flash_crash', 'black_swan', 'high_volatility'])
    
    # Output
    save_trades: bool = True
    save_equity_curve: bool = True
    generate_report: bool = True
    output_dir: str = 'backtest_results'


@dataclass
class BacktestResults:
    """Résultats détaillés du backtesting"""
    # Performance globale
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Statistiques de trading
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    
    # Risque
    annual_volatility: float
    downside_deviation: float
    var_95: float
    cvar_95: float
    beta: float
    alpha: float
    
    # Séries temporelles
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    drawdown_series: pd.Series
    
    # Trades détaillés
    trades: pd.DataFrame
    
    # Walk-forward results
    walk_forward_results: Optional[Dict] = None
    
    # Monte Carlo results
    monte_carlo_results: Optional[Dict] = None
    
    # Stress test results
    stress_test_results: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convertir en dictionnaire"""
        result = asdict(self)
        # Convertir les Series/DataFrame en listes pour JSON
        result['equity_curve'] = self.equity_curve.to_list()
        result['returns'] = self.returns.to_list()
        result['positions'] = self.positions.to_list()
        result['trades'] = self.trades.to_dict('records')
        return result


class AdvancedBacktester:
    """
    Système de backtesting avancé avec walk-forward analysis,
    Monte Carlo simulations et stress testing.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialisation du backtester
        
        Args:
            config: Configuration du backtesting
        """
        self.config = config
        self.results_dir = Path(config.output_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache pour les calculs
        self.cache = {}
        
        # Benchmark data
        self.benchmark_returns = None
        
        # Slippage models
        self.slippage_models = {
            'fixed': self._fixed_slippage,
            'linear': self._linear_slippage,
            'market_impact': self._market_impact_slippage
        }
    
    def backtest(self, strategy: Union[BaseStrategy, Any], 
                data: pd.DataFrame,
                start_date: Optional[str] = None,
                end_date: Optional[str] = None) -> BacktestResults:
        """
        Exécuter un backtest complet
        
        Args:
            strategy: Stratégie à tester (ou agent DRL)
            data: Données historiques
            start_date: Date de début (optionnel)
            end_date: Date de fin (optionnel)
            
        Returns:
            Résultats détaillés du backtest
        """
        logger.info("Début du backtesting")
        
        # Filtrer les données par date si nécessaire
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Backtesting simple
        basic_results = self._run_basic_backtest(strategy, data)
        
        # Walk-forward analysis
        if self.config.walk_forward:
            logger.info("Exécution de l'analyse walk-forward")
            basic_results.walk_forward_results = self._walk_forward_analysis(strategy, data)
        
        # Monte Carlo simulations
        if self.config.monte_carlo_simulations > 0:
            logger.info(f"Exécution de {self.config.monte_carlo_simulations} simulations Monte Carlo")
            basic_results.monte_carlo_results = self._monte_carlo_analysis(basic_results)
        
        # Stress testing
        if self.config.stress_test_scenarios:
            logger.info("Exécution des tests de stress")
            basic_results.stress_test_results = self._stress_test_analysis(strategy, data)
        
        # Sauvegarder les résultats
        if self.config.generate_report:
            self._generate_report(basic_results)
        
        return basic_results
    
    def _run_basic_backtest(self, strategy: Any, data: pd.DataFrame) -> BacktestResults:
        """Exécuter un backtest de base"""
        # Initialisation
        capital = self.config.initial_capital
        position = 0
        trades = []
        equity_curve = [capital]
        positions = [0]
        timestamps = []
        
        # Pour les agents DRL, créer un environnement
        if hasattr(strategy, 'select_action'):  # Agent DRL
            env = TradingEnv(
                data,
                initial_balance=capital,
                transaction_cost=self.config.commission
            )
            state = env.reset()
        
        # Parcourir les données
        for i in tqdm(range(len(data)), desc="Backtesting"):
            current_price = data['close'].iloc[i]
            timestamp = data.index[i]
            timestamps.append(timestamp)
            
            # Obtenir le signal
            if hasattr(strategy, 'select_action'):  # Agent DRL
                action = strategy.select_action(state, training=False)
                # Convertir l'action en signal (-1, 0, 1)
                signal = action - 1 if env.action_space.n == 3 else 0
                
                # Step dans l'environnement pour obtenir le prochain état
                state, _, done, _ = env.step(action)
                if done:
                    state = env.reset()
            else:  # Stratégie classique
                signal = strategy.generate_signal(data.iloc[:i+1])
            
            # Calculer la nouvelle position
            target_position = signal * self.config.max_position_size
            
            # Appliquer les changements de position
            if target_position != position:
                # Calculer le trade
                trade_size = target_position - position
                
                # Appliquer slippage et commission
                execution_price = self._apply_slippage(current_price, trade_size)
                commission = abs(trade_size * execution_price * self.config.commission)
                
                # Mettre à jour le capital
                capital -= trade_size * execution_price * capital + commission
                
                # Enregistrer le trade
                trades.append({
                    'timestamp': timestamp,
                    'type': 'BUY' if trade_size > 0 else 'SELL',
                    'size': abs(trade_size),
                    'price': execution_price,
                    'commission': commission,
                    'capital': capital
                })
                
                position = target_position
            
            # Mettre à jour la valeur du portefeuille
            portfolio_value = capital * (1 + position * (current_price / data['close'].iloc[i-1] - 1))
            equity_curve.append(portfolio_value)
            positions.append(position)
            capital = portfolio_value
        
        # Créer les séries temporelles
        equity_series = pd.Series(equity_curve, index=timestamps)
        returns_series = equity_series.pct_change().fillna(0)
        positions_series = pd.Series(positions[:-1], index=timestamps)
        
        # Calculer les métriques
        results = self._calculate_metrics(
            equity_series,
            returns_series,
            positions_series,
            pd.DataFrame(trades)
        )
        
        return results
    
    def _walk_forward_analysis(self, strategy: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyse walk-forward pour éviter l'overfitting"""
        results = {
            'windows': [],
            'in_sample_performance': [],
            'out_sample_performance': [],
            'efficiency_ratio': []
        }
        
        total_periods = len(data)
        window_start = 0
        
        while window_start + self.config.in_sample_periods + self.config.out_sample_periods <= total_periods:
            # Définir les périodes
            in_sample_end = window_start + self.config.in_sample_periods
            out_sample_end = in_sample_end + self.config.out_sample_periods
            
            # Données in-sample et out-sample
            in_sample_data = data.iloc[window_start:in_sample_end]
            out_sample_data = data.iloc[in_sample_end:out_sample_end]
            
            # Entraîner sur in-sample (si applicable)
            if hasattr(strategy, 'fit'):
                strategy.fit(in_sample_data)
            
            # Tester sur les deux périodes
            in_sample_results = self._run_basic_backtest(strategy, in_sample_data)
            out_sample_results = self._run_basic_backtest(strategy, out_sample_data)
            
            # Enregistrer les résultats
            results['windows'].append({
                'start': data.index[window_start],
                'in_sample_end': data.index[in_sample_end],
                'out_sample_end': data.index[out_sample_end]
            })
            
            results['in_sample_performance'].append(in_sample_results.sharpe_ratio)
            results['out_sample_performance'].append(out_sample_results.sharpe_ratio)
            
            # Efficiency ratio: out-sample / in-sample performance
            if in_sample_results.sharpe_ratio != 0:
                efficiency = out_sample_results.sharpe_ratio / in_sample_results.sharpe_ratio
            else:
                efficiency = 0
            results['efficiency_ratio'].append(efficiency)
            
            # Avancer la fenêtre
            window_start += self.config.step_size
        
        # Statistiques globales
        results['avg_efficiency_ratio'] = np.mean(results['efficiency_ratio'])
        results['stability'] = 1 - np.std(results['out_sample_performance'])
        
        # Test de significativité
        if len(results['in_sample_performance']) > 1:
            t_stat, p_value = stats.ttest_rel(
                results['out_sample_performance'],
                results['in_sample_performance']
            )
            results['statistical_significance'] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return results
    
    def _monte_carlo_analysis(self, base_results: BacktestResults) -> Dict[str, Any]:
        """Simulations Monte Carlo pour analyser la robustesse"""
        returns = base_results.returns.values
        
        # Paramètres de la distribution
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Simulations
        simulated_paths = []
        final_values = []
        max_drawdowns = []
        sharpe_ratios = []
        
        for _ in range(self.config.monte_carlo_simulations):
            # Générer des returns aléatoires
            # Option 1: Distribution normale
            simulated_returns = np.random.normal(mean_return, std_return, len(returns))
            
            # Option 2: Bootstrap (resample historique)
            # simulated_returns = np.random.choice(returns, size=len(returns), replace=True)
            
            # Option 3: Block bootstrap pour préserver l'autocorrélation
            # block_size = 20
            # simulated_returns = self._block_bootstrap(returns, block_size)
            
            # Calculer la courbe d'équité
            equity_curve = self.config.initial_capital * (1 + simulated_returns).cumprod()
            simulated_paths.append(equity_curve)
            
            # Métriques
            final_values.append(equity_curve[-1])
            max_drawdowns.append(self._calculate_max_drawdown(equity_curve))
            
            if np.std(simulated_returns) > 0:
                sharpe = np.mean(simulated_returns) / np.std(simulated_returns) * np.sqrt(252)
                sharpe_ratios.append(sharpe)
        
        # Statistiques des simulations
        results = {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'confidence_intervals': {
                str(ci): np.percentile(final_values, ci * 100)
                for ci in self.config.confidence_intervals
            },
            'probability_of_profit': sum(v > self.config.initial_capital for v in final_values) / len(final_values),
            'probability_of_loss_50pct': sum(v < self.config.initial_capital * 0.5 for v in final_values) / len(final_values),
            'max_drawdown_distribution': {
                'mean': np.mean(max_drawdowns),
                'median': np.median(max_drawdowns),
                'p95': np.percentile(max_drawdowns, 95),
                'p99': np.percentile(max_drawdowns, 99)
            },
            'sharpe_ratio_distribution': {
                'mean': np.mean(sharpe_ratios),
                'median': np.median(sharpe_ratios),
                'std': np.std(sharpe_ratios),
                'p5': np.percentile(sharpe_ratios, 5),
                'p95': np.percentile(sharpe_ratios, 95)
            }
        }
        
        # Value at Risk (VaR) et Conditional VaR
        sorted_returns = np.sort(returns)
        var_index = int(0.05 * len(sorted_returns))
        results['var_95'] = sorted_returns[var_index]
        results['cvar_95'] = np.mean(sorted_returns[:var_index])
        
        return results
    
    def _stress_test_analysis(self, strategy: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Tests de stress avec scénarios extrêmes"""
        results = {}
        
        for scenario in self.config.stress_test_scenarios:
            logger.info(f"Test de stress: {scenario}")
            
            # Générer les données stressées
            stressed_data = self._generate_stress_scenario(data, scenario)
            
            # Backtester sur les données stressées
            stress_results = self._run_basic_backtest(strategy, stressed_data)
            
            results[scenario] = {
                'total_return': stress_results.total_return,
                'max_drawdown': stress_results.max_drawdown,
                'sharpe_ratio': stress_results.sharpe_ratio,
                'number_of_trades': stress_results.total_trades,
                'survival': stress_results.equity_curve.iloc[-1] > 0
            }
        
        # Analyse de sensibilité
        results['sensitivity_analysis'] = self._sensitivity_analysis(strategy, data)
        
        return results
    
    def _generate_stress_scenario(self, data: pd.DataFrame, scenario: str) -> pd.DataFrame:
        """Générer des données pour un scénario de stress"""
        stressed_data = data.copy()
        
        if scenario == 'flash_crash':
            # Chute soudaine de 20% sur une journée aléatoire
            crash_day = np.random.randint(len(data) // 4, 3 * len(data) // 4)
            stressed_data['close'].iloc[crash_day] *= 0.8
            stressed_data['low'].iloc[crash_day] *= 0.75
            
            # Récupération progressive
            recovery_days = 20
            for i in range(1, recovery_days):
                if crash_day + i < len(stressed_data):
                    stressed_data['close'].iloc[crash_day + i] *= 1 + (0.2 / recovery_days)
        
        elif scenario == 'black_swan':
            # Événement extrême: -50% sur 3 jours
            crash_start = np.random.randint(len(data) // 4, 3 * len(data) // 4)
            for i in range(3):
                if crash_start + i < len(stressed_data):
                    stressed_data['close'].iloc[crash_start + i] *= 0.8
                    stressed_data['high'].iloc[crash_start + i] *= 0.85
                    stressed_data['low'].iloc[crash_start + i] *= 0.75
        
        elif scenario == 'high_volatility':
            # Augmenter la volatilité de 3x
            returns = stressed_data['close'].pct_change()
            amplified_returns = returns * 3
            stressed_data['close'] = stressed_data['close'].iloc[0] * (1 + amplified_returns).cumprod()
            
            # Ajuster high/low
            stressed_data['high'] = stressed_data['close'] * 1.02
            stressed_data['low'] = stressed_data['close'] * 0.98
        
        elif scenario == 'liquidity_crisis':
            # Augmenter les spreads et réduire les volumes
            if 'volume' in stressed_data.columns:
                stressed_data['volume'] *= 0.1
            if 'bid' in stressed_data.columns and 'ask' in stressed_data.columns:
                spread = stressed_data['ask'] - stressed_data['bid']
                stressed_data['bid'] -= spread * 2
                stressed_data['ask'] += spread * 2
        
        return stressed_data
    
    def _sensitivity_analysis(self, strategy: Any, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyse de sensibilité aux paramètres"""
        base_params = {
            'commission': self.config.commission,
            'slippage_rate': self.config.slippage_rate,
            'max_position_size': self.config.max_position_size
        }
        
        results = {}
        
        # Tester différentes valeurs de paramètres
        param_ranges = {
            'commission': [0.0005, 0.001, 0.002, 0.005],
            'slippage_rate': [0.0001, 0.0005, 0.001, 0.002],
            'max_position_size': [0.1, 0.2, 0.3, 0.5]
        }
        
        for param, values in param_ranges.items():
            param_results = []
            
            for value in values:
                # Modifier temporairement le paramètre
                original_value = getattr(self.config, param)
                setattr(self.config, param, value)
                
                # Backtester
                result = self._run_basic_backtest(strategy, data)
                param_results.append({
                    'value': value,
                    'sharpe_ratio': result.sharpe_ratio,
                    'total_return': result.total_return,
                    'max_drawdown': result.max_drawdown
                })
                
                # Restaurer la valeur originale
                setattr(self.config, param, original_value)
            
            results[param] = param_results
        
        return results
    
    def _calculate_metrics(self, equity_curve: pd.Series,
                         returns: pd.Series,
                         positions: pd.Series,
                         trades: pd.DataFrame) -> BacktestResults:
        """Calculer toutes les métriques de performance"""
        # Métriques de base
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
        annual_ret = (1 + total_return) ** (252 / len(returns)) - 1
        
        # Ratios de performance
        sharpe = sharpe_ratio(returns, risk_free=self.config.risk_free_rate)
        sortino = sortino_ratio(returns, risk_free=self.config.risk_free_rate)
        calmar = calmar_ratio(returns)
        
        # Drawdown
        drawdown_series = self._calculate_drawdown_series(equity_curve)
        max_dd = drawdown_series.min()
        max_dd_duration = self._calculate_max_drawdown_duration(drawdown_series)
        
        # Statistiques de trading
        if len(trades) > 0:
            winning_trades = trades[trades['type'] == 'SELL']['capital'].diff() > 0
            total_trades = len(trades)
            win_rate = winning_trades.sum() / total_trades if total_trades > 0 else 0
            
            # Profit factor
            gross_profits = trades[trades['capital'].diff() > 0]['capital'].diff().sum()
            gross_losses = abs(trades[trades['capital'].diff() < 0]['capital'].diff().sum())
            profit_factor = gross_profits / gross_losses if gross_losses > 0 else np.inf
        else:
            total_trades = 0
            win_rate = 0
            profit_factor = 0
        
        # Volatilité et risque
        annual_vol = annual_volatility(returns)
        downside_dev = returns[returns < 0].std() * np.sqrt(252)
        
        # Beta et alpha (si benchmark disponible)
        if self.benchmark_returns is not None:
            beta, alpha = self._calculate_beta_alpha(returns, self.benchmark_returns)
        else:
            beta, alpha = 0, 0
        
        return BacktestResults(
            total_return=total_return,
            annual_return=annual_ret,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            total_trades=total_trades,
            winning_trades=int(win_rate * total_trades),
            losing_trades=int((1 - win_rate) * total_trades),
            win_rate=win_rate,
            avg_win=0,  # À calculer
            avg_loss=0,  # À calculer
            profit_factor=profit_factor,
            expectancy=0,  # À calculer
            annual_volatility=annual_vol,
            downside_deviation=downside_dev,
            var_95=np.percentile(returns, 5),
            cvar_95=returns[returns <= np.percentile(returns, 5)].mean(),
            beta=beta,
            alpha=alpha,
            equity_curve=equity_curve,
            returns=returns,
            positions=positions,
            drawdown_series=drawdown_series,
            trades=trades
        )
    
    def _apply_slippage(self, price: float, size: float) -> float:
        """Appliquer le modèle de slippage"""
        slippage_func = self.slippage_models.get(
            self.config.slippage_model, 
            self._fixed_slippage
        )
        return slippage_func(price, size)
    
    def _fixed_slippage(self, price: float, size: float) -> float:
        """Slippage fixe"""
        if size > 0:  # Achat
            return price * (1 + self.config.slippage_rate)
        else:  # Vente
            return price * (1 - self.config.slippage_rate)
    
    def _linear_slippage(self, price: float, size: float) -> float:
        """Slippage linéaire en fonction de la taille"""
        impact = self.config.slippage_rate * abs(size)
        if size > 0:
            return price * (1 + impact)
        else:
            return price * (1 - impact)
    
    def _market_impact_slippage(self, price: float, size: float) -> float:
        """Modèle d'impact de marché (square-root)"""
        # Modèle simplifié: impact = lambda * sqrt(size)
        impact = self.config.slippage_rate * np.sqrt(abs(size))
        if size > 0:
            return price * (1 + impact)
        else:
            return price * (1 - impact)
    
    def _calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculer la série de drawdown"""
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        return drawdown
    
    def _calculate_max_drawdown(self, equity_curve: Union[pd.Series, np.ndarray]) -> float:
        """Calculer le drawdown maximum"""
        if isinstance(equity_curve, pd.Series):
            equity_curve = equity_curve.values
        
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return drawdown.min()
    
    def _calculate_max_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """Calculer la durée maximale de drawdown"""
        underwater = drawdown_series < 0
        
        # Identifier les périodes underwater
        underwater_periods = []
        start = None
        
        for i, is_underwater in enumerate(underwater):
            if is_underwater and start is None:
                start = i
            elif not is_underwater and start is not None:
                underwater_periods.append(i - start)
                start = None
        
        # Si on termine underwater
        if start is not None:
            underwater_periods.append(len(underwater) - start)
        
        return max(underwater_periods) if underwater_periods else 0
    
    def _calculate_beta_alpha(self, returns: pd.Series, 
                            benchmark_returns: pd.Series) -> Tuple[float, float]:
        """Calculer beta et alpha par rapport au benchmark"""
        # Aligner les séries
        aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
        
        if len(aligned_returns) < 20:  # Pas assez de données
            return 0, 0
        
        # Régression linéaire
        covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
        benchmark_variance = np.var(aligned_benchmark)
        
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        alpha = np.mean(aligned_returns) - beta * np.mean(aligned_benchmark)
        
        # Annualiser alpha
        alpha = alpha * 252
        
        return beta, alpha
    
    def _generate_report(self, results: BacktestResults):
        """Générer un rapport détaillé"""
        report_dir = self.results_dir / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        report_dir.mkdir(exist_ok=True)
        
        # 1. Sauvegarder les résultats bruts
        with open(report_dir / 'results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        # 2. Sauvegarder les métriques en JSON
        metrics_dict = {
            'performance': {
                'total_return': results.total_return,
                'annual_return': results.annual_return,
                'sharpe_ratio': results.sharpe_ratio,
                'sortino_ratio': results.sortino_ratio,
                'calmar_ratio': results.calmar_ratio,
                'max_drawdown': results.max_drawdown
            },
            'trading': {
                'total_trades': results.total_trades,
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor
            },
            'risk': {
                'annual_volatility': results.annual_volatility,
                'var_95': results.var_95,
                'cvar_95': results.cvar_95
            }
        }
        
        with open(report_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # 3. Générer les graphiques
        self._generate_charts(results, report_dir)
        
        # 4. Générer un rapport HTML
        self._generate_html_report(results, report_dir)
        
        logger.info(f"Rapport généré dans: {report_dir}")
    
    def _generate_charts(self, results: BacktestResults, output_dir: Path):
        """Générer les graphiques de performance"""
        # Configuration matplotlib
        plt.style.use('seaborn-v0_8-darkgrid')
        fig_size = (12, 8)
        
        # 1. Courbe d'équité
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size, sharex=True)
        
        ax1.plot(results.equity_curve.index, results.equity_curve.values, 
                label='Strategy', linewidth=2)
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Equity Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2.fill_between(results.drawdown_series.index, 
                        results.drawdown_series.values * 100, 0,
                        color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.set_title('Drawdown')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'equity_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Distribution des returns
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
        
        # Histogramme
        ax1.hist(results.returns * 100, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(results.returns.mean() * 100, color='red', linestyle='--', 
                   label=f'Mean: {results.returns.mean()*100:.2f}%')
        ax1.set_xlabel('Returns (%)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Returns Distribution')
        ax1.legend()
        
        # Q-Q plot
        stats.probplot(results.returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'returns_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Rolling metrics
        fig, axes = plt.subplots(3, 1, figsize=fig_size, sharex=True)
        
        # Rolling Sharpe (252 days)
        rolling_sharpe = results.returns.rolling(252).apply(
            lambda x: sharpe_ratio(x) if len(x) == 252 else np.nan
        )
        axes[0].plot(rolling_sharpe.index, rolling_sharpe.values, color='green')
        axes[0].set_ylabel('Sharpe Ratio')
        axes[0].set_title('Rolling 252-Day Sharpe Ratio')
        axes[0].grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol = results.returns.rolling(252).std() * np.sqrt(252) * 100
        axes[1].plot(rolling_vol.index, rolling_vol.values, color='orange')
        axes[1].set_ylabel('Volatility (%)')
        axes[1].set_title('Rolling 252-Day Volatility')
        axes[1].grid(True, alpha=0.3)
        
        # Positions
        axes[2].plot(results.positions.index, results.positions.values, color='purple')
        axes[2].set_ylabel('Position')
        axes[2].set_xlabel('Date')
        axes[2].set_title('Position Over Time')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'rolling_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Walk-forward results (si disponible)
        if results.walk_forward_results:
            self._plot_walk_forward_results(results.walk_forward_results, output_dir)
        
        # 5. Monte Carlo results (si disponible)
        if results.monte_carlo_results:
            self._plot_monte_carlo_results(results.monte_carlo_results, output_dir)
    
    def _plot_walk_forward_results(self, wf_results: Dict, output_dir: Path):
        """Graphiques pour walk-forward analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Performance in-sample vs out-sample
        windows = range(len(wf_results['in_sample_performance']))
        ax1.plot(windows, wf_results['in_sample_performance'], 
                label='In-Sample', marker='o')
        ax1.plot(windows, wf_results['out_sample_performance'], 
                label='Out-Sample', marker='s')
        ax1.set_xlabel('Window')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_title('Walk-Forward Performance')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Efficiency ratio
        ax2.plot(windows, wf_results['efficiency_ratio'], 
                color='red', marker='^')
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
        ax2.axhline(y=wf_results['avg_efficiency_ratio'], 
                   color='green', linestyle='--', 
                   label=f"Avg: {wf_results['avg_efficiency_ratio']:.2f}")
        ax2.set_xlabel('Window')
        ax2.set_ylabel('Efficiency Ratio')
        ax2.set_title('Out-Sample / In-Sample Performance Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'walk_forward_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_monte_carlo_results(self, mc_results: Dict, output_dir: Path):
        """Graphiques pour Monte Carlo simulations"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Distribution des valeurs finales
        final_values = np.random.normal(
            mc_results['mean_final_value'],
            (mc_results['confidence_intervals']['0.95'] - mc_results['confidence_intervals']['0.05']) / 3.92,
            1000
        )
        axes[0, 0].hist(final_values, bins=50, alpha=0.7, color='blue')
        axes[0, 0].axvline(self.config.initial_capital, color='red', 
                          linestyle='--', label='Initial Capital')
        axes[0, 0].axvline(mc_results['mean_final_value'], color='green', 
                          linestyle='-', label='Mean')
        axes[0, 0].set_xlabel('Final Portfolio Value ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Monte Carlo Final Values Distribution')
        axes[0, 0].legend()
        
        # Distribution des drawdowns
        dd_mean = mc_results['max_drawdown_distribution']['mean']
        dd_std = (mc_results['max_drawdown_distribution']['p95'] - 
                 mc_results['max_drawdown_distribution']['p5']) / 3.92
        drawdowns = np.random.normal(dd_mean, dd_std, 1000) * 100
        axes[0, 1].hist(drawdowns, bins=50, alpha=0.7, color='red')
        axes[0, 1].set_xlabel('Max Drawdown (%)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Monte Carlo Max Drawdown Distribution')
        
        # Distribution des Sharpe ratios
        sharpe_mean = mc_results['sharpe_ratio_distribution']['mean']
        sharpe_std = mc_results['sharpe_ratio_distribution']['std']
        sharpes = np.random.normal(sharpe_mean, sharpe_std, 1000)
        axes[1, 0].hist(sharpes, bins=50, alpha=0.7, color='green')
        axes[1, 0].axvline(0, color='red', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('Sharpe Ratio')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Monte Carlo Sharpe Ratio Distribution')
        
        # Probabilités
        labels = ['Profit', 'Loss > 50%']
        probabilities = [
            mc_results['probability_of_profit'],
            mc_results['probability_of_loss_50pct']
        ]
        axes[1, 1].bar(labels, probabilities, color=['green', 'red'])
        axes[1, 1].set_ylabel('Probability')
        axes[1, 1].set_title('Outcome Probabilities')
        axes[1, 1].set_ylim(0, 1)
        
        for i, v in enumerate(probabilities):
            axes[1, 1].text(i, v + 0.01, f'{v:.2%}', ha='center')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'monte_carlo_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_html_report(self, results: BacktestResults, output_dir: Path):
        """Générer un rapport HTML complet"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Backtest Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                h1 {{ color: #333; text-align: center; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 10px; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
                .metric-card {{ 
                    background: white; 
                    padding: 20px; 
                    border-radius: 8px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{ font-size: 24px; font-weight: bold; color: #2196F3; }}
                .metric-label {{ color: #666; margin-top: 5px; }}
                .chart-container {{ 
                    background: white; 
                    padding: 20px; 
                    margin: 20px 0; 
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                img {{ max-width: 100%; height: auto; }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    background: white;
                    margin: 20px 0;
                }}
                th, td {{ 
                    padding: 12px; 
                    text-align: left; 
                    border-bottom: 1px solid #ddd;
                }}
                th {{ background-color: #f8f8f8; font-weight: bold; }}
                .positive {{ color: #4CAF50; }}
                .negative {{ color: #F44336; }}
            </style>
        </head>
        <body>
            <h1>Backtest Report</h1>
            
            <h2>Performance Summary</h2>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">{results.total_return*100:.2f}%</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{results.sharpe_ratio:.2f}</div>
                    <div class="metric-label">Sharpe Ratio</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{abs(results.max_drawdown)*100:.2f}%</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{results.win_rate*100:.1f}%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{results.profit_factor:.2f}</div>
                    <div class="metric-label">Profit Factor</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{results.total_trades}</div>
                    <div class="metric-label">Total Trades</div>
                </div>
            </div>
            
            <h2>Equity Curve</h2>
            <div class="chart-container">
                <img src="equity_curve.png" alt="Equity Curve">
            </div>
            
            <h2>Returns Distribution</h2>
            <div class="chart-container">
                <img src="returns_distribution.png" alt="Returns Distribution">
            </div>
            
            <h2>Rolling Metrics</h2>
            <div class="chart-container">
                <img src="rolling_metrics.png" alt="Rolling Metrics">
            </div>
        """
        
        # Ajouter walk-forward results si disponible
        if results.walk_forward_results:
            html_content += f"""
            <h2>Walk-Forward Analysis</h2>
            <div class="chart-container">
                <img src="walk_forward_analysis.png" alt="Walk-Forward Analysis">
                <p>Average Efficiency Ratio: {results.walk_forward_results['avg_efficiency_ratio']:.2f}</p>
                <p>Stability Score: {results.walk_forward_results['stability']:.2f}</p>
            </div>
            """
        
        # Ajouter Monte Carlo results si disponible
        if results.monte_carlo_results:
            html_content += f"""
            <h2>Monte Carlo Analysis</h2>
            <div class="chart-container">
                <img src="monte_carlo_analysis.png" alt="Monte Carlo Analysis">
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Probability of Profit</td>
                        <td class="positive">{results.monte_carlo_results['probability_of_profit']*100:.1f}%</td>
                    </tr>
                    <tr>
                        <td>95% Confidence Interval</td>
                        <td>${results.monte_carlo_results['confidence_intervals']['0.05']:,.0f} - 
                            ${results.monte_carlo_results['confidence_intervals']['0.95']:,.0f}</td>
                    </tr>
                    <tr>
                        <td>Value at Risk (95%)</td>
                        <td class="negative">{results.monte_carlo_results['var_95']*100:.2f}%</td>
                    </tr>
                </table>
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_dir / 'report.html', 'w') as f:
            f.write(html_content)


# Fonctions utilitaires pour backtesting parallèle
def parallel_backtest(strategies: List[Any], data: pd.DataFrame, 
                     config: BacktestConfig, n_jobs: int = -1) -> Dict[str, BacktestResults]:
    """
    Backtester plusieurs stratégies en parallèle
    
    Args:
        strategies: Liste des stratégies à tester
        data: Données historiques
        config: Configuration du backtesting
        n_jobs: Nombre de processus (-1 pour tous les CPU)
        
    Returns:
        Dictionnaire des résultats par stratégie
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    def backtest_single(strategy_tuple):
        name, strategy = strategy_tuple
        backtester = AdvancedBacktester(config)
        return name, backtester.backtest(strategy, data)
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        strategy_tuples = [(f"strategy_{i}", s) for i, s in enumerate(strategies)]
        results = dict(executor.map(backtest_single, strategy_tuples))
    
    return results


# Exemple d'utilisation
def main():
    """Exemple d'utilisation du backtester"""
    # Configuration
    config = BacktestConfig(
        initial_capital=100000,
        commission=0.001,
        walk_forward=True,
        monte_carlo_simulations=1000,
        stress_test_scenarios=['flash_crash', 'black_swan', 'high_volatility']
    )
    
    # Créer des données de test
    dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 101,
        'low': np.random.randn(len(dates)).cumsum() + 99,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.exponential(1000000, len(dates))
    }, index=dates)
    
    # Créer une stratégie simple (exemple)
    class SimpleStrategy:
        def generate_signal(self, data):
            # Stratégie momentum simple
            if len(data) < 20:
                return 0
            returns = data['close'].pct_change()
            if returns.iloc[-1] > returns.rolling(20).mean().iloc[-1]:
                return 1
            else:
                return -1
    
    strategy = SimpleStrategy()
    
    # Backtester
    backtester = AdvancedBacktester(config)
    results = backtester.backtest(strategy, data)
    
    print(f"Total Return: {results.total_return*100:.2f}%")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {results.max_drawdown*100:.2f}%")
    print(f"Win Rate: {results.win_rate*100:.1f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()