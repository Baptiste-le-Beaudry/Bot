"""
Hyperparameter Optimization Module
Optimisation avancée des hyperparamètres pour maximiser la rentabilité des stratégies.
Utilise Optuna, Ray Tune et d'autres techniques d'optimisation bayésienne.
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
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Optimisation libraries
import optuna
from optuna.samplers import TPESampler, CmaEsSampler, NSGAIISampler
from optuna.pruners import MedianPruner, HyperbandPruner, ThresholdPruner
from optuna.integration import TorchDistributedTrial
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_slice,
    plot_contour
)

# Alternative optimizers
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
from hyperopt.pyll import scope
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.hyperopt import HyperOptSearch

# Scikit-optimize for Gaussian Processes
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

# Genetic algorithms
from deap import base, creator, tools, algorithms

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import internal modules
from ..trainer import DRLTrainer, TrainingConfig
from ..backtesting import AdvancedBacktester, BacktestConfig
from ...utils.metrics import calculate_sharpe_ratio

# Configuration du logger
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration pour l'optimisation d'hyperparamètres"""
    # Algorithme d'optimisation
    optimizer: str = "optuna"  # optuna, hyperopt, ray_tune, skopt, genetic
    sampler: str = "tpe"  # tpe, cmaes, random, grid
    
    # Objectifs
    objectives: List[str] = field(default_factory=lambda: ["sharpe_ratio"])
    multi_objective: bool = False
    
    # Budget
    n_trials: int = 100
    timeout: Optional[int] = None  # Secondes
    max_concurrent_trials: int = 4
    
    # Pruning
    enable_pruning: bool = True
    pruner: str = "median"  # median, hyperband, threshold
    n_startup_trials: int = 10
    n_warmup_steps: int = 10
    
    # Espace de recherche
    search_space: Dict[str, Any] = field(default_factory=dict)
    
    # Évaluation
    n_splits: int = 3  # Cross-validation folds
    validation_metric: str = "sharpe_ratio"
    early_stopping_rounds: int = 50
    
    # Ressources
    gpu_per_trial: float = 0.25
    cpu_per_trial: int = 2
    distributed: bool = False
    
    # Sauvegarde
    study_name: str = "trading_hyperparam_opt"
    storage: str = "sqlite:///optuna.db"
    save_interval: int = 10
    resume: bool = True
    
    # Visualisation
    generate_plots: bool = True
    plot_interval: int = 20


@dataclass
class OptimizationResults:
    """Résultats de l'optimisation"""
    best_params: Dict[str, Any]
    best_value: float
    best_trial_number: int
    
    # Historique
    trials_history: List[Dict[str, Any]]
    param_importance: Dict[str, float]
    
    # Statistiques
    n_trials_completed: int
    n_trials_pruned: int
    optimization_time: float
    
    # Analyse
    convergence_plot: Optional[str] = None
    param_importance_plot: Optional[str] = None
    parallel_coordinate_plot: Optional[str] = None
    
    # Multi-objectif
    pareto_front: Optional[List[Dict]] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class HyperparameterOptimizer:
    """
    Système d'optimisation avancé des hyperparamètres pour le trading algorithmique.
    Supporte plusieurs algorithmes d'optimisation et objectifs.
    """
    
    def __init__(self, config: OptimizationConfig):
        """
        Initialisation de l'optimiseur
        
        Args:
            config: Configuration de l'optimisation
        """
        self.config = config
        self.results_dir = Path(f"hyperopt_results/{config.study_name}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache pour éviter de réévaluer
        self.evaluation_cache = {}
        
        # Initialiser Ray si nécessaire
        if config.distributed or config.optimizer == "ray_tune":
            ray.init(ignore_reinit_error=True)
    
    def optimize(self, 
                training_data: pd.DataFrame,
                validation_data: Optional[pd.DataFrame] = None,
                base_config: Optional[Union[TrainingConfig, BacktestConfig]] = None) -> OptimizationResults:
        """
        Lancer l'optimisation des hyperparamètres
        
        Args:
            training_data: Données d'entraînement
            validation_data: Données de validation (optionnel)
            base_config: Configuration de base à optimiser
            
        Returns:
            Résultats de l'optimisation
        """
        logger.info(f"Début de l'optimisation avec {self.config.optimizer}")
        start_time = time.time()
        
        # Sélectionner l'optimiseur
        if self.config.optimizer == "optuna":
            results = self._optimize_optuna(training_data, validation_data, base_config)
        elif self.config.optimizer == "hyperopt":
            results = self._optimize_hyperopt(training_data, validation_data, base_config)
        elif self.config.optimizer == "ray_tune":
            results = self._optimize_ray_tune(training_data, validation_data, base_config)
        elif self.config.optimizer == "skopt":
            results = self._optimize_skopt(training_data, validation_data, base_config)
        elif self.config.optimizer == "genetic":
            results = self._optimize_genetic(training_data, validation_data, base_config)
        else:
            raise ValueError(f"Optimiseur non supporté: {self.config.optimizer}")
        
        # Temps total
        results.optimization_time = time.time() - start_time
        
        # Sauvegarder les résultats
        self._save_results(results)
        
        # Générer les visualisations
        if self.config.generate_plots:
            self._generate_visualizations(results)
        
        logger.info(f"Optimisation terminée en {results.optimization_time:.2f}s")
        logger.info(f"Meilleurs paramètres: {results.best_params}")
        logger.info(f"Meilleure valeur: {results.best_value:.4f}")
        
        return results
    
    def _optimize_optuna(self, training_data: pd.DataFrame,
                        validation_data: Optional[pd.DataFrame],
                        base_config: Any) -> OptimizationResults:
        """Optimisation avec Optuna"""
        
        # Créer ou charger l'étude
        sampler = self._create_optuna_sampler()
        pruner = self._create_optuna_pruner()
        
        if self.config.multi_objective:
            study = optuna.create_study(
                study_name=self.config.study_name,
                storage=self.config.storage,
                sampler=sampler,
                pruner=pruner,
                directions=["maximize"] * len(self.config.objectives),
                load_if_exists=self.config.resume
            )
        else:
            study = optuna.create_study(
                study_name=self.config.study_name,
                storage=self.config.storage,
                sampler=sampler,
                pruner=pruner,
                direction="maximize",
                load_if_exists=self.config.resume
            )
        
        # Fonction objectif
        def objective(trial):
            # Suggérer les hyperparamètres
            params = self._suggest_params_optuna(trial)
            
            # Évaluer
            if self.config.multi_objective:
                values = self._evaluate_params_multi(params, training_data, validation_data, base_config, trial)
                return values
            else:
                value = self._evaluate_params(params, training_data, validation_data, base_config, trial)
                return value
        
        # Callbacks
        callbacks = []
        if self.config.save_interval > 0:
            callbacks.append(self._create_save_callback())
        if self.config.generate_plots and self.config.plot_interval > 0:
            callbacks.append(self._create_plot_callback(study))
        
        # Optimiser
        if self.config.distributed:
            # Optimisation distribuée
            study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                n_jobs=self.config.max_concurrent_trials,
                callbacks=callbacks
            )
        else:
            # Optimisation séquentielle ou parallèle locale
            study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.timeout,
                callbacks=callbacks
            )
        
        # Extraire les résultats
        if self.config.multi_objective:
            results = self._extract_results_multi_objective(study)
        else:
            results = self._extract_results_single_objective(study)
        
        return results
    
    def _optimize_hyperopt(self, training_data: pd.DataFrame,
                          validation_data: Optional[pd.DataFrame],
                          base_config: Any) -> OptimizationResults:
        """Optimisation avec Hyperopt"""
        
        # Définir l'espace de recherche
        space = self._create_hyperopt_space()
        
        # Fonction objectif
        def objective(params):
            # Nettoyer les paramètres
            clean_params = self._clean_hyperopt_params(params)
            
            # Évaluer
            value = self._evaluate_params(clean_params, training_data, validation_data, base_config)
            
            # Hyperopt minimise, donc on retourne le négatif
            return {'loss': -value, 'status': STATUS_OK}
        
        # Trials pour stocker l'historique
        trials = Trials()
        
        # Optimiser
        best = fmin(
            fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=self.config.n_trials,
            trials=trials,
            verbose=True
        )
        
        # Extraire les résultats
        results = self._extract_results_hyperopt(best, trials)
        
        return results
    
    def _optimize_ray_tune(self, training_data: pd.DataFrame,
                          validation_data: Optional[pd.DataFrame],
                          base_config: Any) -> OptimizationResults:
        """Optimisation avec Ray Tune"""
        
        # Configuration Ray
        ray_config = self._create_ray_config()
        
        # Fonction d'entraînement
        def train_fn(config, checkpoint_dir=None):
            # Évaluer les paramètres
            value = self._evaluate_params(config, training_data, validation_data, base_config)
            
            # Reporter à Ray
            tune.report(score=value)
        
        # Scheduler
        if self.config.enable_pruning:
            scheduler = ASHAScheduler(
                metric="score",
                mode="max",
                max_t=100,
                grace_period=self.config.n_warmup_steps
            )
        else:
            scheduler = None
        
        # Search algorithm
        if self.config.sampler == "optuna":
            search_alg = OptunaSearch(metric="score", mode="max")
        else:
            search_alg = HyperOptSearch(metric="score", mode="max")
        
        # Lancer l'optimisation
        analysis = tune.run(
            train_fn,
            config=ray_config,
            num_samples=self.config.n_trials,
            scheduler=scheduler,
            search_alg=search_alg,
            resources_per_trial={
                "cpu": self.config.cpu_per_trial,
                "gpu": self.config.gpu_per_trial
            },
            local_dir=str(self.results_dir),
            verbose=1
        )
        
        # Extraire les résultats
        results = self._extract_results_ray(analysis)
        
        return results
    
    def _optimize_skopt(self, training_data: pd.DataFrame,
                       validation_data: Optional[pd.DataFrame],
                       base_config: Any) -> OptimizationResults:
        """Optimisation avec Scikit-Optimize (Gaussian Processes)"""
        
        # Définir l'espace de recherche
        space, param_names = self._create_skopt_space()
        
        # Fonction objectif
        @use_named_args(space)
        def objective(**params):
            # Évaluer
            value = self._evaluate_params(params, training_data, validation_data, base_config)
            # Skopt minimise
            return -value
        
        # Optimiser
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=self.config.n_trials,
            n_initial_points=self.config.n_startup_trials,
            acq_func='EI',  # Expected Improvement
            verbose=True
        )
        
        # Extraire les résultats
        results = self._extract_results_skopt(result, param_names)
        
        return results
    
    def _optimize_genetic(self, training_data: pd.DataFrame,
                         validation_data: Optional[pd.DataFrame],
                         base_config: Any) -> OptimizationResults:
        """Optimisation avec algorithme génétique"""
        
        # Configuration DEAP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        toolbox = base.Toolbox()
        
        # Définir les gènes
        self._register_genetic_operators(toolbox)
        
        # Fonction d'évaluation
        def evaluate(individual):
            params = self._decode_genetic_individual(individual)
            value = self._evaluate_params(params, training_data, validation_data, base_config)
            return (value,)
        
        toolbox.register("evaluate", evaluate)
        
        # Opérateurs génétiques
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        # Population initiale
        population = toolbox.population(n=50)
        
        # Statistiques
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Hall of fame
        hof = tools.HallOfFame(10)
        
        # Évolution
        population, log = algorithms.eaMuPlusLambda(
            population, toolbox,
            mu=50, lambda_=100,
            cxpb=0.5, mutpb=0.2,
            ngen=self.config.n_trials // 100,
            stats=stats, halloffame=hof,
            verbose=True
        )
        
        # Extraire les résultats
        results = self._extract_results_genetic(hof, log)
        
        return results
    
    def _suggest_params_optuna(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggérer des paramètres avec Optuna"""
        params = {}
        
        # Parcourir l'espace de recherche
        for param_name, param_config in self.config.search_space.items():
            param_type = param_config['type']
            
            if param_type == 'float':
                params[param_name] = trial.suggest_float(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'int':
                params[param_name] = trial.suggest_int(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    log=param_config.get('log', False)
                )
            elif param_type == 'categorical':
                params[param_name] = trial.suggest_categorical(
                    param_name,
                    param_config['choices']
                )
            elif param_type == 'discrete_uniform':
                params[param_name] = trial.suggest_discrete_uniform(
                    param_name,
                    param_config['min'],
                    param_config['max'],
                    param_config['step']
                )
        
        return params
    
    def _evaluate_params(self, params: Dict[str, Any],
                        training_data: pd.DataFrame,
                        validation_data: Optional[pd.DataFrame],
                        base_config: Any,
                        trial: Optional[Any] = None) -> float:
        """Évaluer un ensemble de paramètres"""
        
        # Vérifier le cache
        param_hash = str(sorted(params.items()))
        if param_hash in self.evaluation_cache:
            return self.evaluation_cache[param_hash]
        
        try:
            # Créer la configuration avec les nouveaux paramètres
            if isinstance(base_config, TrainingConfig):
                config = self._create_training_config(params, base_config)
                value = self._evaluate_training_config(config, training_data, validation_data, trial)
            else:
                config = self._create_backtest_config(params, base_config)
                value = self._evaluate_backtest_config(config, training_data, trial)
            
            # Mettre en cache
            self.evaluation_cache[param_hash] = value
            
            return value
            
        except Exception as e:
            logger.error(f"Erreur lors de l'évaluation: {e}")
            return -np.inf
    
    def _evaluate_training_config(self, config: TrainingConfig,
                                training_data: pd.DataFrame,
                                validation_data: Optional[pd.DataFrame],
                                trial: Optional[Any] = None) -> float:
        """Évaluer une configuration d'entraînement"""
        
        # Cross-validation si pas de données de validation
        if validation_data is None and self.config.n_splits > 1:
            scores = []
            fold_size = len(training_data) // self.config.n_splits
            
            for i in range(self.config.n_splits):
                # Diviser les données
                val_start = i * fold_size
                val_end = (i + 1) * fold_size if i < self.config.n_splits - 1 else len(training_data)
                
                train_fold = pd.concat([
                    training_data.iloc[:val_start],
                    training_data.iloc[val_end:]
                ])
                val_fold = training_data.iloc[val_start:val_end]
                
                # Entraîner et évaluer
                score = self._train_and_evaluate(config, train_fold, val_fold, trial)
                scores.append(score)
                
                # Pruning intermédiaire
                if trial and hasattr(trial, 'report'):
                    trial.report(np.mean(scores), i)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            
            return np.mean(scores)
        
        else:
            # Évaluation simple
            return self._train_and_evaluate(config, training_data, validation_data, trial)
    
    def _train_and_evaluate(self, config: TrainingConfig,
                          train_data: pd.DataFrame,
                          val_data: pd.DataFrame,
                          trial: Optional[Any] = None) -> float:
        """Entraîner et évaluer un modèle"""
        
        # Créer le trainer
        trainer = DRLTrainer(config)
        
        # Configuration d'entraînement réduite pour l'optimisation
        config.episodes = min(config.episodes, 200)  # Limiter pour accélérer
        
        # Entraîner
        results = trainer.train(pd.concat([train_data, val_data]))
        
        # Extraire la métrique d'évaluation
        if self.config.validation_metric == "sharpe_ratio":
            return results['best_sharpe_ratio']
        elif self.config.validation_metric == "total_return":
            return results['test_results']['total_return']
        elif self.config.validation_metric == "max_drawdown":
            return -abs(results['test_results']['max_drawdown'])
        else:
            return results['test_results'].get(self.config.validation_metric, 0)
    
    def _evaluate_backtest_config(self, config: BacktestConfig,
                                data: pd.DataFrame,
                                trial: Optional[Any] = None) -> float:
        """Évaluer une configuration de backtesting"""
        
        # Créer une stratégie simple pour le test
        # (Dans la pratique, cela devrait être votre stratégie réelle)
        from ...strategies.base_strategy import BaseStrategy
        
        class TestStrategy(BaseStrategy):
            def generate_signal(self, data):
                # Stratégie momentum simple
                if len(data) < 20:
                    return 0
                returns = data['close'].pct_change()
                if returns.iloc[-1] > returns.rolling(20).mean().iloc[-1]:
                    return 1
                else:
                    return -1
        
        strategy = TestStrategy()
        
        # Backtester
        backtester = AdvancedBacktester(config)
        results = backtester.backtest(strategy, data)
        
        # Retourner la métrique
        if self.config.validation_metric == "sharpe_ratio":
            return results.sharpe_ratio
        elif self.config.validation_metric == "total_return":
            return results.total_return
        elif self.config.validation_metric == "max_drawdown":
            return -abs(results.max_drawdown)
        else:
            return getattr(results, self.config.validation_metric, 0)
    
    def _create_training_config(self, params: Dict[str, Any],
                              base_config: TrainingConfig) -> TrainingConfig:
        """Créer une configuration d'entraînement avec les nouveaux paramètres"""
        config_dict = base_config.to_dict()
        config_dict.update(params)
        return TrainingConfig(**config_dict)
    
    def _create_backtest_config(self, params: Dict[str, Any],
                              base_config: BacktestConfig) -> BacktestConfig:
        """Créer une configuration de backtest avec les nouveaux paramètres"""
        config_dict = asdict(base_config)
        config_dict.update(params)
        return BacktestConfig(**config_dict)
    
    def _create_optuna_sampler(self):
        """Créer le sampler Optuna"""
        if self.config.sampler == "tpe":
            return TPESampler(
                n_startup_trials=self.config.n_startup_trials,
                n_ei_candidates=24
            )
        elif self.config.sampler == "cmaes":
            return CmaEsSampler(
                n_startup_trials=self.config.n_startup_trials
            )
        elif self.config.sampler == "random":
            return optuna.samplers.RandomSampler()
        elif self.config.sampler == "grid":
            return optuna.samplers.GridSampler(self.config.search_space)
        else:
            return TPESampler()
    
    def _create_optuna_pruner(self):
        """Créer le pruner Optuna"""
        if not self.config.enable_pruning:
            return None
        
        if self.config.pruner == "median":
            return MedianPruner(
                n_startup_trials=self.config.n_startup_trials,
                n_warmup_steps=self.config.n_warmup_steps
            )
        elif self.config.pruner == "hyperband":
            return HyperbandPruner(
                min_resource=1,
                max_resource=self.config.n_trials,
                reduction_factor=3
            )
        elif self.config.pruner == "threshold":
            return ThresholdPruner(upper=0.5)
        else:
            return MedianPruner()
    
    def _extract_results_single_objective(self, study: optuna.Study) -> OptimizationResults:
        """Extraire les résultats pour optimisation mono-objectif"""
        
        # Meilleur trial
        best_trial = study.best_trial
        
        # Historique des trials
        trials_history = []
        for trial in study.trials:
            trial_dict = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start,
                'datetime_complete': trial.datetime_complete
            }
            trials_history.append(trial_dict)
        
        # Importance des paramètres
        try:
            importance = optuna.importance.get_param_importances(study)
        except:
            importance = {}
        
        # Statistiques
        n_trials_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        n_trials_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        
        return OptimizationResults(
            best_params=best_trial.params,
            best_value=best_trial.value,
            best_trial_number=best_trial.number,
            trials_history=trials_history,
            param_importance=importance,
            n_trials_completed=n_trials_completed,
            n_trials_pruned=n_trials_pruned,
            optimization_time=0  # Sera mis à jour plus tard
        )
    
    def _extract_results_multi_objective(self, study: optuna.Study) -> OptimizationResults:
        """Extraire les résultats pour optimisation multi-objectif"""
        
        # Front de Pareto
        pareto_front = []
        for trial in study.best_trials:
            pareto_point = {
                'params': trial.params,
                'values': trial.values,
                'number': trial.number
            }
            pareto_front.append(pareto_point)
        
        # Choisir le "meilleur" selon un critère
        # Par exemple, maximiser la somme pondérée des objectifs
        best_trial = max(study.best_trials, key=lambda t: sum(t.values))
        
        # Le reste est similaire à mono-objectif
        results = self._extract_results_single_objective(study)
        results.pareto_front = pareto_front
        results.best_value = best_trial.values[0]  # Premier objectif
        
        return results
    
    def _generate_visualizations(self, results: OptimizationResults):
        """Générer les visualisations des résultats"""
        
        # 1. Historique d'optimisation
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Valeurs des objectifs
        trial_numbers = [t['number'] for t in results.trials_history if t['value'] is not None]
        trial_values = [t['value'] for t in results.trials_history if t['value'] is not None]
        
        ax1.plot(trial_numbers, trial_values, 'b-', alpha=0.5, label='All trials')
        
        # Meilleure valeur cumulative
        best_values = []
        current_best = -np.inf
        for v in trial_values:
            if v > current_best:
                current_best = v
            best_values.append(current_best)
        
        ax1.plot(trial_numbers, best_values, 'r-', linewidth=2, label='Best so far')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Optimization History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Distribution des valeurs
        ax2.hist(trial_values, bins=30, alpha=0.7, color='green')
        ax2.axvline(results.best_value, color='red', linestyle='--', 
                   label=f'Best: {results.best_value:.4f}')
        ax2.set_xlabel('Objective Value')
        ax2.set_ylabel('Count')
        ax2.set_title('Objective Value Distribution')
        ax2.legend()
        
        plt.tight_layout()
        convergence_path = self.results_dir / 'convergence_plot.png'
        plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
        plt.close()
        results.convergence_plot = str(convergence_path)
        
        # 2. Importance des paramètres
        if results.param_importance:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            params = list(results.param_importance.keys())
            importances = list(results.param_importance.values())
            
            y_pos = np.arange(len(params))
            ax.barh(y_pos, importances)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(params)
            ax.set_xlabel('Importance')
            ax.set_title('Parameter Importance')
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            importance_path = self.results_dir / 'param_importance.png'
            plt.savefig(importance_path, dpi=300, bbox_inches='tight')
            plt.close()
            results.param_importance_plot = str(importance_path)
        
        # 3. Coordonnées parallèles (pour multi-dimensionnel)
        if len(results.best_params) > 2:
            self._plot_parallel_coordinates(results)
        
        # 4. Front de Pareto (si multi-objectif)
        if results.pareto_front:
            self._plot_pareto_front(results)
    
    def _plot_parallel_coordinates(self, results: OptimizationResults):
        """Graphique en coordonnées parallèles"""
        
        # Préparer les données
        data = []
        for trial in results.trials_history:
            if trial['value'] is not None and trial['state'] == 'COMPLETE':
                row = trial['params'].copy()
                row['objective'] = trial['value']
                data.append(row)
        
        if not data:
            return
        
        df = pd.DataFrame(data)
        
        # Créer le graphique avec plotly
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=df['objective'],
                    colorscale='Viridis',
                    showscale=True,
                    reversescale=True,
                    cmin=df['objective'].min(),
                    cmax=df['objective'].max()
                ),
                dimensions=[
                    dict(
                        range=[df[col].min(), df[col].max()],
                        label=col,
                        values=df[col]
                    ) for col in df.columns
                ]
            )
        )
        
        fig.update_layout(
            title="Parallel Coordinates Plot",
            width=1200,
            height=600
        )
        
        parallel_path = self.results_dir / 'parallel_coordinates.html'
        fig.write_html(str(parallel_path))
        results.parallel_coordinate_plot = str(parallel_path)
    
    def _plot_pareto_front(self, results: OptimizationResults):
        """Visualiser le front de Pareto"""
        
        if len(results.pareto_front[0]['values']) != 2:
            return  # Ne visualiser que pour 2 objectifs
        
        # Extraire les valeurs
        obj1_values = [p['values'][0] for p in results.pareto_front]
        obj2_values = [p['values'][1] for p in results.pareto_front]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Tous les points
        all_obj1 = [t['values'][0] for t in results.trials_history 
                   if t.get('values') and len(t['values']) >= 2]
        all_obj2 = [t['values'][1] for t in results.trials_history 
                   if t.get('values') and len(t['values']) >= 2]
        
        ax.scatter(all_obj1, all_obj2, alpha=0.3, label='All solutions')
        ax.scatter(obj1_values, obj2_values, color='red', s=100, 
                  label='Pareto front', zorder=5)
        
        # Relier les points du front
        sorted_indices = np.argsort(obj1_values)
        sorted_obj1 = np.array(obj1_values)[sorted_indices]
        sorted_obj2 = np.array(obj2_values)[sorted_indices]
        ax.plot(sorted_obj1, sorted_obj2, 'r--', alpha=0.5)
        
        ax.set_xlabel(f'Objective 1: {self.config.objectives[0]}')
        ax.set_ylabel(f'Objective 2: {self.config.objectives[1]}')
        ax.set_title('Pareto Front')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pareto_path = self.results_dir / 'pareto_front.png'
        plt.savefig(pareto_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_results(self, results: OptimizationResults):
        """Sauvegarder les résultats de l'optimisation"""
        
        # JSON pour les métriques de base
        json_path = self.results_dir / 'optimization_results.json'
        with open(json_path, 'w') as f:
            json_dict = {
                'best_params': results.best_params,
                'best_value': results.best_value,
                'best_trial_number': results.best_trial_number,
                'n_trials_completed': results.n_trials_completed,
                'n_trials_pruned': results.n_trials_pruned,
                'optimization_time': results.optimization_time,
                'param_importance': results.param_importance
            }
            json.dump(json_dict, f, indent=2)
        
        # Pickle pour l'objet complet
        pickle_path = self.results_dir / 'optimization_results.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        # CSV pour l'historique des trials
        if results.trials_history:
            trials_df = pd.DataFrame(results.trials_history)
            trials_df.to_csv(self.results_dir / 'trials_history.csv', index=False)
        
        logger.info(f"Résultats sauvegardés dans {self.results_dir}")


# Espaces de recherche prédéfinis pour différents modèles
PREDEFINED_SEARCH_SPACES = {
    'dqn': {
        'learning_rate': {'type': 'float', 'min': 1e-5, 'max': 1e-2, 'log': True},
        'batch_size': {'type': 'categorical', 'choices': [32, 64, 128, 256]},
        'gamma': {'type': 'float', 'min': 0.9, 'max': 0.999},
        'epsilon_decay': {'type': 'float', 'min': 0.99, 'max': 0.999},
        'memory_size': {'type': 'int', 'min': 10000, 'max': 200000},
        'hidden_layers': {
            'type': 'categorical',
            'choices': [[256, 128], [512, 256, 128], [256, 256, 128, 64]]
        },
        'dropout_rate': {'type': 'float', 'min': 0.0, 'max': 0.5}
    },
    
    'ppo': {
        'learning_rate': {'type': 'float', 'min': 1e-5, 'max': 1e-3, 'log': True},
        'n_steps': {'type': 'int', 'min': 128, 'max': 2048},
        'batch_size': {'type': 'categorical', 'choices': [32, 64, 128]},
        'n_epochs': {'type': 'int', 'min': 3, 'max': 30},
        'gamma': {'type': 'float', 'min': 0.9, 'max': 0.999},
        'gae_lambda': {'type': 'float', 'min': 0.9, 'max': 1.0},
        'clip_range': {'type': 'float', 'min': 0.1, 'max': 0.4},
        'vf_coef': {'type': 'float', 'min': 0.1, 'max': 1.0},
        'ent_coef': {'type': 'float', 'min': 0.0, 'max': 0.1}
    },
    
    'trading_strategy': {
        'lookback_period': {'type': 'int', 'min': 10, 'max': 100},
        'threshold': {'type': 'float', 'min': 0.001, 'max': 0.05},
        'stop_loss': {'type': 'float', 'min': 0.01, 'max': 0.1},
        'take_profit': {'type': 'float', 'min': 0.01, 'max': 0.2},
        'position_size': {'type': 'float', 'min': 0.1, 'max': 1.0},
        'max_positions': {'type': 'int', 'min': 1, 'max': 10}
    }
}


# Fonction utilitaire pour optimisation rapide
def quick_optimize(model_type: str, training_data: pd.DataFrame,
                  n_trials: int = 50) -> Dict[str, Any]:
    """
    Optimisation rapide avec configuration prédéfinie
    
    Args:
        model_type: Type de modèle ('dqn', 'ppo', 'trading_strategy')
        training_data: Données d'entraînement
        n_trials: Nombre d'essais
        
    Returns:
        Meilleurs hyperparamètres
    """
    # Configuration
    config = OptimizationConfig(
        optimizer="optuna",
        n_trials=n_trials,
        search_space=PREDEFINED_SEARCH_SPACES.get(model_type, {}),
        objectives=["sharpe_ratio"],
        enable_pruning=True
    )
    
    # Optimiser
    optimizer = HyperparameterOptimizer(config)
    results = optimizer.optimize(training_data)
    
    return results.best_params


# Exemple d'utilisation
def main():
    """Exemple d'utilisation de l'optimiseur"""
    
    # Données de test
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='1h')
    data = pd.DataFrame({
        'open': np.random.randn(len(dates)).cumsum() + 100,
        'high': np.random.randn(len(dates)).cumsum() + 101,
        'low': np.random.randn(len(dates)).cumsum() + 99,
        'close': np.random.randn(len(dates)).cumsum() + 100,
        'volume': np.random.exponential(1000, len(dates))
    }, index=dates)
    
    # Configuration de l'espace de recherche
    search_space = {
        'learning_rate': {'type': 'float', 'min': 1e-5, 'max': 1e-2, 'log': True},
        'batch_size': {'type': 'categorical', 'choices': [32, 64, 128, 256]},
        'gamma': {'type': 'float', 'min': 0.9, 'max': 0.999},
        'hidden_layers': {
            'type': 'categorical',
            'choices': [[256, 128], [512, 256], [256, 128, 64]]
        }
    }
    
    # Configuration de l'optimisation
    config = OptimizationConfig(
        optimizer="optuna",
        n_trials=100,
        search_space=search_space,
        objectives=["sharpe_ratio"],
        multi_objective=False,
        enable_pruning=True,
        distributed=False
    )
    
    # Créer l'optimiseur
    optimizer = HyperparameterOptimizer(config)
    
    # Configuration de base pour le modèle
    from ..trainer import TrainingConfig
    base_config = TrainingConfig(
        model_type="dqn",
        episodes=200,
        initial_balance=100000
    )
    
    # Optimiser
    results = optimizer.optimize(data, base_config=base_config)
    
    print(f"Meilleurs paramètres: {results.best_params}")
    print(f"Meilleure valeur: {results.best_value:.4f}")
    print(f"Trials complétés: {results.n_trials_completed}")
    print(f"Trials élagués: {results.n_trials_pruned}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()