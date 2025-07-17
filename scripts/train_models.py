#!/usr/bin/env python3
"""
Script d'entraînement des modèles de Machine Learning et Deep Reinforcement Learning
pour le système de trading algorithmique haute performance.

Ce script orchestre l'entraînement des différents modèles IA du système :
- Deep Q-Network (DQN)
- Proximal Policy Optimization (PPO)
- Soft Actor-Critic (SAC)
- Ensemble d'agents
- Optimisation d'hyperparamètres

Usage:
    python scripts/train_models.py --config config/training_config.yaml
    python scripts/train_models.py --model dqn --symbol BTCUSDT --timesteps 100000
    python scripts/train_models.py --ensemble --optimize --parallel 4
"""

import argparse
import asyncio
import logging
import multiprocessing
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import warnings

# Ajout du chemin racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppression des warnings non critiques
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Imports des modules du système
from config.settings import (
    TRAINING_CONFIG, MODEL_PATHS, DATA_PATHS, 
    LOGGING_CONFIG, PERFORMANCE_TARGETS
)
from ml.models.dqn import DQNAgent
from ml.models.ppo import PPOAgent
from ml.models.sac import SACAgent
from ml.models.ensemble_agent import EnsembleAgent
from ml.training.trainer import ModelTrainer
from ml.training.backtesting import AdvancedBacktester
from ml.training.hyperopt import HyperparameterOptimizer
from ml.environments.trading_env import TradingEnvironment
from ml.environments.multi_asset_env import MultiAssetEnvironment
from ml.features.feature_engineering import FeatureEngineer
from ml.features.market_regime import MarketRegimeDetector
from data.collectors.multi_exchange import MultiExchangeCollector
from data.processors.data_normalizer import DataNormalizer
from data.storage.data_manager import DataManager
from utils.logger import setup_logger, get_logger
from utils.metrics import ModelMetrics
from utils.helpers import save_json, load_json, create_directories
from monitoring.performance_tracker import PerformanceTracker

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = get_logger(__name__)


class TrainingOrchestrator:
    """
    Orchestrateur principal pour l'entraînement des modèles IA.
    
    Gère l'entraînement séquentiel ou parallèle des différents modèles,
    l'optimisation des hyperparamètres, le backtesting et la validation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise l'orchestrateur d'entraînement.
        
        Args:
            config: Configuration d'entraînement
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.TrainingOrchestrator")
        
        # Composants principaux
        self.data_manager = None
        self.feature_engineer = None
        self.regime_detector = None
        self.environments = {}
        self.models = {}
        self.trainers = {}
        self.backtester = None
        self.optimizer = None
        self.metrics = ModelMetrics()
        self.performance_tracker = PerformanceTracker()
        
        # Résultats d'entraînement
        self.training_results = {}
        self.best_models = {}
        self.performance_reports = {}
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialise tous les composants nécessaires."""
        try:
            self.logger.info("Initialisation des composants d'entraînement...")
            
            # Gestionnaire de données
            self.data_manager = DataManager(self.config.get('data', {}))
            
            # Ingénierie des features
            self.feature_engineer = FeatureEngineer(
                self.config.get('features', {})
            )
            
            # Détecteur de régimes de marché
            self.regime_detector = MarketRegimeDetector(
                self.config.get('regime_detection', {})
            )
            
            # Backtester avancé
            self.backtester = AdvancedBacktester(
                self.config.get('backtesting', {})
            )
            
            # Optimisateur d'hyperparamètres
            self.optimizer = HyperparameterOptimizer(
                self.config.get('optimization', {})
            )
            
            self.logger.info("Composants initialisés avec succès")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation : {e}")
            raise
    
    async def prepare_data(self, symbols: List[str], timeframe: str = "1h") -> Dict[str, Any]:
        """
        Prépare les données d'entraînement pour tous les symboles.
        
        Args:
            symbols: Liste des symboles à traiter
            timeframe: Timeframe des données
            
        Returns:
            Données préparées pour l'entraînement
        """
        self.logger.info(f"Préparation des données pour {len(symbols)} symboles...")
        
        try:
            # Collecte des données
            end_date = datetime.now()
            start_date = end_date - timedelta(
                days=self.config.get('data_period_days', 365)
            )
            
            all_data = {}
            
            for symbol in symbols:
                self.logger.info(f"Collecte des données pour {symbol}...")
                
                # Récupération des données de marché
                raw_data = await self.data_manager.get_historical_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if raw_data is None or raw_data.empty:
                    self.logger.warning(f"Pas de données disponibles pour {symbol}")
                    continue
                
                # Ingénierie des features
                features = self.feature_engineer.create_features(raw_data)
                
                # Détection des régimes de marché
                regimes = self.regime_detector.detect_regimes(raw_data)
                features['market_regime'] = regimes
                
                # Normalisation
                normalizer = DataNormalizer()
                normalized_data = normalizer.normalize(features)
                
                all_data[symbol] = {
                    'raw_data': raw_data,
                    'features': features,
                    'normalized_data': normalized_data,
                    'regimes': regimes
                }
                
                self.logger.info(f"Données préparées pour {symbol}: {len(raw_data)} points")
            
            self.logger.info(f"Préparation terminée pour {len(all_data)} symboles")
            return all_data
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la préparation des données : {e}")
            raise
    
    def create_environments(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crée les environnements de trading pour l'entraînement.
        
        Args:
            data: Données préparées
            
        Returns:
            Environnements de trading créés
        """
        self.logger.info("Création des environnements de trading...")
        
        environments = {}
        
        try:
            # Environnement single-asset pour chaque symbole
            for symbol, symbol_data in data.items():
                env_config = {
                    **self.config.get('environment', {}),
                    'symbol': symbol,
                    'data': symbol_data['normalized_data'],
                    'regimes': symbol_data['regimes']
                }
                
                environments[f"single_{symbol}"] = TradingEnvironment(env_config)
                self.logger.info(f"Environnement créé pour {symbol}")
            
            # Environnement multi-assets
            if len(data) > 1:
                multi_env_config = {
                    **self.config.get('multi_environment', {}),
                    'symbols': list(data.keys()),
                    'data': {k: v['normalized_data'] for k, v in data.items()},
                    'regimes': {k: v['regimes'] for k, v in data.items()}
                }
                
                environments['multi_asset'] = MultiAssetEnvironment(multi_env_config)
                self.logger.info("Environnement multi-assets créé")
            
            self.environments = environments
            return environments
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la création des environnements : {e}")
            raise
    
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialise tous les modèles d'IA.
        
        Returns:
            Modèles initialisés
        """
        self.logger.info("Initialisation des modèles IA...")
        
        models = {}
        
        try:
            model_configs = self.config.get('models', {})
            
            # DQN Agent
            if model_configs.get('dqn', {}).get('enabled', True):
                dqn_config = {
                    **model_configs.get('dqn', {}),
                    'observation_space': self._get_observation_space(),
                    'action_space': self._get_action_space()
                }
                models['dqn'] = DQNAgent(dqn_config)
                self.logger.info("Modèle DQN initialisé")
            
            # PPO Agent
            if model_configs.get('ppo', {}).get('enabled', True):
                ppo_config = {
                    **model_configs.get('ppo', {}),
                    'observation_space': self._get_observation_space(),
                    'action_space': self._get_action_space()
                }
                models['ppo'] = PPOAgent(ppo_config)
                self.logger.info("Modèle PPO initialisé")
            
            # SAC Agent
            if model_configs.get('sac', {}).get('enabled', True):
                sac_config = {
                    **model_configs.get('sac', {}),
                    'observation_space': self._get_observation_space(),
                    'action_space': self._get_action_space()
                }
                models['sac'] = SACAgent(sac_config)
                self.logger.info("Modèle SAC initialisé")
            
            # Ensemble Agent
            if model_configs.get('ensemble', {}).get('enabled', False) and len(models) > 1:
                ensemble_config = {
                    **model_configs.get('ensemble', {}),
                    'base_models': list(models.keys())
                }
                models['ensemble'] = EnsembleAgent(ensemble_config)
                self.logger.info("Modèle Ensemble initialisé")
            
            self.models = models
            return models
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation des modèles : {e}")
            raise
    
    def _get_observation_space(self) -> int:
        """Retourne la taille de l'espace d'observation."""
        return self.config.get('environment', {}).get('observation_space', 50)
    
    def _get_action_space(self) -> int:
        """Retourne la taille de l'espace d'action."""
        return self.config.get('environment', {}).get('action_space', 3)
    
    async def train_model(
        self, 
        model_name: str, 
        environment_name: str,
        timesteps: int = None
    ) -> Dict[str, Any]:
        """
        Entraîne un modèle spécifique.
        
        Args:
            model_name: Nom du modèle à entraîner
            environment_name: Nom de l'environnement
            timesteps: Nombre de pas d'entraînement
            
        Returns:
            Résultats d'entraînement
        """
        self.logger.info(f"Entraînement du modèle {model_name} sur {environment_name}...")
        
        try:
            model = self.models[model_name]
            environment = self.environments[environment_name]
            
            # Configuration de l'entraînement
            training_config = {
                **self.config.get('training', {}),
                'timesteps': timesteps or self.config.get('training', {}).get('timesteps', 100000),
                'model_name': model_name,
                'environment_name': environment_name
            }
            
            # Création du trainer
            trainer = ModelTrainer(model, environment, training_config)
            self.trainers[f"{model_name}_{environment_name}"] = trainer
            
            # Entraînement
            start_time = time.time()
            results = await trainer.train()
            training_time = time.time() - start_time
            
            # Ajout des métriques de performance
            results['training_time'] = training_time
            results['timesteps_per_second'] = results['timesteps'] / training_time
            
            # Sauvegarde du modèle
            model_path = self._save_model(model, model_name, environment_name)
            results['model_path'] = model_path
            
            # Mise à jour des résultats
            key = f"{model_name}_{environment_name}"
            self.training_results[key] = results
            
            self.logger.info(
                f"Entraînement terminé pour {model_name}: "
                f"Reward final: {results.get('final_reward', 'N/A')}, "
                f"Temps: {training_time:.2f}s"
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement de {model_name} : {e}")
            raise
    
    async def train_all_models(self, parallel: bool = False) -> Dict[str, Any]:
        """
        Entraîne tous les modèles.
        
        Args:
            parallel: Si True, entraîne les modèles en parallèle
            
        Returns:
            Résultats d'entraînement de tous les modèles
        """
        self.logger.info(f"Entraînement de tous les modèles (parallel={parallel})...")
        
        all_results = {}
        
        try:
            # Préparation des tâches d'entraînement
            training_tasks = []
            
            for model_name in self.models.keys():
                for env_name in self.environments.keys():
                    # Eviter l'entraînement ensemble sur single-asset si pas pertinent
                    if model_name == 'ensemble' and 'single_' in env_name:
                        continue
                    
                    training_tasks.append((model_name, env_name))
            
            if parallel and len(training_tasks) > 1:
                # Entraînement parallèle
                self.logger.info(f"Lancement de {len(training_tasks)} entraînements en parallèle")
                
                tasks = [
                    self.train_model(model_name, env_name) 
                    for model_name, env_name in training_tasks
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    model_name, env_name = training_tasks[i]
                    key = f"{model_name}_{env_name}"
                    
                    if isinstance(result, Exception):
                        self.logger.error(f"Erreur pour {key}: {result}")
                        all_results[key] = {'error': str(result)}
                    else:
                        all_results[key] = result
            
            else:
                # Entraînement séquentiel
                for model_name, env_name in training_tasks:
                    try:
                        result = await self.train_model(model_name, env_name)
                        all_results[f"{model_name}_{env_name}"] = result
                    except Exception as e:
                        self.logger.error(f"Erreur pour {model_name}_{env_name}: {e}")
                        all_results[f"{model_name}_{env_name}"] = {'error': str(e)}
            
            self.logger.info(f"Entraînement terminé pour {len(all_results)} combinaisons")
            return all_results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'entraînement global : {e}")
            raise
    
    async def optimize_hyperparameters(
        self, 
        model_name: str, 
        n_trials: int = 50
    ) -> Dict[str, Any]:
        """
        Optimise les hyperparamètres d'un modèle.
        
        Args:
            model_name: Nom du modèle à optimiser
            n_trials: Nombre d'essais d'optimisation
            
        Returns:
            Meilleurs hyperparamètres trouvés
        """
        self.logger.info(f"Optimisation des hyperparamètres pour {model_name}...")
        
        try:
            model = self.models[model_name]
            environment = self.environments[list(self.environments.keys())[0]]
            
            # Configuration de l'optimisation
            optimization_config = {
                **self.config.get('optimization', {}),
                'n_trials': n_trials,
                'model_name': model_name
            }
            
            # Optimisation
            best_params = await self.optimizer.optimize(
                model, environment, optimization_config
            )
            
            self.logger.info(f"Hyperparamètres optimisés pour {model_name}: {best_params}")
            
            # Mise à jour du modèle avec les meilleurs paramètres
            self.models[model_name].update_hyperparameters(best_params)
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'optimisation de {model_name} : {e}")
            raise
    
    async def backtest_models(self) -> Dict[str, Any]:
        """
        Effectue le backtesting de tous les modèles entraînés.
        
        Returns:
            Résultats de backtesting
        """
        self.logger.info("Backtesting des modèles entraînés...")
        
        backtest_results = {}
        
        try:
            for key, training_result in self.training_results.items():
                if 'error' in training_result:
                    continue
                
                model_name, env_name = key.split('_', 1)
                
                self.logger.info(f"Backtesting de {key}...")
                
                # Configuration du backtesting
                backtest_config = {
                    **self.config.get('backtesting', {}),
                    'model_path': training_result.get('model_path'),
                    'model_name': model_name,
                    'environment_name': env_name
                }
                
                # Exécution du backtest
                result = await self.backtester.run_backtest(backtest_config)
                backtest_results[key] = result
                
                # Calcul des métriques de performance
                metrics = self.metrics.calculate_metrics(result)
                backtest_results[key]['metrics'] = metrics
                
                self.logger.info(
                    f"Backtest {key}: "
                    f"Sharpe: {metrics.get('sharpe_ratio', 'N/A'):.3f}, "
                    f"Return: {metrics.get('total_return', 'N/A'):.2%}"
                )
            
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Erreur lors du backtesting : {e}")
            raise
    
    def _save_model(self, model, model_name: str, env_name: str) -> str:
        """
        Sauvegarde un modèle entraîné.
        
        Args:
            model: Modèle à sauvegarder
            model_name: Nom du modèle
            env_name: Nom de l'environnement
            
        Returns:
            Chemin de sauvegarde
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_dir = Path(MODEL_PATHS['trained_models'])
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = model_dir / f"{model_name}_{env_name}_{timestamp}"
            model.save(str(model_path))
            
            self.logger.info(f"Modèle sauvegardé: {model_path}")
            return str(model_path)
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde du modèle : {e}")
            raise
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Génère un rapport complet de l'entraînement.
        
        Returns:
            Rapport complet
        """
        self.logger.info("Génération du rapport d'entraînement...")
        
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'config': self.config,
                'training_results': self.training_results,
                'performance_reports': self.performance_reports,
                'summary': {
                    'total_models_trained': len([
                        r for r in self.training_results.values() 
                        if 'error' not in r
                    ]),
                    'total_errors': len([
                        r for r in self.training_results.values() 
                        if 'error' in r
                    ]),
                    'best_performing_model': self._find_best_model(),
                    'total_training_time': sum([
                        r.get('training_time', 0) 
                        for r in self.training_results.values()
                        if 'error' not in r
                    ])
                }
            }
            
            # Sauvegarde du rapport
            report_path = Path('reports') / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            save_json(report, str(report_path))
            
            self.logger.info(f"Rapport sauvegardé: {report_path}")
            return report
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération du rapport : {e}")
            raise
    
    def _find_best_model(self) -> Optional[str]:
        """
        Trouve le meilleur modèle basé sur les performances.
        
        Returns:
            Nom du meilleur modèle ou None
        """
        try:
            best_model = None
            best_score = float('-inf')
            
            for key, result in self.training_results.items():
                if 'error' in result:
                    continue
                
                # Score basé sur la récompense finale et la stabilité
                score = result.get('final_reward', 0)
                if score > best_score:
                    best_score = score
                    best_model = key
            
            return best_model
            
        except Exception:
            return None


async def main():
    """Fonction principale du script d'entraînement."""
    parser = argparse.ArgumentParser(
        description="Entraînement des modèles IA pour le trading algorithmique"
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/training_config.yaml',
        help='Chemin vers le fichier de configuration'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['dqn', 'ppo', 'sac', 'ensemble', 'all'],
        default='all',
        help='Modèle à entraîner'
    )
    
    parser.add_argument(
        '--symbols', 
        type=str, 
        nargs='+',
        default=['BTCUSDT', 'ETHUSDT'],
        help='Symboles à trader'
    )
    
    parser.add_argument(
        '--timesteps', 
        type=int, 
        default=None,
        help='Nombre de pas d\'entraînement'
    )
    
    parser.add_argument(
        '--optimize', 
        action='store_true',
        help='Optimiser les hyperparamètres'
    )
    
    parser.add_argument(
        '--parallel', 
        action='store_true',
        help='Entraînement parallèle'
    )
    
    parser.add_argument(
        '--backtest', 
        action='store_true',
        help='Effectuer le backtesting après entraînement'
    )
    
    parser.add_argument(
        '--timeframe', 
        type=str, 
        default='1h',
        help='Timeframe des données'
    )
    
    args = parser.parse_args()
    
    try:
        # Chargement de la configuration
        if Path(args.config).exists():
            config = load_json(args.config)
            logger.info(f"Configuration chargée depuis {args.config}")
        else:
            config = TRAINING_CONFIG
            logger.warning(f"Fichier de config non trouvé, utilisation de la config par défaut")
        
        # Création des répertoires nécessaires
        create_directories([
            'logs', 'models', 'reports', 'data/cache'
        ])
        
        # Initialisation de l'orchestrateur
        orchestrator = TrainingOrchestrator(config)
        
        # Préparation des données
        logger.info("Préparation des données d'entraînement...")
        data = await orchestrator.prepare_data(args.symbols, args.timeframe)
        
        if not data:
            logger.error("Aucune donnée disponible pour l'entraînement")
            return
        
        # Création des environnements
        environments = orchestrator.create_environments(data)
        
        # Initialisation des modèles
        models = orchestrator.initialize_models()
        
        # Optimisation des hyperparamètres si demandée
        if args.optimize:
            logger.info("Optimisation des hyperparamètres...")
            models_to_optimize = [args.model] if args.model != 'all' else list(models.keys())
            
            for model_name in models_to_optimize:
                if model_name != 'ensemble':  # L'ensemble est optimisé différemment
                    await orchestrator.optimize_hyperparameters(model_name)
        
        # Entraînement des modèles
        if args.model == 'all':
            training_results = await orchestrator.train_all_models(args.parallel)
        else:
            # Entraînement d'un modèle spécifique
            env_name = list(environments.keys())[0]
            training_results = await orchestrator.train_model(
                args.model, env_name, args.timesteps
            )
        
        # Backtesting si demandé
        if args.backtest:
            logger.info("Lancement du backtesting...")
            backtest_results = await orchestrator.backtest_models()
            orchestrator.performance_reports = backtest_results
        
        # Génération du rapport final
        report = orchestrator.generate_report()
        
        logger.info("Entraînement terminé avec succès !")
        logger.info(f"Modèles entraînés: {len([r for r in training_results.values() if 'error' not in r])}")
        logger.info(f"Meilleur modèle: {report['summary']['best_performing_model']}")
        
    except KeyboardInterrupt:
        logger.info("Entraînement interrompu par l'utilisateur")
    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    # Configuration pour éviter les erreurs d'event loop sur Windows
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Lancement du script
    asyncio.run(main())