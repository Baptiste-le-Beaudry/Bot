"""
DRL Model Trainer Module
Système d'entraînement pour les modèles de Deep Reinforcement Learning.
Supporte DQN, PPO, SAC et les stratégies d'ensemble pour maximiser la rentabilité.
"""

import os
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import ray
from ray import tune
from ray.rllib.agents import dqn, ppo, sac
import optuna

# Import des modules internes
from ..environments.trading_env import TradingEnv
from ..models.dqn import DQNAgent
from ..models.ppo import PPOAgent
from ..models.sac import SACAgent
from ..models.ensemble_agent import EnsembleAgent
from ...data.processors.data_normalizer import DataNormalizer
from ...utils.metrics import calculate_sharpe_ratio, calculate_max_drawdown

# Configuration du logger
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration pour l'entraînement"""
    # Modèle
    model_type: str = "dqn"  # dqn, ppo, sac, ensemble
    model_name: str = "trading_bot_v1"
    
    # Données
    symbol: str = "BTC/USD"
    start_date: str = "2022-01-01"
    end_date: str = "2024-01-01"
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Entraînement
    episodes: int = 1000
    max_steps_per_episode: int = 1000
    batch_size: int = 64
    learning_rate: float = 0.0001
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Architecture réseau
    hidden_layers: List[int] = None
    activation: str = "relu"
    dropout_rate: float = 0.1
    
    # Mémoire et buffer
    memory_size: int = 100000
    min_memory_size: int = 10000
    update_frequency: int = 4
    target_update_frequency: int = 1000
    
    # Optimisation
    optimizer: str = "adam"
    gradient_clip: float = 1.0
    weight_decay: float = 0.0001
    
    # Exploration
    exploration_strategy: str = "epsilon_greedy"  # epsilon_greedy, boltzmann, ucb
    temperature: float = 1.0
    
    # Récompenses
    reward_function: str = "sharpe"  # sharpe, profit, risk_adjusted
    risk_free_rate: float = 0.02
    transaction_cost: float = 0.001
    
    # Environnement
    initial_balance: float = 100000
    max_position_size: float = 0.3
    leverage: float = 1.0
    
    # Monitoring
    log_interval: int = 10
    save_interval: int = 100
    eval_interval: int = 50
    tensorboard: bool = True
    wandb_project: str = None
    
    # Resources
    num_workers: int = 4
    gpu: bool = torch.cuda.is_available()
    distributed: bool = False
    
    def __post_init__(self):
        if self.hidden_layers is None:
            self.hidden_layers = [256, 128, 64]
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class TrainingMetrics:
    """Métriques d'entraînement"""
    episode: int
    total_reward: float
    average_reward: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    epsilon: float = None
    loss: float = None
    learning_rate: float = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


class DRLTrainer:
    """
    Trainer principal pour les modèles de Deep Reinforcement Learning.
    Gère l'entraînement, la validation et l'optimisation des hyperparamètres.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialisation du trainer
        
        Args:
            config: Configuration d'entraînement
        """
        self.config = config
        
        # Créer les répertoires
        self.model_dir = Path(f"models/{config.model_name}")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.model_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Device
        self.device = torch.device("cuda" if config.gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"Utilisation du device: {self.device}")
        
        # Initialiser l'environnement
        self.env = None
        self.val_env = None
        self.test_env = None
        
        # Agent
        self.agent = None
        
        # Métriques
        self.training_metrics = []
        self.validation_metrics = []
        self.best_sharpe_ratio = -np.inf
        self.best_model_path = None
        
        # Monitoring
        self.writer = None
        if config.tensorboard:
            self.writer = SummaryWriter(log_dir=self.model_dir / "tensorboard")
        
        if config.wandb_project:
            wandb.init(project=config.wandb_project, config=config.to_dict())
        
        # Early stopping
        self.patience = 50
        self.patience_counter = 0
        
        # Distributed training
        if config.distributed:
            ray.init(ignore_reinit_error=True)
    
    def setup_environments(self, train_data: pd.DataFrame, 
                          val_data: pd.DataFrame, 
                          test_data: pd.DataFrame):
        """
        Configurer les environnements de trading
        
        Args:
            train_data: Données d'entraînement
            val_data: Données de validation
            test_data: Données de test
        """
        env_config = {
            'initial_balance': self.config.initial_balance,
            'max_position_size': self.config.max_position_size,
            'transaction_cost': self.config.transaction_cost,
            'leverage': self.config.leverage,
            'reward_function': self.config.reward_function,
            'risk_free_rate': self.config.risk_free_rate
        }
        
        # Environnement d'entraînement
        self.env = TradingEnv(train_data, **env_config)
        
        # Environnement de validation
        self.val_env = TradingEnv(val_data, **env_config)
        
        # Environnement de test
        self.test_env = TradingEnv(test_data, **env_config)
        
        logger.info(f"Environnements créés - Train: {len(train_data)}, "
                   f"Val: {len(val_data)}, Test: {len(test_data)}")
    
    def create_agent(self) -> Any:
        """Créer l'agent selon le type de modèle"""
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        
        agent_config = {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'hidden_layers': self.config.hidden_layers,
            'learning_rate': self.config.learning_rate,
            'gamma': self.config.gamma,
            'batch_size': self.config.batch_size,
            'memory_size': self.config.memory_size,
            'epsilon_start': self.config.epsilon_start,
            'epsilon_end': self.config.epsilon_end,
            'epsilon_decay': self.config.epsilon_decay,
            'device': self.device
        }
        
        if self.config.model_type == "dqn":
            agent = DQNAgent(**agent_config)
        elif self.config.model_type == "ppo":
            agent = PPOAgent(**agent_config)
        elif self.config.model_type == "sac":
            agent = SACAgent(**agent_config)
        elif self.config.model_type == "ensemble":
            agent = EnsembleAgent(
                agents=['dqn', 'ppo', 'sac'],
                **agent_config
            )
        else:
            raise ValueError(f"Type de modèle non supporté: {self.config.model_type}")
        
        logger.info(f"Agent {self.config.model_type.upper()} créé avec "
                   f"{sum(p.numel() for p in agent.parameters())} paramètres")
        
        return agent
    
    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Entraîner le modèle
        
        Args:
            data: DataFrame avec toutes les données
            
        Returns:
            Résultats d'entraînement
        """
        # Diviser les données
        train_data, val_data, test_data = self._split_data(data)
        
        # Configurer les environnements
        self.setup_environments(train_data, val_data, test_data)
        
        # Créer l'agent
        self.agent = self.create_agent()
        
        # Boucle d'entraînement principale
        logger.info(f"Début de l'entraînement pour {self.config.episodes} épisodes")
        
        for episode in range(self.config.episodes):
            # Entraîner un épisode
            metrics = self._train_episode(episode)
            self.training_metrics.append(metrics)
            
            # Logging
            if episode % self.config.log_interval == 0:
                self._log_metrics(metrics, "train")
            
            # Validation
            if episode % self.config.eval_interval == 0:
                val_metrics = self._validate()
                self.validation_metrics.append(val_metrics)
                self._log_metrics(val_metrics, "val")
                
                # Early stopping
                if val_metrics.sharpe_ratio > self.best_sharpe_ratio:
                    self.best_sharpe_ratio = val_metrics.sharpe_ratio
                    self.patience_counter = 0
                    self._save_best_model()
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        logger.info(f"Early stopping à l'épisode {episode}")
                        break
            
            # Sauvegarde régulière
            if episode % self.config.save_interval == 0:
                self._save_checkpoint(episode)
        
        # Test final
        test_results = self._test()
        
        # Résultats finaux
        results = {
            'best_sharpe_ratio': self.best_sharpe_ratio,
            'test_results': test_results,
            'training_history': self.training_metrics,
            'validation_history': self.validation_metrics,
            'model_path': self.best_model_path
        }
        
        # Sauvegarder les résultats
        self._save_results(results)
        
        # Cleanup
        if self.writer:
            self.writer.close()
        if self.config.wandb_project:
            wandb.finish()
        
        return results
    
    def _train_episode(self, episode: int) -> TrainingMetrics:
        """Entraîner un épisode"""
        state = self.env.reset()
        total_reward = 0
        losses = []
        
        for step in range(self.config.max_steps_per_episode):
            # Sélectionner une action
            action = self.agent.select_action(state)
            
            # Exécuter l'action
            next_state, reward, done, info = self.env.step(action)
            
            # Stocker la transition
            self.agent.remember(state, action, reward, next_state, done)
            
            # Mise à jour du modèle
            if len(self.agent.memory) >= self.config.min_memory_size:
                if step % self.config.update_frequency == 0:
                    loss = self.agent.train_step()
                    if loss is not None:
                        losses.append(loss)
            
            total_reward += reward
            state = next_state
            
            if done:
                break
        
        # Mise à jour du réseau cible (DQN)
        if hasattr(self.agent, 'update_target_network'):
            if episode % self.config.target_update_frequency == 0:
                self.agent.update_target_network()
        
        # Calculer les métriques
        metrics = self._calculate_episode_metrics(
            episode, total_reward, losses, self.env.get_info()
        )
        
        return metrics
    
    def _validate(self) -> TrainingMetrics:
        """Valider le modèle"""
        self.agent.eval()
        state = self.val_env.reset()
        total_reward = 0
        
        with torch.no_grad():
            for step in range(self.config.max_steps_per_episode):
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, info = self.val_env.step(action)
                total_reward += reward
                state = next_state
                
                if done:
                    break
        
        metrics = self._calculate_episode_metrics(
            -1, total_reward, [], self.val_env.get_info()
        )
        
        self.agent.train()
        return metrics
    
    def _test(self) -> Dict[str, Any]:
        """Tester le modèle final"""
        # Charger le meilleur modèle
        if self.best_model_path:
            self.agent.load(self.best_model_path)
        
        self.agent.eval()
        state = self.test_env.reset()
        total_reward = 0
        actions = []
        rewards = []
        positions = []
        
        with torch.no_grad():
            for step in range(len(self.test_env.data)):
                action = self.agent.select_action(state, training=False)
                next_state, reward, done, info = self.test_env.step(action)
                
                actions.append(action)
                rewards.append(reward)
                positions.append(info.get('position', 0))
                
                total_reward += reward
                state = next_state
                
                if done:
                    break
        
        # Analyser les résultats
        test_info = self.test_env.get_info()
        
        results = {
            'total_reward': total_reward,
            'sharpe_ratio': test_info.get('sharpe_ratio', 0),
            'max_drawdown': test_info.get('max_drawdown', 0),
            'total_return': test_info.get('total_return', 0),
            'win_rate': test_info.get('win_rate', 0),
            'profit_factor': test_info.get('profit_factor', 0),
            'total_trades': test_info.get('total_trades', 0),
            'actions': actions,
            'rewards': rewards,
            'positions': positions,
            'portfolio_value': test_info.get('portfolio_values', [])
        }
        
        return results
    
    def optimize_hyperparameters(self, data: pd.DataFrame, 
                               n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimiser les hyperparamètres avec Optuna
        
        Args:
            data: Données d'entraînement
            n_trials: Nombre d'essais
            
        Returns:
            Meilleurs hyperparamètres
        """
        def objective(trial):
            # Suggérer des hyperparamètres
            config = TrainingConfig(
                learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
                batch_size=trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
                gamma=trial.suggest_float('gamma', 0.9, 0.999),
                hidden_layers=[
                    trial.suggest_int('hidden_1', 64, 512),
                    trial.suggest_int('hidden_2', 32, 256),
                    trial.suggest_int('hidden_3', 16, 128)
                ],
                dropout_rate=trial.suggest_float('dropout_rate', 0.0, 0.5),
                epsilon_decay=trial.suggest_float('epsilon_decay', 0.99, 0.999),
                memory_size=trial.suggest_int('memory_size', 10000, 200000),
                episodes=200  # Moins d'épisodes pour l'optimisation
            )
            
            # Créer et entraîner un nouveau trainer
            trainer = DRLTrainer(config)
            results = trainer.train(data)
            
            # Retourner la métrique à optimiser (négatif car Optuna minimise)
            return -results['best_sharpe_ratio']
        
        # Créer l'étude Optuna
        study = optuna.create_study(
            study_name=f"{self.config.model_name}_hyperparam_opt",
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10)
        )
        
        # Optimiser
        study.optimize(objective, n_trials=n_trials, n_jobs=self.config.num_workers)
        
        # Résultats
        best_params = study.best_params
        best_value = -study.best_value
        
        logger.info(f"Meilleurs hyperparamètres trouvés: {best_params}")
        logger.info(f"Meilleur Sharpe ratio: {best_value:.4f}")
        
        # Sauvegarder l'étude
        study_path = self.model_dir / "optuna_study.pkl"
        with open(study_path, 'wb') as f:
            import pickle
            pickle.dump(study, f)
        
        return {
            'best_params': best_params,
            'best_sharpe_ratio': best_value,
            'study': study
        }
    
    def train_ensemble(self, data: pd.DataFrame, 
                      models: List[str] = ['dqn', 'ppo', 'sac']) -> Dict[str, Any]:
        """
        Entraîner un ensemble de modèles
        
        Args:
            data: Données d'entraînement
            models: Liste des modèles à entraîner
            
        Returns:
            Résultats de l'ensemble
        """
        ensemble_results = {}
        trained_models = {}
        
        # Entraîner chaque modèle individuellement
        for model_type in models:
            logger.info(f"Entraînement du modèle {model_type}")
            
            # Créer une config pour ce modèle
            model_config = TrainingConfig(
                model_type=model_type,
                model_name=f"{self.config.model_name}_{model_type}",
                **{k: v for k, v in self.config.to_dict().items() 
                   if k not in ['model_type', 'model_name']}
            )
            
            # Entraîner
            trainer = DRLTrainer(model_config)
            results = trainer.train(data)
            
            ensemble_results[model_type] = results
            trained_models[model_type] = trainer.agent
        
        # Créer l'agent ensemble
        logger.info("Création de l'agent ensemble")
        
        train_data, val_data, test_data = self._split_data(data)
        self.setup_environments(train_data, val_data, test_data)
        
        ensemble_agent = EnsembleAgent(
            agents=trained_models,
            voting='weighted',  # Pondération basée sur la performance
            weights={model: results['best_sharpe_ratio'] 
                    for model, results in ensemble_results.items()}
        )
        
        # Tester l'ensemble
        self.agent = ensemble_agent
        ensemble_test_results = self._test()
        
        # Résultats finaux
        results = {
            'individual_results': ensemble_results,
            'ensemble_test_results': ensemble_test_results,
            'model_weights': ensemble_agent.weights
        }
        
        # Sauvegarder l'ensemble
        ensemble_path = self.model_dir / "ensemble_model.pt"
        ensemble_agent.save(ensemble_path)
        results['ensemble_model_path'] = str(ensemble_path)
        
        return results
    
    def continuous_learning(self, live_data_stream: Callable) -> None:
        """
        Apprentissage continu avec des données en temps réel
        
        Args:
            live_data_stream: Fonction qui retourne les nouvelles données
        """
        logger.info("Démarrage de l'apprentissage continu")
        
        # Buffer pour les nouvelles données
        data_buffer = deque(maxlen=1000)
        update_frequency = 100  # Mise à jour tous les 100 points
        
        while True:
            try:
                # Obtenir les nouvelles données
                new_data = live_data_stream()
                if new_data is not None:
                    data_buffer.extend(new_data)
                
                # Mise à jour périodique
                if len(data_buffer) >= update_frequency:
                    # Convertir en DataFrame
                    df = pd.DataFrame(list(data_buffer))
                    
                    # Entraînement incrémental
                    self._incremental_training(df)
                    
                    # Vider le buffer
                    data_buffer.clear()
                
                # Pause pour éviter la surcharge
                time.sleep(1)
                
            except KeyboardInterrupt:
                logger.info("Arrêt de l'apprentissage continu")
                break
            except Exception as e:
                logger.error(f"Erreur dans l'apprentissage continu: {e}")
                time.sleep(10)
    
    def _incremental_training(self, new_data: pd.DataFrame):
        """Entraînement incrémental avec nouvelles données"""
        # Créer un environnement temporaire
        temp_env = TradingEnv(
            new_data,
            initial_balance=self.config.initial_balance,
            transaction_cost=self.config.transaction_cost
        )
        
        # Entraîner quelques épisodes
        for _ in range(10):
            state = temp_env.reset()
            
            for step in range(len(new_data)):
                action = self.agent.select_action(state)
                next_state, reward, done, _ = temp_env.step(action)
                
                self.agent.remember(state, action, reward, next_state, done)
                
                if len(self.agent.memory) >= self.config.batch_size:
                    self.agent.train_step()
                
                state = next_state
                if done:
                    break
        
        logger.info("Mise à jour incrémentale complétée")
    
    # Méthodes utilitaires
    
    def _split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Diviser les données en train/val/test"""
        n = len(data)
        train_end = int(n * (1 - self.config.validation_split - self.config.test_split))
        val_end = int(n * (1 - self.config.test_split))
        
        train_data = data.iloc[:train_end]
        val_data = data.iloc[train_end:val_end]
        test_data = data.iloc[val_end:]
        
        return train_data, val_data, test_data
    
    def _calculate_episode_metrics(self, episode: int, total_reward: float,
                                 losses: List[float], info: Dict) -> TrainingMetrics:
        """Calculer les métriques d'un épisode"""
        return TrainingMetrics(
            episode=episode,
            total_reward=total_reward,
            average_reward=total_reward / self.config.max_steps_per_episode,
            sharpe_ratio=info.get('sharpe_ratio', 0),
            max_drawdown=info.get('max_drawdown', 0),
            win_rate=info.get('win_rate', 0),
            profit_factor=info.get('profit_factor', 0),
            total_trades=info.get('total_trades', 0),
            epsilon=getattr(self.agent, 'epsilon', None),
            loss=np.mean(losses) if losses else None,
            learning_rate=self.config.learning_rate
        )
    
    def _log_metrics(self, metrics: TrainingMetrics, phase: str):
        """Logger les métriques"""
        log_str = f"[{phase.upper()}] Episode {metrics.episode}: "
        log_str += f"Reward: {metrics.total_reward:.2f}, "
        log_str += f"Sharpe: {metrics.sharpe_ratio:.4f}, "
        log_str += f"Drawdown: {metrics.max_drawdown:.4f}, "
        log_str += f"Win Rate: {metrics.win_rate:.2%}"
        
        logger.info(log_str)
        
        # TensorBoard
        if self.writer:
            for key, value in metrics.to_dict().items():
                if value is not None and isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{phase}/{key}", value, metrics.episode)
        
        # Weights & Biases
        if self.config.wandb_project:
            wandb.log({f"{phase}_{k}": v for k, v in metrics.to_dict().items() 
                      if v is not None})
    
    def _save_checkpoint(self, episode: int):
        """Sauvegarder un checkpoint"""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'config': self.config.to_dict(),
            'metrics': self.training_metrics[-1].to_dict() if self.training_metrics else None
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_ep{episode}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint sauvegardé: {checkpoint_path}")
    
    def _save_best_model(self):
        """Sauvegarder le meilleur modèle"""
        self.best_model_path = self.model_dir / "best_model.pt"
        
        model_data = {
            'model_state_dict': self.agent.state_dict(),
            'config': self.config.to_dict(),
            'sharpe_ratio': self.best_sharpe_ratio,
            'timestamp': datetime.now().isoformat()
        }
        
        torch.save(model_data, self.best_model_path)
        logger.info(f"Meilleur modèle sauvegardé: {self.best_model_path}")
    
    def _save_results(self, results: Dict[str, Any]):
        """Sauvegarder les résultats d'entraînement"""
        results_path = self.model_dir / "training_results.json"
        
        # Convertir les métriques en dict pour JSON
        results_json = {
            'best_sharpe_ratio': results['best_sharpe_ratio'],
            'test_results': results['test_results'],
            'config': self.config.to_dict(),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        # Sauvegarder aussi l'historique complet en pickle
        history_path = self.model_dir / "training_history.pkl"
        import pickle
        with open(history_path, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Résultats sauvegardés dans {self.model_dir}")
    
    def load_model(self, model_path: str):
        """Charger un modèle sauvegardé"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Recréer l'agent si nécessaire
        if self.agent is None:
            # Utiliser la config sauvegardée
            saved_config = TrainingConfig(**checkpoint['config'])
            self.config = saved_config
            
            # Créer un environnement temporaire pour obtenir les dimensions
            dummy_env = TradingEnv(pd.DataFrame())
            self.agent = self.create_agent()
        
        # Charger les poids
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.agent.to(self.device)
        
        logger.info(f"Modèle chargé depuis {model_path}")


# Fonction utilitaire pour l'entraînement distribué avec Ray
def train_distributed(config: TrainingConfig, data: pd.DataFrame, 
                     num_workers: int = 4) -> Dict[str, Any]:
    """
    Entraînement distribué avec Ray
    
    Args:
        config: Configuration d'entraînement
        data: Données d'entraînement
        num_workers: Nombre de workers
        
    Returns:
        Résultats d'entraînement
    """
    @ray.remote
    class DistributedTrainer:
        def __init__(self, config, data):
            self.trainer = DRLTrainer(config)
            self.data = data
        
        def train(self):
            return self.trainer.train(self.data)
    
    # Créer les workers
    workers = [DistributedTrainer.remote(config, data) for _ in range(num_workers)]
    
    # Entraîner en parallèle
    results = ray.get([worker.train.remote() for worker in workers])
    
    # Sélectionner le meilleur modèle
    best_idx = np.argmax([r['best_sharpe_ratio'] for r in results])
    best_result = results[best_idx]
    
    logger.info(f"Meilleur modèle (worker {best_idx}): "
               f"Sharpe ratio = {best_result['best_sharpe_ratio']:.4f}")
    
    return best_result


# Exemple d'utilisation
def main():
    """Exemple d'utilisation du trainer"""
    # Configuration
    config = TrainingConfig(
        model_type="dqn",
        model_name="btc_trading_bot",
        symbol="BTC/USD",
        episodes=1000,
        learning_rate=0.0001,
        batch_size=64,
        hidden_layers=[256, 128, 64],
        tensorboard=True
    )
    
    # Charger les données (exemple avec données simulées)
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='1h')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(len(dates)).cumsum() + 40000,
        'high': np.random.randn(len(dates)).cumsum() + 40100,
        'low': np.random.randn(len(dates)).cumsum() + 39900,
        'close': np.random.randn(len(dates)).cumsum() + 40000,
        'volume': np.random.exponential(1000, len(dates))
    })
    
    # Créer le trainer
    trainer = DRLTrainer(config)
    
    # Option 1: Entraînement simple
    results = trainer.train(data)
    print(f"Sharpe ratio final: {results['best_sharpe_ratio']:.4f}")
    
    # Option 2: Optimisation des hyperparamètres
    # best_params = trainer.optimize_hyperparameters(data, n_trials=50)
    
    # Option 3: Entraînement d'ensemble
    # ensemble_results = trainer.train_ensemble(data, models=['dqn', 'ppo', 'sac'])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()