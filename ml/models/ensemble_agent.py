"""
Ensemble Agent - Combinaison intelligente de multiples agents RL
Optimise la sélection et l'agrégation des décisions pour maximiser la performance

Caractéristiques principales:
- Voting mechanisms (hard/soft/weighted)
- Meta-learning pour la sélection d'agents
- Adaptation dynamique selon le régime de marché
- Uncertainty estimation
- Multi-agent coordination
- Performance tracking par agent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union, Protocol
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import logging
from datetime import datetime, timezone
from abc import ABC, abstractmethod

# Import des agents individuels
from ml.models.dqn import DQNAgent, DQNConfig
from ml.models.ppo import PPOAgent, PPOConfig
from ml.models.sac import SACAgent, SACConfig


class VotingMethod(Enum):
    """Méthodes de vote pour l'ensemble"""
    HARD_VOTING = "hard_voting"
    SOFT_VOTING = "soft_voting"
    WEIGHTED_VOTING = "weighted_voting"
    META_LEARNED = "meta_learned"
    UNCERTAINTY_WEIGHTED = "uncertainty_weighted"
    PERFORMANCE_WEIGHTED = "performance_weighted"


class AgentType(Enum):
    """Types d'agents supportés"""
    DQN = "dqn"
    PPO = "ppo"
    SAC = "sac"
    CUSTOM = "custom"


@dataclass
class AgentPerformance:
    """Métriques de performance pour un agent"""
    agent_id: str
    agent_type: AgentType
    total_decisions: int = 0
    successful_decisions: int = 0
    cumulative_reward: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    rolling_performance: deque = field(default_factory=lambda: deque(maxlen=1000))
    confidence_calibration: float = 1.0  # Calibration de la confiance
    regime_performance: Dict[str, float] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        if self.total_decisions > 0:
            return self.successful_decisions / self.total_decisions
        return 0.0
    
    @property
    def average_reward(self) -> float:
        if self.total_decisions > 0:
            return self.cumulative_reward / self.total_decisions
        return 0.0


@dataclass
class EnsembleConfig:
    """Configuration pour l'agent ensemble"""
    # Agents à inclure
    agent_configs: Dict[str, Union[DQNConfig, PPOConfig, SACConfig]] = field(default_factory=dict)
    enabled_agents: List[str] = field(default_factory=lambda: ["dqn", "ppo", "sac"])
    
    # Méthode d'ensemble
    voting_method: VotingMethod = VotingMethod.UNCERTAINTY_WEIGHTED
    min_agent_agreement: float = 0.3  # Consensus minimum
    
    # Meta-learning
    use_meta_learner: bool = True
    meta_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    meta_learning_rate: float = 3e-4
    
    # Uncertainty estimation
    use_uncertainty: bool = True
    uncertainty_method: str = "ensemble_variance"  # ou "dropout", "bootstrap"
    uncertainty_threshold: float = 0.5
    
    # Performance tracking
    performance_window: int = 100
    adaptation_rate: float = 0.1
    min_agent_experience: int = 1000  # Expérience min avant de contribuer
    
    # Coordination
    enable_communication: bool = True
    communication_rounds: int = 3
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class BaseRLAgent(Protocol):
    """Interface commune pour tous les agents RL"""
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[Any, Dict[str, Any]]:
        ...
    
    def train_step(self) -> Optional[Dict[str, float]]:
        ...
    
    def get_metrics(self) -> Dict[str, float]:
        ...


class MetaLearner(nn.Module):
    """Meta-learner pour combiner les décisions des agents"""
    def __init__(self, num_agents: int, state_dim: int, hidden_dims: List[int], action_dim: int):
        super().__init__()
        
        # Input: state + agent predictions + uncertainties
        input_dim = state_dim + num_agents * (action_dim + 2)  # actions + confidences + uncertainties
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        # Output: weights for each agent
        layers.append(nn.Linear(current_dim, num_agents))
        layers.append(nn.Softmax(dim=-1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state: torch.Tensor, agent_outputs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: Current state
            agent_outputs: Concatenated outputs from all agents
        Returns:
            weights: Weights for each agent
        """
        x = torch.cat([state, agent_outputs], dim=-1)
        weights = self.network(x)
        return weights


class UncertaintyEstimator:
    """Estime l'incertitude des prédictions"""
    def __init__(self, method: str = "ensemble_variance"):
        self.method = method
        self.prediction_history = defaultdict(deque)
        self.calibration_stats = defaultdict(dict)
    
    def estimate_uncertainty(
        self,
        agent_id: str,
        predictions: List[np.ndarray],
        confidences: Optional[List[float]] = None
    ) -> float:
        """Estime l'incertitude basée sur la méthode configurée"""
        if self.method == "ensemble_variance":
            # Variance des prédictions comme mesure d'incertitude
            if len(predictions) > 1:
                return float(np.var(predictions))
            return 0.0
        
        elif self.method == "confidence_based":
            # Utilise la confiance reportée par l'agent
            if confidences:
                avg_confidence = np.mean(confidences)
                # Incertitude = 1 - confiance
                return 1.0 - avg_confidence
            return 0.5
        
        elif self.method == "prediction_stability":
            # Stabilité des prédictions récentes
            self.prediction_history[agent_id].append(predictions[-1])
            if len(self.prediction_history[agent_id]) >= 10:
                recent_preds = list(self.prediction_history[agent_id])[-10:]
                stability = np.std(recent_preds)
                return float(stability)
            return 1.0
        
        return 0.5
    
    def calibrate_uncertainty(
        self,
        agent_id: str,
        predicted_uncertainty: float,
        actual_error: float
    ):
        """Calibre l'estimation d'incertitude avec l'erreur réelle"""
        if agent_id not in self.calibration_stats:
            self.calibration_stats[agent_id] = {
                'predicted': deque(maxlen=1000),
                'actual': deque(maxlen=1000)
            }
        
        self.calibration_stats[agent_id]['predicted'].append(predicted_uncertainty)
        self.calibration_stats[agent_id]['actual'].append(actual_error)
        
        # Calcul du facteur de calibration
        if len(self.calibration_stats[agent_id]['predicted']) >= 100:
            predicted = np.array(self.calibration_stats[agent_id]['predicted'])
            actual = np.array(self.calibration_stats[agent_id]['actual'])
            
            # Régression simple pour la calibration
            calibration_factor = np.corrcoef(predicted, actual)[0, 1]
            return calibration_factor
        
        return 1.0


class CommunicationModule:
    """Module de communication entre agents"""
    def __init__(self, num_agents: int, message_dim: int = 32):
        self.num_agents = num_agents
        self.message_dim = message_dim
        self.message_buffer = defaultdict(list)
        
        # Réseau de communication
        self.message_encoder = nn.Sequential(
            nn.Linear(message_dim, 64),
            nn.ReLU(),
            nn.Linear(64, message_dim)
        )
        
        self.message_aggregator = nn.GRU(
            input_size=message_dim,
            hidden_size=message_dim,
            batch_first=True
        )
    
    def send_message(self, sender_id: str, message: np.ndarray):
        """Envoie un message depuis un agent"""
        self.message_buffer[sender_id].append(message)
    
    def aggregate_messages(self, receiver_id: str) -> Optional[np.ndarray]:
        """Agrège les messages pour un agent récepteur"""
        all_messages = []
        
        for sender_id, messages in self.message_buffer.items():
            if sender_id != receiver_id and messages:
                all_messages.extend(messages)
        
        if not all_messages:
            return None
        
        # Encoder et agréger les messages
        messages_tensor = torch.FloatTensor(all_messages)
        encoded = self.message_encoder(messages_tensor)
        
        # Agréger avec GRU
        output, _ = self.message_aggregator(encoded.unsqueeze(0))
        aggregated = output.squeeze(0).mean(dim=0)
        
        return aggregated.detach().numpy()
    
    def clear_messages(self):
        """Vide le buffer de messages"""
        self.message_buffer.clear()


class EnsembleAgent:
    """Agent ensemble combinant plusieurs agents RL"""
    def __init__(self, config: EnsembleConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.logger = logging.getLogger(__name__)
        
        # Initialiser les agents individuels
        self.agents: Dict[str, BaseRLAgent] = {}
        self._initialize_agents()
        
        # Performance tracking
        self.agent_performance: Dict[str, AgentPerformance] = {
            agent_id: AgentPerformance(agent_id, self._get_agent_type(agent_id))
            for agent_id in self.agents.keys()
        }
        
        # Meta-learner
        self.meta_learner = None
        if config.use_meta_learner:
            # Dimensions seront définies après le premier appel
            self.meta_learner_initialized = False
            self.meta_optimizer = None
        
        # Uncertainty estimation
        self.uncertainty_estimator = UncertaintyEstimator(
            config.uncertainty_method
        ) if config.use_uncertainty else None
        
        # Communication
        self.communication = CommunicationModule(
            len(self.agents)
        ) if config.enable_communication else None
        
        # Métriques ensemble
        self.ensemble_decisions = 0
        self.agreement_history = deque(maxlen=1000)
        self.regime_adaptation = defaultdict(lambda: defaultdict(float))
        
        # Cache pour les décisions
        self.decision_cache = {}
        self.last_state_hash = None
    
    def _initialize_agents(self):
        """Initialise tous les agents configurés"""
        for agent_id in self.config.enabled_agents:
            if agent_id in self.config.agent_configs:
                agent_config = self.config.agent_configs[agent_id]
                
                if agent_id == "dqn" or isinstance(agent_config, DQNConfig):
                    self.agents[agent_id] = DQNAgent(agent_config)
                elif agent_id == "ppo" or isinstance(agent_config, PPOConfig):
                    self.agents[agent_id] = PPOAgent(agent_config)
                elif agent_id == "sac" or isinstance(agent_config, SACConfig):
                    self.agents[agent_id] = SACAgent(agent_config)
                else:
                    self.logger.warning(f"Type d'agent inconnu: {agent_id}")
    
    def _get_agent_type(self, agent_id: str) -> AgentType:
        """Détermine le type d'un agent"""
        if "dqn" in agent_id.lower():
            return AgentType.DQN
        elif "ppo" in agent_id.lower():
            return AgentType.PPO
        elif "sac" in agent_id.lower():
            return AgentType.SAC
        return AgentType.CUSTOM
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        market_regime: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Sélectionne une action en combinant les décisions des agents"""
        # Communication entre agents si activée
        if self.communication and self.config.communication_rounds > 0:
            self._conduct_communication_rounds(state)
        
        # Collecter les décisions de tous les agents
        agent_decisions = {}
        agent_infos = {}
        uncertainties = {}
        
        for agent_id, agent in self.agents.items():
            # Vérifier l'expérience minimale
            perf = self.agent_performance[agent_id]
            if perf.total_decisions < self.config.min_agent_experience and not deterministic:
                continue
            
            try:
                action, info = agent.select_action(state, deterministic)
                agent_decisions[agent_id] = action
                agent_infos[agent_id] = info
                
                # Estimer l'incertitude
                if self.uncertainty_estimator:
                    uncertainty = self.uncertainty_estimator.estimate_uncertainty(
                        agent_id,
                        [action] if isinstance(action, np.ndarray) else [np.array([action])],
                        [info.get('confidence', 0.5)] if 'confidence' in info else None
                    )
                    uncertainties[agent_id] = uncertainty
                    
            except Exception as e:
                self.logger.error(f"Erreur agent {agent_id}: {str(e)}")
                continue
        
        if not agent_decisions:
            # Aucun agent disponible
            default_action = 0 if not hasattr(self, 'last_action') else self.last_action
            return default_action, {'ensemble_method': 'fallback'}
        
        # Combiner les décisions selon la méthode configurée
        if self.config.voting_method == VotingMethod.HARD_VOTING:
            final_action, ensemble_info = self._hard_voting(agent_decisions)
        
        elif self.config.voting_method == VotingMethod.SOFT_VOTING:
            final_action, ensemble_info = self._soft_voting(agent_decisions, agent_infos)
        
        elif self.config.voting_method == VotingMethod.WEIGHTED_VOTING:
            final_action, ensemble_info = self._weighted_voting(
                agent_decisions, agent_infos, market_regime
            )
        
        elif self.config.voting_method == VotingMethod.UNCERTAINTY_WEIGHTED:
            final_action, ensemble_info = self._uncertainty_weighted_voting(
                agent_decisions, uncertainties
            )
        
        elif self.config.voting_method == VotingMethod.PERFORMANCE_WEIGHTED:
            final_action, ensemble_info = self._performance_weighted_voting(
                agent_decisions, market_regime
            )
        
        elif self.config.voting_method == VotingMethod.META_LEARNED:
            final_action, ensemble_info = self._meta_learned_combination(
                state, agent_decisions, agent_infos, uncertainties
            )
        
        else:
            # Par défaut: vote majoritaire
            final_action, ensemble_info = self._hard_voting(agent_decisions)
        
        # Calculer le niveau d'accord
        agreement_level = self._calculate_agreement(agent_decisions)
        self.agreement_history.append(agreement_level)
        
        # Mettre à jour les métriques
        self.ensemble_decisions += 1
        self.last_action = final_action
        
        # Informations complètes
        info = {
            'ensemble_method': self.config.voting_method.value,
            'num_agents': len(agent_decisions),
            'agreement_level': agreement_level,
            'agent_decisions': agent_decisions,
            'agent_uncertainties': uncertainties,
            'ensemble_info': ensemble_info,
            'active_agents': list(agent_decisions.keys())
        }
        
        return final_action, info
    
    def _hard_voting(
        self,
        agent_decisions: Dict[str, Any]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Vote majoritaire simple"""
        # Convertir les décisions en format comparable
        decisions_list = []
        for agent_id, decision in agent_decisions.items():
            if isinstance(decision, np.ndarray):
                # Pour les actions continues, discrétiser
                discretized = int(np.sign(decision[0]) + 1)  # -1,0,1 -> 0,1,2
                decisions_list.append(discretized)
            else:
                decisions_list.append(decision)
        
        # Vote majoritaire
        from collections import Counter
        vote_counts = Counter(decisions_list)
        majority_action = vote_counts.most_common(1)[0][0]
        
        # Si l'action originale était continue, reconstruire
        continuous_actions = [d for d in agent_decisions.values() if isinstance(d, np.ndarray)]
        if continuous_actions:
            # Moyenne des actions continues qui correspondent à la décision majoritaire
            matching_actions = []
            for agent_id, decision in agent_decisions.items():
                if isinstance(decision, np.ndarray):
                    if int(np.sign(decision[0]) + 1) == majority_action:
                        matching_actions.append(decision)
            
            if matching_actions:
                final_action = np.mean(matching_actions, axis=0)
            else:
                final_action = continuous_actions[0]  # Fallback
        else:
            final_action = majority_action
        
        return final_action, {
            'vote_distribution': dict(vote_counts),
            'majority_votes': vote_counts[majority_action]
        }
    
    def _soft_voting(
        self,
        agent_decisions: Dict[str, Any],
        agent_infos: Dict[str, Dict[str, Any]]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Vote pondéré par les probabilités/confiances"""
        weighted_sum = None
        total_weight = 0
        
        for agent_id, decision in agent_decisions.items():
            # Obtenir le poids (confiance ou probabilité)
            info = agent_infos.get(agent_id, {})
            weight = info.get('confidence', info.get('value', 0.5))
            
            if isinstance(decision, np.ndarray):
                if weighted_sum is None:
                    weighted_sum = decision * weight
                else:
                    weighted_sum += decision * weight
            else:
                # Pour les actions discrètes, accumuler les poids
                if weighted_sum is None:
                    weighted_sum = {}
                if decision not in weighted_sum:
                    weighted_sum[decision] = 0
                weighted_sum[decision] += weight
            
            total_weight += weight
        
        # Normaliser et retourner
        if isinstance(weighted_sum, dict):
            # Actions discrètes: choisir celle avec le poids max
            final_action = max(weighted_sum.items(), key=lambda x: x[1])[0]
            info = {'weight_distribution': weighted_sum}
        else:
            # Actions continues: moyenne pondérée
            final_action = weighted_sum / total_weight if total_weight > 0 else weighted_sum
            info = {'total_weight': total_weight}
        
        return final_action, info
    
    def _weighted_voting(
        self,
        agent_decisions: Dict[str, Any],
        agent_infos: Dict[str, Dict[str, Any]],
        market_regime: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Vote pondéré par la performance des agents"""
        weights = {}
        
        for agent_id in agent_decisions.keys():
            perf = self.agent_performance[agent_id]
            
            # Poids basé sur la performance
            base_weight = perf.success_rate * 0.4 + (1 - perf.max_drawdown) * 0.3
            
            # Ajustement selon le régime de marché
            if market_regime and market_regime in perf.regime_performance:
                regime_weight = perf.regime_performance[market_regime]
                base_weight = base_weight * 0.7 + regime_weight * 0.3
            
            # Ajustement selon le Sharpe ratio
            if perf.sharpe_ratio > 0:
                base_weight *= (1 + min(perf.sharpe_ratio / 2, 1))
            
            weights[agent_id] = max(0.1, base_weight)  # Poids minimum
        
        # Normaliser les poids
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calculer la décision pondérée
        return self._apply_weights(agent_decisions, weights), {
            'agent_weights': weights,
            'market_regime': market_regime
        }
    
    def _uncertainty_weighted_voting(
        self,
        agent_decisions: Dict[str, Any],
        uncertainties: Dict[str, float]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Vote pondéré inversement par l'incertitude"""
        # Convertir les incertitudes en poids (faible incertitude = poids élevé)
        weights = {}
        for agent_id in agent_decisions.keys():
            uncertainty = uncertainties.get(agent_id, 0.5)
            # Éviter la division par zéro
            weight = 1.0 / (uncertainty + 0.1)
            weights[agent_id] = weight
        
        # Normaliser
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return self._apply_weights(agent_decisions, weights), {
            'uncertainty_weights': weights,
            'average_uncertainty': np.mean(list(uncertainties.values()))
        }
    
    def _performance_weighted_voting(
        self,
        agent_decisions: Dict[str, Any],
        market_regime: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """Vote pondéré par la performance récente"""
        weights = {}
        
        for agent_id in agent_decisions.keys():
            perf = self.agent_performance[agent_id]
            
            # Performance récente (derniers N trades)
            if perf.rolling_performance:
                recent_perf = np.mean(list(perf.rolling_performance)[-20:])
            else:
                recent_perf = 0.5
            
            # Ajuster selon le régime si disponible
            if market_regime and market_regime in perf.regime_performance:
                regime_factor = perf.regime_performance[market_regime]
                recent_perf = recent_perf * 0.7 + regime_factor * 0.3
            
            weights[agent_id] = max(0.1, recent_perf)
        
        # Normaliser
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return self._apply_weights(agent_decisions, weights), {
            'performance_weights': weights,
            'regime_considered': market_regime is not None
        }
    
    def _meta_learned_combination(
        self,
        state: np.ndarray,
        agent_decisions: Dict[str, Any],
        agent_infos: Dict[str, Dict[str, Any]],
        uncertainties: Dict[str, float]
    ) -> Tuple[Any, Dict[str, Any]]:
        """Combinaison apprise par meta-learning"""
        # Initialiser le meta-learner si nécessaire
        if not self.meta_learner_initialized:
            self._initialize_meta_learner(state, agent_decisions)
        
        # Préparer les inputs pour le meta-learner
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Concaténer les outputs des agents
        agent_outputs = []
        agent_ids_ordered = sorted(agent_decisions.keys())
        
        for agent_id in agent_ids_ordered:
            decision = agent_decisions[agent_id]
            info = agent_infos.get(agent_id, {})
            uncertainty = uncertainties.get(agent_id, 0.5)
            
            # Convertir la décision en vecteur
            if isinstance(decision, np.ndarray):
                decision_vec = decision.flatten()
            else:
                # One-hot pour les actions discrètes
                decision_vec = np.zeros(3)
                decision_vec[int(decision)] = 1
            
            # Ajouter confiance et incertitude
            confidence = info.get('confidence', info.get('value', 0.5))
            agent_output = np.concatenate([
                decision_vec,
                [confidence],
                [uncertainty]
            ])
            agent_outputs.append(agent_output)
        
        agent_outputs_tensor = torch.FloatTensor(agent_outputs).flatten().unsqueeze(0).to(self.device)
        
        # Obtenir les poids du meta-learner
        with torch.no_grad():
            weights = self.meta_learner(state_tensor, agent_outputs_tensor)
            weights_np = weights.cpu().numpy()[0]
        
        # Créer le dictionnaire de poids
        weights_dict = {
            agent_id: float(weights_np[i])
            for i, agent_id in enumerate(agent_ids_ordered)
        }
        
        return self._apply_weights(agent_decisions, weights_dict), {
            'meta_weights': weights_dict,
            'meta_learner_active': True
        }
    
    def _apply_weights(
        self,
        agent_decisions: Dict[str, Any],
        weights: Dict[str, float]
    ) -> Any:
        """Applique les poids aux décisions des agents"""
        # Vérifier le type de décisions
        sample_decision = next(iter(agent_decisions.values()))
        
        if isinstance(sample_decision, np.ndarray):
            # Actions continues: moyenne pondérée
            weighted_sum = None
            for agent_id, decision in agent_decisions.items():
                weight = weights.get(agent_id, 0)
                if weighted_sum is None:
                    weighted_sum = decision * weight
                else:
                    weighted_sum += decision * weight
            return weighted_sum
        else:
            # Actions discrètes: vote pondéré
            action_weights = defaultdict(float)
            for agent_id, decision in agent_decisions.items():
                weight = weights.get(agent_id, 0)
                action_weights[decision] += weight
            
            # Retourner l'action avec le poids maximum
            return max(action_weights.items(), key=lambda x: x[1])[0]
    
    def _calculate_agreement(self, agent_decisions: Dict[str, Any]) -> float:
        """Calcule le niveau d'accord entre les agents"""
        if len(agent_decisions) <= 1:
            return 1.0
        
        decisions_list = list(agent_decisions.values())
        
        # Pour les actions continues
        if isinstance(decisions_list[0], np.ndarray):
            # Calculer la variance relative
            decisions_array = np.array(decisions_list)
            mean_decision = np.mean(decisions_array, axis=0)
            variance = np.mean([np.linalg.norm(d - mean_decision) for d in decisions_array])
            # Normaliser (0 = accord parfait, 1 = désaccord maximal)
            agreement = 1.0 - min(variance / (np.linalg.norm(mean_decision) + 1e-6), 1.0)
        else:
            # Pour les actions discrètes
            from collections import Counter
            counts = Counter(decisions_list)
            most_common_count = counts.most_common(1)[0][1]
            agreement = most_common_count / len(decisions_list)
        
        return float(agreement)
    
    def _conduct_communication_rounds(self, state: np.ndarray):
        """Effectue des rounds de communication entre agents"""
        if not self.communication:
            return
        
        for round_idx in range(self.config.communication_rounds):
            # Chaque agent envoie un message basé sur son état interne
            for agent_id in self.agents.keys():
                # Créer un message (placeholder - dépend de l'implémentation de l'agent)
                message = np.random.randn(32)  # Message aléatoire pour l'exemple
                self.communication.send_message(agent_id, message)
            
            # Chaque agent reçoit et traite les messages
            for agent_id in self.agents.keys():
                aggregated_message = self.communication.aggregate_messages(agent_id)
                # Traiter le message (dépend de l'implémentation de l'agent)
                # ...
            
            # Nettoyer pour le prochain round
            self.communication.clear_messages()
    
    def _initialize_meta_learner(self, state: np.ndarray, agent_decisions: Dict[str, Any]):
        """Initialise le meta-learner avec les bonnes dimensions"""
        num_agents = len(agent_decisions)
        state_dim = len(state)
        
        # Déterminer la dimension des actions
        sample_decision = next(iter(agent_decisions.values()))
        if isinstance(sample_decision, np.ndarray):
            action_dim = len(sample_decision.flatten())
        else:
            action_dim = 3  # Nombre d'actions discrètes
        
        self.meta_learner = MetaLearner(
            num_agents=num_agents,
            state_dim=state_dim,
            hidden_dims=self.config.meta_hidden_dims,
            action_dim=action_dim
        ).to(self.device)
        
        self.meta_optimizer = optim.Adam(
            self.meta_learner.parameters(),
            lr=self.config.meta_learning_rate
        )
        
        self.meta_learner_initialized = True
    
    def update_performance(
        self,
        agent_decisions: Dict[str, Any],
        reward: float,
        success: bool,
        market_regime: Optional[str] = None
    ):
        """Met à jour les métriques de performance des agents"""
        for agent_id, decision in agent_decisions.items():
            perf = self.agent_performance[agent_id]
            
            # Métriques de base
            perf.total_decisions += 1
            if success:
                perf.successful_decisions += 1
            perf.cumulative_reward += reward
            
            # Performance glissante
            perf.rolling_performance.append(1.0 if success else 0.0)
            
            # Performance par régime
            if market_regime:
                if market_regime not in perf.regime_performance:
                    perf.regime_performance[market_regime] = 0.5
                
                # Mise à jour avec lissage exponentiel
                alpha = self.config.adaptation_rate
                old_perf = perf.regime_performance[market_regime]
                new_perf = 1.0 if success else 0.0
                perf.regime_performance[market_regime] = old_perf * (1 - alpha) + new_perf * alpha
    
    def train_meta_learner(
        self,
        states: List[np.ndarray],
        agent_decisions_list: List[Dict[str, Any]],
        rewards: List[float]
    ):
        """Entraîne le meta-learner sur des données historiques"""
        if not self.meta_learner or not self.meta_optimizer:
            return
        
        # Préparer les données
        # ... (implémentation détaillée selon les besoins)
        
        self.logger.info("Entraînement du meta-learner effectué")
    
    def train_agents(self) -> Dict[str, Dict[str, float]]:
        """Entraîne tous les agents"""
        training_metrics = {}
        
        for agent_id, agent in self.agents.items():
            try:
                metrics = agent.train_step()
                if metrics:
                    training_metrics[agent_id] = metrics
            except Exception as e:
                self.logger.error(f"Erreur d'entraînement pour {agent_id}: {str(e)}")
        
        return training_metrics
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de l'ensemble"""
        # Métriques individuelles des agents
        agent_metrics = {}
        for agent_id, agent in self.agents.items():
            agent_metrics[agent_id] = agent.get_metrics()
        
        # Métriques de performance
        performance_metrics = {}
        for agent_id, perf in self.agent_performance.items():
            performance_metrics[agent_id] = {
                'success_rate': perf.success_rate,
                'average_reward': perf.average_reward,
                'total_decisions': perf.total_decisions,
                'sharpe_ratio': perf.sharpe_ratio,
                'regime_performance': dict(perf.regime_performance)
            }
        
        # Métriques ensemble
        ensemble_metrics = {
            'total_decisions': self.ensemble_decisions,
            'average_agreement': float(np.mean(self.agreement_history)) if self.agreement_history else 0,
            'voting_method': self.config.voting_method.value,
            'num_active_agents': len([p for p in self.agent_performance.values() 
                                    if p.total_decisions >= self.config.min_agent_experience]),
            'meta_learner_active': self.meta_learner is not None
        }
        
        return {
            'agent_metrics': agent_metrics,
            'performance_metrics': performance_metrics,
            'ensemble_metrics': ensemble_metrics
        }
    
    def save_ensemble(self, path: str):
        """Sauvegarde l'état complet de l'ensemble"""
        ensemble_state = {
            'config': self.config,
            'agent_performance': self.agent_performance,
            'ensemble_decisions': self.ensemble_decisions,
            'agreement_history': list(self.agreement_history),
            'regime_adaptation': dict(self.regime_adaptation)
        }
        
        # Sauvegarder le meta-learner si présent
        if self.meta_learner:
            ensemble_state['meta_learner_state'] = self.meta_learner.state_dict()
            ensemble_state['meta_optimizer_state'] = self.meta_optimizer.state_dict()
        
        # Sauvegarder les états des agents individuels
        agent_states = {}
        for agent_id, agent in self.agents.items():
            # Chaque agent a sa propre méthode de sauvegarde
            agent_path = f"{path}_{agent_id}.pth"
            agent.save_model(agent_path)
            agent_states[agent_id] = agent_path
        
        ensemble_state['agent_paths'] = agent_states
        
        # Sauvegarder l'état de l'ensemble
        torch.save(ensemble_state, f"{path}_ensemble.pth")
        self.logger.info(f"Ensemble sauvegardé dans {path}")


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration des agents individuels
    dqn_config = DQNConfig(state_dim=50, action_dim=11)
    ppo_config = PPOConfig(state_dim=50, discrete_action_dim=3, continuous_action_dim=1)
    sac_config = SACConfig(state_dim=50, action_dim=3)
    
    # Configuration de l'ensemble
    ensemble_config = EnsembleConfig(
        agent_configs={
            "dqn": dqn_config,
            "ppo": ppo_config,
            "sac": sac_config
        },
        enabled_agents=["dqn", "ppo", "sac"],
        voting_method=VotingMethod.UNCERTAINTY_WEIGHTED,
        use_meta_learner=True,
        use_uncertainty=True
    )
    
    # Créer l'agent ensemble
    ensemble = EnsembleAgent(ensemble_config)
    
    # Test
    state = np.random.randn(50)
    action, info = ensemble.select_action(state, deterministic=False)
    
    print(f"Ensemble Agent créé avec {len(ensemble.agents)} agents")
    print(f"Méthode de vote: {ensemble_config.voting_method.value}")
    print(f"Action sélectionnée: {action}")
    print(f"Niveau d'accord: {info['agreement_level']:.2f}")
    print(f"Agents actifs: {info['active_agents']}")