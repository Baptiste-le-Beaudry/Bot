"""
Deep Q-Network (DQN) - Modèle de Deep Reinforcement Learning pour le Trading
Implémentation optimisée pour la gestion de portefeuille et l'exécution d'ordres

Caractéristiques principales:
- Architecture neuronale profonde avec mécanismes d'attention
- Double DQN pour stabilité
- Experience replay prioritisé
- Dueling network architecture
- Support multi-assets et actions continues
- Intégration native avec l'environnement de trading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import deque, namedtuple
import random
from datetime import datetime, timezone
import logging
from enum import Enum

# Types pour le trading
State = np.ndarray
Action = Union[int, np.ndarray]
Reward = float
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done', 'info'])


class ActionSpace(Enum):
    """Espace d'actions pour le trading"""
    HOLD = 0
    BUY = 1
    SELL = 2
    # Actions étendues pour position sizing
    BUY_25 = 3   # Acheter 25% de la position max
    BUY_50 = 4   # Acheter 50%
    BUY_75 = 5   # Acheter 75%
    BUY_100 = 6  # Acheter 100%
    SELL_25 = 7  # Vendre 25%
    SELL_50 = 8  # Vendre 50%
    SELL_75 = 9  # Vendre 75%
    SELL_100 = 10 # Vendre 100%


@dataclass
class DQNConfig:
    """Configuration pour le DQN"""
    # Architecture du réseau
    state_dim: int = 100  # Dimension de l'état (features)
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    action_dim: int = 11  # Nombre d'actions possibles
    
    # Hyperparamètres d'apprentissage
    learning_rate: float = 3e-4
    batch_size: int = 64
    gamma: float = 0.99  # Facteur de discount
    tau: float = 0.001  # Soft update pour target network
    
    # Experience replay
    buffer_size: int = 100000
    prioritized_replay: bool = True
    alpha: float = 0.6  # Prioritization exponent
    beta: float = 0.4   # Importance sampling exponent
    beta_increment: float = 0.001
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # Architecture avancée
    use_dueling: bool = True
    use_double_dqn: bool = True
    use_noisy_net: bool = True
    use_attention: bool = True
    
    # Trading spécifique
    reward_scaling: float = 1.0
    use_lstm: bool = True  # Pour capturer les dépendances temporelles
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    
    # Entraînement
    update_frequency: int = 4
    target_update_frequency: int = 1000
    gradient_clip: float = 10.0
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class NoisyLinear(nn.Module):
    """Couche linéaire bruitée pour exploration"""
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Paramètres
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class AttentionLayer(nn.Module):
    """Mécanisme d'attention pour les features importantes"""
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        self.W_q = nn.Linear(feature_dim, hidden_dim)
        self.W_k = nn.Linear(feature_dim, hidden_dim)
        self.W_v = nn.Linear(feature_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, feature_dim)
        
        self.scale = np.sqrt(hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, feature_dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Self-attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        context = torch.matmul(attention_weights, V)
        output = self.W_o(context)
        
        # Residual connection
        return x + output


class DuelingDQN(nn.Module):
    """Architecture Dueling DQN avec support optionnel pour LSTM et Attention"""
    def __init__(self, config: DQNConfig):
        super().__init__()
        self.config = config
        
        # LSTM pour capturer les dépendances temporelles
        if config.use_lstm:
            self.lstm = nn.LSTM(
                input_size=config.state_dim,
                hidden_size=config.lstm_hidden_size,
                num_layers=config.lstm_num_layers,
                batch_first=True,
                dropout=0.2 if config.lstm_num_layers > 1 else 0
            )
            feature_dim = config.lstm_hidden_size
        else:
            feature_dim = config.state_dim
        
        # Attention layer
        if config.use_attention:
            self.attention = AttentionLayer(feature_dim)
        
        # Shared layers
        self.shared_layers = nn.ModuleList()
        input_dim = feature_dim
        
        for hidden_dim in config.hidden_dims[:-1]:
            if config.use_noisy_net:
                self.shared_layers.append(NoisyLinear(input_dim, hidden_dim))
            else:
                self.shared_layers.append(nn.Linear(input_dim, hidden_dim))
            self.shared_layers.append(nn.ReLU())
            self.shared_layers.append(nn.Dropout(0.2))
            input_dim = hidden_dim
        
        final_hidden_dim = config.hidden_dims[-1]
        
        if config.use_dueling:
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(input_dim, final_hidden_dim),
                nn.ReLU(),
                nn.Linear(final_hidden_dim, 1)
            )
            
            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(input_dim, final_hidden_dim),
                nn.ReLU(),
                nn.Linear(final_hidden_dim, config.action_dim)
            )
        else:
            # Standard Q-network
            self.q_network = nn.Sequential(
                nn.Linear(input_dim, final_hidden_dim),
                nn.ReLU(),
                nn.Linear(final_hidden_dim, config.action_dim)
            )
    
    def forward(
        self, 
        state: torch.Tensor, 
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass avec support optionnel pour hidden state LSTM"""
        x = state
        
        # LSTM processing
        if self.config.use_lstm:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # Add sequence dimension
            
            if hidden is None:
                x, hidden = self.lstm(x)
            else:
                x, hidden = self.lstm(x, hidden)
            
            # Take last output
            if x.shape[1] > 1:
                x = x[:, -1, :]
            else:
                x = x.squeeze(1)
        
        # Attention
        if self.config.use_attention:
            x = self.attention(x)
        
        # Shared layers
        for layer in self.shared_layers:
            x = layer(x)
        
        # Dueling architecture
        if self.config.use_dueling:
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            # Combine value and advantage
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q_values = self.q_network(x)
        
        return q_values, hidden
    
    def reset_noise(self):
        """Reset noise for NoisyNet layers"""
        for layer in self.shared_layers:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()


class PrioritizedReplayBuffer:
    """Buffer de replay avec prioritisation pour DQN"""
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0
    
    def push(self, experience: Experience):
        """Ajoute une expérience avec priorité maximale"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
        
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """Échantillonne un batch avec importance sampling"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
        
        # Calcul des probabilités
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Échantillonnage
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Met à jour les priorités basées sur les TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Agent DQN complet pour le trading algorithmique"""
    def __init__(self, config: DQNConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks
        self.q_network = DuelingDQN(config).to(self.device)
        self.target_network = DuelingDQN(config).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=10000, eta_min=1e-5
        )
        
        # Replay buffer
        if config.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(config.buffer_size, config.alpha)
        else:
            self.replay_buffer = deque(maxlen=config.buffer_size)
        
        # Training state
        self.epsilon = config.epsilon_start
        self.beta = config.beta
        self.steps = 0
        self.episodes = 0
        self.hidden_state = None
        
        # Metrics
        self.losses = deque(maxlen=1000)
        self.rewards = deque(maxlen=1000)
        self.q_values = deque(maxlen=1000)
        
        self.logger = logging.getLogger(__name__)
    
    def select_action(
        self, 
        state: np.ndarray, 
        training: bool = True
    ) -> Tuple[int, Optional[Dict[str, Any]]]:
        """Sélectionne une action selon la politique epsilon-greedy ou noisy net"""
        # Epsilon-greedy exploration (si pas de noisy net)
        if training and not self.config.use_noisy_net:
            if random.random() < self.epsilon:
                action = random.randint(0, self.config.action_dim - 1)
                return action, {'exploration': True, 'epsilon': self.epsilon}
        
        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values, self.hidden_state = self.q_network(state_tensor, self.hidden_state)
            action = q_values.argmax(dim=1).item()
            
            # Enregistrer les Q-values pour monitoring
            self.q_values.append(q_values.cpu().numpy())
            
            return action, {
                'exploration': False,
                'q_values': q_values.cpu().numpy(),
                'epsilon': self.epsilon
            }
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: Optional[Dict[str, Any]] = None
    ):
        """Stocke une expérience dans le replay buffer"""
        experience = Experience(state, action, reward, next_state, done, info or {})
        
        if self.config.prioritized_replay:
            self.replay_buffer.push(experience)
        else:
            self.replay_buffer.append(experience)
        
        self.rewards.append(reward)
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """Effectue une étape d'entraînement"""
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        # Échantillonnage
        if self.config.prioritized_replay:
            experiences, indices, weights = self.replay_buffer.sample(
                self.config.batch_size, self.beta
            )
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            experiences = random.sample(self.replay_buffer, self.config.batch_size)
            weights = torch.ones(self.config.batch_size).to(self.device)
            indices = None
        
        # Préparation des batches
        batch = Experience(*zip(*experiences))
        
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Current Q values
        current_q_values, _ = self.q_network(state_batch)
        current_q_values = current_q_values.gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        # Next Q values
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: use online network to select action, target network to evaluate
                next_actions, _ = self.q_network(next_state_batch)
                next_actions = next_actions.argmax(dim=1)
                next_q_values_target, _ = self.target_network(next_state_batch)
                next_q_values = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                next_q_values, _ = self.target_network(next_state_batch)
                next_q_values = next_q_values.max(dim=1)[0]
            
            # Compute targets
            targets = reward_batch + self.config.gamma * next_q_values * (1 - done_batch)
        
        # Compute loss
        td_errors = targets - current_q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        # Update priorities if using prioritized replay
        if self.config.prioritized_replay and indices is not None:
            self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config.gradient_clip)
        
        self.optimizer.step()
        self.scheduler.step()
        
        # Update target network
        if self.steps % self.config.target_update_frequency == 0:
            self.soft_update_target_network()
        
        # Update exploration parameters
        self.update_exploration_params()
        
        # Reset noise if using noisy net
        if self.config.use_noisy_net:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        self.steps += 1
        self.losses.append(loss.item())
        
        return {
            'loss': loss.item(),
            'td_error_mean': td_errors.mean().item(),
            'q_value_mean': current_q_values.mean().item(),
            'epsilon': self.epsilon,
            'beta': self.beta
        }
    
    def soft_update_target_network(self):
        """Soft update du target network"""
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    def update_exploration_params(self):
        """Met à jour les paramètres d'exploration"""
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        
        # Increase beta for importance sampling
        if self.config.prioritized_replay:
            self.beta = min(1.0, self.beta + self.config.beta_increment)
    
    def reset_episode(self):
        """Reset pour un nouvel épisode"""
        self.hidden_state = None
        self.episodes += 1
    
    def save_model(self, path: str):
        """Sauvegarde le modèle"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'beta': self.beta,
            'config': self.config
        }, path)
        
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Charge le modèle"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.epsilon = checkpoint['epsilon']
        self.beta = checkpoint.get('beta', self.config.beta)
        
        self.logger.info(f"Model loaded from {path}")
    
    def get_metrics(self) -> Dict[str, float]:
        """Retourne les métriques d'entraînement"""
        return {
            'avg_loss': np.mean(self.losses) if self.losses else 0,
            'avg_reward': np.mean(self.rewards) if self.rewards else 0,
            'avg_q_value': np.mean([q.mean() for q in self.q_values]) if self.q_values else 0,
            'epsilon': self.epsilon,
            'beta': self.beta,
            'buffer_size': len(self.replay_buffer),
            'steps': self.steps,
            'episodes': self.episodes
        }


# Fonctions utilitaires pour le preprocessing des états
def preprocess_state(
    market_data: Dict[str, Any],
    technical_indicators: Dict[str, float],
    portfolio_state: Dict[str, float],
    config: DQNConfig
) -> np.ndarray:
    """Préprocesse l'état pour le DQN"""
    features = []
    
    # Prix et volumes
    features.extend([
        market_data['price'],
        market_data['volume'],
        market_data['bid'],
        market_data['ask'],
        market_data['spread']
    ])
    
    # Indicateurs techniques
    features.extend([
        technical_indicators.get('sma_20', 0),
        technical_indicators.get('sma_50', 0),
        technical_indicators.get('rsi', 50),
        technical_indicators.get('macd', 0),
        technical_indicators.get('bb_upper', 0),
        technical_indicators.get('bb_lower', 0),
        technical_indicators.get('atr', 0)
    ])
    
    # État du portefeuille
    features.extend([
        portfolio_state['position'],
        portfolio_state['unrealized_pnl'],
        portfolio_state['cash_balance'],
        portfolio_state['total_value']
    ])
    
    # Padding si nécessaire
    if len(features) < config.state_dim:
        features.extend([0] * (config.state_dim - len(features)))
    
    return np.array(features[:config.state_dim], dtype=np.float32)


def calculate_reward(
    pnl: float,
    sharpe_ratio: float,
    max_drawdown: float,
    position_held: int,
    config: DQNConfig
) -> float:
    """Calcule la récompense pour le DQN avec pénalités de risque"""
    # Récompense de base (PnL)
    reward = pnl * config.reward_scaling
    
    # Bonus pour Sharpe ratio élevé
    if sharpe_ratio > 1.5:
        reward += 0.1 * sharpe_ratio
    
    # Pénalité pour drawdown
    if max_drawdown > 0.1:  # 10%
        reward -= 0.5 * max_drawdown
    
    # Pénalité pour tenir des positions trop longtemps
    if abs(position_held) > 100:
        reward -= 0.01 * abs(position_held)
    
    return reward


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    config = DQNConfig(
        state_dim=50,
        hidden_dims=[256, 128, 64],
        action_dim=11,
        use_dueling=True,
        use_double_dqn=True,
        use_lstm=True,
        prioritized_replay=True
    )
    
    # Créer l'agent
    agent = DQNAgent(config)
    
    # Exemple d'entraînement
    print(f"DQN Agent créé avec {sum(p.numel() for p in agent.q_network.parameters())} paramètres")
    print(f"Architecture: Dueling={config.use_dueling}, Double={config.use_double_dqn}, LSTM={config.use_lstm}")
    print(f"Device: {config.device}")