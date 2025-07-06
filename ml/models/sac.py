"""
Soft Actor-Critic (SAC) - Algorithme de Deep RL pour Trading avec Actions Continues
Implémentation optimisée pour la gestion de portefeuille avec entropy regularization

Caractéristiques principales:
- Maximum entropy reinforcement learning
- Double Q-networks pour stabilité
- Actions continues pour position sizing précis
- Temperature auto-ajustable
- Support multi-assets
- Reparameterization trick pour backprop
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import deque
import random
import logging
from datetime import datetime, timezone
import math


@dataclass
class SACConfig:
    """Configuration pour SAC"""
    # Architecture
    state_dim: int = 100
    action_dim: int = 3  # [position_size, stop_loss_distance, take_profit_distance]
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 256])
    
    # Hyperparamètres SAC
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 3e-4
    learning_rate_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    
    # Entropy
    init_temperature: float = 0.2
    target_entropy: Optional[float] = None  # Si None, calculé automatiquement
    learnable_temperature: bool = True
    
    # Architecture avancée
    use_layer_norm: bool = True
    use_spectral_norm: bool = False
    activation: str = "relu"  # relu, tanh, swish
    dropout_rate: float = 0.1
    
    # Replay buffer
    buffer_size: int = 1000000
    batch_size: int = 256
    
    # Trading spécifique
    action_scale: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.1, 0.1]))
    action_bias: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.05, 0.05]))
    reward_scale: float = 5.0
    
    # Entraînement
    gradient_clip: float = 1.0
    update_frequency: int = 1
    actor_update_frequency: int = 2
    target_update_frequency: int = 2
    warmup_steps: int = 10000
    
    # Régularisation
    weight_decay: float = 1e-4
    gradient_penalty: float = 0.0
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class ReplayBuffer:
    """Replay buffer optimisé pour SAC"""
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.position = 0
        self.size = 0
        
        # Pré-allocation des arrays pour efficacité
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        # Métadonnées optionnelles
        self.metadata = [None] * capacity
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Ajoute une transition"""
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.metadata[self.position] = metadata
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Échantillonne un batch"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_states': torch.FloatTensor(self.next_states[indices]),
            'dones': torch.FloatTensor(self.dones[indices])
        }
    
    def __len__(self):
        return self.size


def create_mlp(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    activation: str = "relu",
    use_layer_norm: bool = True,
    use_spectral_norm: bool = False,
    dropout_rate: float = 0.1,
    output_activation: Optional[str] = None
) -> nn.Module:
    """Crée un MLP avec les options spécifiées"""
    layers = []
    dims = [input_dim] + hidden_dims
    
    # Activation functions
    activations = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "swish": nn.SiLU,
        "gelu": nn.GELU
    }
    
    activation_fn = activations.get(activation, nn.ReLU)
    
    # Hidden layers
    for i in range(len(dims) - 1):
        layer = nn.Linear(dims[i], dims[i + 1])
        
        # Spectral normalization
        if use_spectral_norm:
            layer = nn.utils.spectral_norm(layer)
        
        layers.append(layer)
        
        # Layer normalization
        if use_layer_norm:
            layers.append(nn.LayerNorm(dims[i + 1]))
        
        layers.append(activation_fn())
        
        # Dropout
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
    
    # Output layer
    output_layer = nn.Linear(dims[-1], output_dim)
    if use_spectral_norm:
        output_layer = nn.utils.spectral_norm(output_layer)
    layers.append(output_layer)
    
    # Output activation
    if output_activation == "tanh":
        layers.append(nn.Tanh())
    elif output_activation == "sigmoid":
        layers.append(nn.Sigmoid())
    
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """Actor network avec politique gaussienne pour actions continues"""
    def __init__(self, config: SACConfig):
        super().__init__()
        self.config = config
        
        # Feature extractor
        self.feature_net = create_mlp(
            config.state_dim,
            config.hidden_dims[:-1],
            config.hidden_dims[-1],
            activation=config.activation,
            use_layer_norm=config.use_layer_norm,
            use_spectral_norm=config.use_spectral_norm,
            dropout_rate=config.dropout_rate
        )
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(config.hidden_dims[-1], config.action_dim)
        self.log_std_head = nn.Linear(config.hidden_dims[-1], config.action_dim)
        
        # Action rescaling
        self.register_buffer('action_scale', torch.FloatTensor(config.action_scale))
        self.register_buffer('action_bias', torch.FloatTensor(config.action_bias))
        
        # Limits for log_std
        self.log_std_min = -20
        self.log_std_max = 2
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        # Small initialization for output layers
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.constant_(self.mean_head.bias, 0)
        nn.init.constant_(self.log_std_head.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        Returns: mean, log_std
        """
        features = self.feature_net(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        Returns: action, log_prob, mean
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if deterministic:
            # Use mean for deterministic policy
            action = mean
            log_prob = None
        else:
            # Sample from Gaussian
            normal = Normal(mean, std)
            x_t = normal.rsample()  # Reparameterization trick
            action = torch.tanh(x_t)
            
            # Compute log probability with correction for tanh squashing
            log_prob = normal.log_prob(x_t)
            # Enforcing action bounds
            log_prob -= torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            
        # Scale actions
        action = action * self.action_scale + self.action_bias
        
        return action, log_prob, torch.tanh(mean) * self.action_scale + self.action_bias


class QNetwork(nn.Module):
    """Critic network pour estimer Q(s,a)"""
    def __init__(self, config: SACConfig):
        super().__init__()
        self.config = config
        
        self.q_net = create_mlp(
            config.state_dim + config.action_dim,
            config.hidden_dims,
            1,
            activation=config.activation,
            use_layer_norm=config.use_layer_norm,
            use_spectral_norm=config.use_spectral_norm,
            dropout_rate=config.dropout_rate
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = torch.cat([state, action], dim=1)
        q_value = self.q_net(x)
        return q_value


class SACAgent:
    """Agent SAC complet pour le trading"""
    def __init__(self, config: SACConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks
        self.actor = GaussianPolicy(config).to(self.device)
        self.q1 = QNetwork(config).to(self.device)
        self.q2 = QNetwork(config).to(self.device)
        self.q1_target = QNetwork(config).to(self.device)
        self.q2_target = QNetwork(config).to(self.device)
        
        # Copy parameters to targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.learning_rate_actor,
            weight_decay=config.weight_decay
        )
        self.q1_optimizer = optim.Adam(
            self.q1.parameters(),
            lr=config.learning_rate_critic,
            weight_decay=config.weight_decay
        )
        self.q2_optimizer = optim.Adam(
            self.q2.parameters(),
            lr=config.learning_rate_critic,
            weight_decay=config.weight_decay
        )
        
        # Entropy temperature
        if config.target_entropy is None:
            # Heuristic: -dim(A)
            self.target_entropy = -config.action_dim
        else:
            self.target_entropy = config.target_entropy
        
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.learning_rate_alpha)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            config.buffer_size,
            config.state_dim,
            config.action_dim
        )
        
        # Training state
        self.training_step = 0
        self.episodes = 0
        
        # Metrics
        self.metrics = {
            'actor_loss': deque(maxlen=1000),
            'q1_loss': deque(maxlen=1000),
            'q2_loss': deque(maxlen=1000),
            'alpha_loss': deque(maxlen=1000),
            'alpha': deque(maxlen=1000),
            'rewards': deque(maxlen=1000)
        }
        
        self.logger = logging.getLogger(__name__)
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Sélectionne une action"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, mean = self.actor.sample(state_tensor, deterministic)
            
            action_np = action.cpu().numpy()[0]
            
            info = {
                'log_prob': log_prob.cpu().item() if log_prob is not None else 0,
                'mean_action': mean.cpu().numpy()[0],
                'alpha': self.alpha
            }
            
            return action_np, info
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Stocke une transition"""
        # Scale reward
        scaled_reward = reward * self.config.reward_scale
        
        self.replay_buffer.push(state, action, scaled_reward, next_state, done, metadata)
        self.metrics['rewards'].append(reward)
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """Effectue une étape d'entraînement"""
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.config.batch_size)
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Update critics
        q1_loss, q2_loss = self._update_critics(states, actions, rewards, next_states, dones)
        
        # Update actor
        actor_loss = None
        if self.training_step % self.config.actor_update_frequency == 0:
            actor_loss = self._update_actor(states)
        
        # Update temperature
        alpha_loss = None
        if self.config.learnable_temperature:
            alpha_loss = self._update_temperature(states)
        
        # Update target networks
        if self.training_step % self.config.target_update_frequency == 0:
            self._soft_update_targets()
        
        self.training_step += 1
        
        # Metrics
        metrics = {
            'q1_loss': q1_loss,
            'q2_loss': q2_loss,
            'alpha': self.alpha
        }
        
        if actor_loss is not None:
            metrics['actor_loss'] = actor_loss
        
        if alpha_loss is not None:
            metrics['alpha_loss'] = alpha_loss
        
        return metrics
    
    def _update_critics(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ) -> Tuple[float, float]:
        """Update Q-networks"""
        with torch.no_grad():
            # Sample next actions
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            
            # Compute target Q-values
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next)
            
            # Add entropy term
            q_next = q_next - self.alpha * next_log_probs
            
            # Compute targets
            q_targets = rewards + self.config.gamma * (1 - dones) * q_next
        
        # Current Q-values
        q1_values = self.q1(states, actions)
        q2_values = self.q2(states, actions)
        
        # Compute losses
        q1_loss = F.mse_loss(q1_values, q_targets)
        q2_loss = F.mse_loss(q2_values, q_targets)
        
        # Optimize Q1
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self.config.gradient_clip)
        self.q1_optimizer.step()
        
        # Optimize Q2
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self.config.gradient_clip)
        self.q2_optimizer.step()
        
        # Store metrics
        self.metrics['q1_loss'].append(q1_loss.item())
        self.metrics['q2_loss'].append(q2_loss.item())
        
        return q1_loss.item(), q2_loss.item()
    
    def _update_actor(self, states: torch.Tensor) -> float:
        """Update policy network"""
        # Sample actions
        actions, log_probs, _ = self.actor.sample(states)
        
        # Compute Q-values
        q1_values = self.q1(states, actions)
        q2_values = self.q2(states, actions)
        q_values = torch.min(q1_values, q2_values)
        
        # Actor loss (maximize Q - α * log_prob)
        actor_loss = -(q_values - self.alpha * log_probs).mean()
        
        # Add gradient penalty if specified
        if self.config.gradient_penalty > 0:
            # Compute gradient penalty
            gradients = torch.autograd.grad(
                outputs=actor_loss,
                inputs=self.actor.parameters(),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )
            gradient_penalty = sum(
                (grad ** 2).sum() for grad in gradients if grad is not None
            )
            actor_loss = actor_loss + self.config.gradient_penalty * gradient_penalty
        
        # Optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
        self.actor_optimizer.step()
        
        self.metrics['actor_loss'].append(actor_loss.item())
        
        return actor_loss.item()
    
    def _update_temperature(self, states: torch.Tensor) -> float:
        """Update entropy temperature"""
        with torch.no_grad():
            actions, log_probs, _ = self.actor.sample(states)
        
        # Temperature loss
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        # Optimize
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update alpha value
        self.alpha = self.log_alpha.exp().item()
        
        self.metrics['alpha_loss'].append(alpha_loss.item())
        self.metrics['alpha'].append(self.alpha)
        
        return alpha_loss.item()
    
    def _soft_update_targets(self):
        """Soft update of target networks"""
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
        
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(
                self.config.tau * param.data + (1 - self.config.tau) * target_param.data
            )
    
    def save_model(self, path: str):
        """Sauvegarde le modèle"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'q1_target_state_dict': self.q1_target.state_dict(),
            'q2_target_state_dict': self.q2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'training_step': self.training_step,
            'episodes': self.episodes,
            'config': self.config
        }, path)
        
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Charge le modèle"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.q1_target.load_state_dict(checkpoint['q1_target_state_dict'])
        self.q2_target.load_state_dict(checkpoint['q2_target_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp().item()
        self.training_step = checkpoint['training_step']
        self.episodes = checkpoint['episodes']
        
        self.logger.info(f"Model loaded from {path}")
    
    def get_metrics(self) -> Dict[str, float]:
        """Retourne les métriques d'entraînement"""
        return {
            'avg_actor_loss': np.mean(self.metrics['actor_loss']) if self.metrics['actor_loss'] else 0,
            'avg_q1_loss': np.mean(self.metrics['q1_loss']) if self.metrics['q1_loss'] else 0,
            'avg_q2_loss': np.mean(self.metrics['q2_loss']) if self.metrics['q2_loss'] else 0,
            'avg_alpha_loss': np.mean(self.metrics['alpha_loss']) if self.metrics['alpha_loss'] else 0,
            'current_alpha': self.alpha,
            'avg_reward': np.mean(self.metrics['rewards']) if self.metrics['rewards'] else 0,
            'buffer_size': len(self.replay_buffer),
            'training_steps': self.training_step,
            'episodes': self.episodes
        }


# Fonctions utilitaires pour le trading
def create_trading_action_interpretation(
    raw_action: np.ndarray,
    current_position: float,
    current_price: float,
    config: SACConfig
) -> Dict[str, float]:
    """
    Interprète l'action SAC pour le trading
    Action[0]: Position size target (-1 to 1)
    Action[1]: Stop loss distance (0 to 0.1)
    Action[2]: Take profit distance (0 to 0.1)
    """
    position_target = float(raw_action[0])
    stop_loss_distance = float(raw_action[1])
    take_profit_distance = float(raw_action[2])
    
    # Calculer le changement de position requis
    position_change = position_target - current_position
    
    # Calculer les niveaux de prix
    if position_target > 0:  # Long position
        stop_loss_price = current_price * (1 - stop_loss_distance)
        take_profit_price = current_price * (1 + take_profit_distance)
    elif position_target < 0:  # Short position
        stop_loss_price = current_price * (1 + stop_loss_distance)
        take_profit_price = current_price * (1 - take_profit_distance)
    else:  # Neutral
        stop_loss_price = 0
        take_profit_price = 0
    
    return {
        'position_target': position_target,
        'position_change': position_change,
        'stop_loss_price': stop_loss_price,
        'take_profit_price': take_profit_price,
        'stop_loss_distance': stop_loss_distance,
        'take_profit_distance': take_profit_distance
    }


def compute_sac_trading_reward(
    pnl: float,
    position: float,
    volatility: float,
    sharpe_increment: float,
    max_drawdown: float,
    trade_frequency: float,
    config: SACConfig
) -> float:
    """Calcule la récompense SAC adaptée au trading"""
    # Base reward: PnL ajusté au risque
    risk_adjusted_pnl = pnl / (volatility + 1e-6)
    reward = risk_adjusted_pnl
    
    # Bonus pour amélioration du Sharpe
    if sharpe_increment > 0:
        reward += 0.5 * sharpe_increment
    
    # Pénalité pour drawdown excessif
    if max_drawdown > 0.1:  # 10%
        reward -= (max_drawdown - 0.1) * 2.0
    
    # Pénalité pour trading excessif
    if trade_frequency > 100:  # Plus de 100 trades par jour
        reward -= 0.01 * (trade_frequency - 100)
    
    # Reward shaping pour encourager des positions raisonnables
    position_penalty = 0.01 * (position ** 4)  # Pénalité plus forte pour positions extrêmes
    reward -= position_penalty
    
    return float(reward)


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    config = SACConfig(
        state_dim=50,
        action_dim=3,
        hidden_dims=[256, 256],
        use_layer_norm=True,
        learnable_temperature=True
    )
    
    # Créer l'agent
    agent = SACAgent(config)
    
    # Info
    total_params = (
        sum(p.numel() for p in agent.actor.parameters()) +
        sum(p.numel() for p in agent.q1.parameters()) +
        sum(p.numel() for p in agent.q2.parameters())
    )
    
    print(f"SAC Agent créé avec {total_params:,} paramètres")
    print(f"Architecture: LayerNorm={config.use_layer_norm}, SpectralNorm={config.use_spectral_norm}")
    print(f"Actions: {config.action_dim} continues (position, SL, TP)")
    print(f"Temperature: α={agent.alpha:.4f} (learnable={config.learnable_temperature})")
    print(f"Device: {config.device}")