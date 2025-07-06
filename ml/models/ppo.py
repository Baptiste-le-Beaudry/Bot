"""
Proximal Policy Optimization (PPO) - Algorithme de Policy Gradient pour le Trading
Implémentation optimisée pour la gestion de portefeuille avec actions continues

Caractéristiques principales:
- Architecture Actor-Critic avec attention
- Support des actions continues (position sizing)
- Clipping de ratio pour stabilité
- GAE (Generalized Advantage Estimation)
- Multi-head attention pour features complexes
- Support multi-assets simultané
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, Categorical
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import deque
import logging
from datetime import datetime, timezone
import math


@dataclass
class PPOConfig:
    """Configuration pour PPO"""
    # Architecture
    state_dim: int = 100
    continuous_action_dim: int = 1  # Position size (-1 to 1)
    discrete_action_dim: int = 3    # Buy, Hold, Sell
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    
    # Hyperparamètres PPO
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epsilon_clip: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    
    # Architecture avancée
    use_attention: bool = True
    num_attention_heads: int = 8
    use_lstm: bool = True
    lstm_hidden_size: int = 256
    lstm_num_layers: int = 2
    dropout_rate: float = 0.2
    
    # Entraînement
    batch_size: int = 64
    n_epochs: int = 10
    gradient_clip: float = 0.5
    max_trajectory_length: int = 2048
    
    # Trading spécifique
    max_position_size: float = 1.0
    position_penalty: float = 0.01  # Pénalité pour grandes positions
    transaction_cost: float = 0.001  # 0.1% par transaction
    
    # Normalisation
    use_batch_norm: bool = True
    use_layer_norm: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MultiHeadAttention(nn.Module):
    """Multi-head attention pour capturer les relations complexes entre features"""
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # Linear transformations
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Output projection
        output = self.out(context)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block avec attention et feed-forward"""
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual
        attn_output = self.attention(x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class ActorNetwork(nn.Module):
    """Actor network pour PPO avec support d'actions continues et discrètes"""
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # LSTM pour capturer les dépendances temporelles
        if config.use_lstm:
            self.lstm = nn.LSTM(
                input_size=config.hidden_dims[0],
                hidden_size=config.lstm_hidden_size,
                num_layers=config.lstm_num_layers,
                batch_first=True,
                dropout=config.dropout_rate if config.lstm_num_layers > 1 else 0
            )
            current_dim = config.lstm_hidden_size
        else:
            current_dim = config.hidden_dims[0]
        
        # Transformer blocks si attention activée
        if config.use_attention:
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(
                    current_dim,
                    config.num_attention_heads,
                    current_dim * 4,
                    config.dropout_rate
                ) for _ in range(2)
            ])
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for hidden_dim in config.hidden_dims[1:]:
            layer = [nn.Linear(current_dim, hidden_dim)]
            
            if config.use_batch_norm:
                layer.append(nn.BatchNorm1d(hidden_dim))
            elif config.use_layer_norm:
                layer.append(nn.LayerNorm(hidden_dim))
            
            layer.extend([nn.ReLU(), nn.Dropout(config.dropout_rate)])
            self.hidden_layers.append(nn.Sequential(*layer))
            current_dim = hidden_dim
        
        # Output heads
        # Action discrete (Buy/Hold/Sell)
        self.action_head = nn.Linear(current_dim, config.discrete_action_dim)
        
        # Position sizing (continuous)
        self.position_mean = nn.Linear(current_dim, config.continuous_action_dim)
        self.position_log_std = nn.Linear(current_dim, config.continuous_action_dim)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        # Small initial values for output layers
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        nn.init.orthogonal_(self.position_mean.weight, gain=0.01)
        nn.init.orthogonal_(self.position_log_std.weight, gain=0.01)
    
    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass
        Returns: action_logits, position_mean, position_log_std, hidden_state
        """
        # Feature extraction
        x = self.feature_extractor(state)
        
        # LSTM processing
        if self.config.use_lstm:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            
            if hidden is None:
                x, hidden = self.lstm(x)
            else:
                x, hidden = self.lstm(x, hidden)
            
            if x.shape[1] == 1:
                x = x.squeeze(1)
            else:
                x = x[:, -1, :]  # Take last output
        
        # Attention blocks
        if self.config.use_attention:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            
            for transformer in self.transformer_blocks:
                x = transformer(x)
            
            x = x.squeeze(1) if x.shape[1] == 1 else x[:, -1, :]
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Output heads
        action_logits = self.action_head(x)
        position_mean = self.position_mean(x)
        position_log_std = self.position_log_std(x)
        
        # Clamp log_std for numerical stability
        position_log_std = torch.clamp(position_log_std, min=-20, max=2)
        
        return action_logits, position_mean, position_log_std, hidden


class CriticNetwork(nn.Module):
    """Critic network pour estimer la value function"""
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        
        # Shared architecture with actor (mais paramètres indépendants)
        self.feature_extractor = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        if config.use_lstm:
            self.lstm = nn.LSTM(
                input_size=config.hidden_dims[0],
                hidden_size=config.lstm_hidden_size,
                num_layers=config.lstm_num_layers,
                batch_first=True,
                dropout=config.dropout_rate if config.lstm_num_layers > 1 else 0
            )
            current_dim = config.lstm_hidden_size
        else:
            current_dim = config.hidden_dims[0]
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        for hidden_dim in config.hidden_dims[1:]:
            layer = [nn.Linear(current_dim, hidden_dim)]
            
            if config.use_layer_norm:
                layer.append(nn.LayerNorm(hidden_dim))
            
            layer.extend([nn.ReLU(), nn.Dropout(config.dropout_rate)])
            self.hidden_layers.append(nn.Sequential(*layer))
            current_dim = hidden_dim
        
        # Value head
        self.value_head = nn.Linear(current_dim, 1)
        
        # Initialize
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
    
    def forward(
        self,
        state: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass returning value estimate"""
        x = self.feature_extractor(state)
        
        if self.config.use_lstm:
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            
            if hidden is None:
                x, hidden = self.lstm(x)
            else:
                x, hidden = self.lstm(x, hidden)
            
            x = x.squeeze(1) if x.shape[1] == 1 else x[:, -1, :]
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        value = self.value_head(x)
        
        return value, hidden


class PPOMemory:
    """Memory buffer pour stocker les trajectoires"""
    def __init__(self):
        self.states = []
        self.actions = []
        self.positions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.hidden_states_actor = []
        self.hidden_states_critic = []
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        position: float,
        logprob: float,
        reward: float,
        value: float,
        done: bool,
        hidden_actor: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        hidden_critic: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        """Ajoute une transition"""
        self.states.append(state)
        self.actions.append(action)
        self.positions.append(position)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.hidden_states_actor.append(hidden_actor)
        self.hidden_states_critic.append(hidden_critic)
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Retourne toutes les données comme tensors"""
        return {
            'states': torch.FloatTensor(np.array(self.states)),
            'actions': torch.LongTensor(self.actions),
            'positions': torch.FloatTensor(self.positions),
            'logprobs': torch.FloatTensor(self.logprobs),
            'rewards': torch.FloatTensor(self.rewards),
            'values': torch.FloatTensor(self.values),
            'dones': torch.FloatTensor(self.dones)
        }
    
    def clear(self):
        """Vide la mémoire"""
        self.states.clear()
        self.actions.clear()
        self.positions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
        self.hidden_states_actor.clear()
        self.hidden_states_critic.clear()
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """Agent PPO complet pour le trading"""
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Networks
        self.actor = ActorNetwork(config).to(self.device)
        self.critic = CriticNetwork(config).to(self.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=config.learning_rate_actor,
            eps=1e-5
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=config.learning_rate_critic,
            eps=1e-5
        )
        
        # Learning rate schedulers
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=10000, eta_min=1e-5
        )
        self.critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, T_max=10000, eta_min=1e-5
        )
        
        # Memory
        self.memory = PPOMemory()
        
        # Hidden states
        self.hidden_actor = None
        self.hidden_critic = None
        
        # Training metrics
        self.training_step = 0
        self.episodes = 0
        self.metrics = {
            'actor_loss': deque(maxlen=1000),
            'critic_loss': deque(maxlen=1000),
            'entropy': deque(maxlen=1000),
            'rewards': deque(maxlen=1000)
        }
        
        self.logger = logging.getLogger(__name__)
    
    def select_action(
        self,
        state: np.ndarray,
        training: bool = True
    ) -> Tuple[int, float, float, Dict[str, Any]]:
        """
        Sélectionne une action et une taille de position
        Returns: action, position, log_prob, info
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Actor forward pass
            action_logits, pos_mean, pos_log_std, self.hidden_actor = self.actor(
                state_tensor, self.hidden_actor
            )
            
            # Critic forward pass pour la value
            value, self.hidden_critic = self.critic(state_tensor, self.hidden_critic)
            
            # Sample discrete action
            if training:
                action_probs = F.softmax(action_logits, dim=-1)
                action_dist = Categorical(action_probs)
                action = action_dist.sample()
                action_logprob = action_dist.log_prob(action)
            else:
                # Greedy action for evaluation
                action = action_logits.argmax(dim=-1)
                action_probs = F.softmax(action_logits, dim=-1)
                action_logprob = torch.log(action_probs[0, action])
            
            # Sample continuous position size
            if training:
                pos_std = torch.exp(pos_log_std)
                pos_dist = Normal(pos_mean, pos_std)
                position = pos_dist.sample()
                position_logprob = pos_dist.log_prob(position).sum(dim=-1)
            else:
                # Use mean for evaluation
                position = pos_mean
                position_logprob = torch.tensor(0.0)
            
            # Clamp position to valid range
            position = torch.clamp(position, -self.config.max_position_size, self.config.max_position_size)
            
            # Total log probability
            total_logprob = action_logprob + position_logprob
            
            info = {
                'value': value.item(),
                'action_probs': action_probs.cpu().numpy(),
                'position_mean': pos_mean.item(),
                'position_std': torch.exp(pos_log_std).item() if training else 0,
                'entropy': -action_logprob.item()
            }
            
            return (
                action.item(),
                position.item(),
                total_logprob.item(),
                info
            )
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        position: float,
        logprob: float,
        reward: float,
        value: float,
        done: bool
    ):
        """Stocke une transition dans la mémoire"""
        self.memory.add(
            state, action, position, logprob, reward, value, done,
            self.hidden_actor, self.hidden_critic
        )
        self.metrics['rewards'].append(reward)
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards).to(self.device)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.config.gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_advantage
            last_advantage = advantages[t]
        
        returns = advantages + values
        
        return advantages, returns
    
    def train(self, next_state: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Effectue l'entraînement PPO"""
        if len(self.memory) < self.config.batch_size:
            return {}
        
        # Get batch
        batch = self.memory.get_batch()
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        positions = batch['positions'].to(self.device)
        old_logprobs = batch['logprobs'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        values = batch['values'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # Compute next value if provided
        with torch.no_grad():
            if next_state is not None:
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                next_value, _ = self.critic(next_state_tensor)
                next_value = next_value.squeeze()
            else:
                next_value = torch.tensor(0.0).to(self.device)
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        for epoch in range(self.config.n_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_positions = positions[batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Actor loss
                action_logits, pos_mean, pos_log_std, _ = self.actor(batch_states)
                
                # Action probabilities
                action_probs = F.softmax(action_logits, dim=-1)
                action_dist = Categorical(action_probs)
                action_logprobs = action_dist.log_prob(batch_actions)
                
                # Position probabilities
                pos_std = torch.exp(pos_log_std)
                pos_dist = Normal(pos_mean, pos_std)
                position_logprobs = pos_dist.log_prob(batch_positions.unsqueeze(-1)).sum(dim=-1)
                
                # Total log probabilities
                logprobs = action_logprobs + position_logprobs
                
                # Ratio for PPO
                ratio = torch.exp(logprobs - batch_old_logprobs)
                
                # Clipped surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.epsilon_clip, 1 + self.config.epsilon_clip) * batch_advantages
                
                # Actor loss
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy bonus
                action_entropy = action_dist.entropy().mean()
                position_entropy = pos_dist.entropy().sum(dim=-1).mean()
                total_entropy_batch = action_entropy + position_entropy
                
                # Position penalty (discourage extreme positions)
                position_penalty = (batch_positions.abs() ** 2).mean() * self.config.position_penalty
                
                # Total actor loss
                actor_loss = actor_loss - self.config.entropy_coef * total_entropy_batch + position_penalty
                
                # Critic loss
                values, _ = self.critic(batch_states)
                critic_loss = F.mse_loss(values.squeeze(), batch_returns)
                
                # Backward pass
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
                self.actor_optimizer.step()
                
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip)
                self.critic_optimizer.step()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += total_entropy_batch.item()
        
        # Update schedulers
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        # Clear memory
        self.memory.clear()
        
        # Update metrics
        avg_actor_loss = total_actor_loss / (self.config.n_epochs * (len(states) // self.config.batch_size))
        avg_critic_loss = total_critic_loss / (self.config.n_epochs * (len(states) // self.config.batch_size))
        avg_entropy = total_entropy / (self.config.n_epochs * (len(states) // self.config.batch_size))
        
        self.metrics['actor_loss'].append(avg_actor_loss)
        self.metrics['critic_loss'].append(avg_critic_loss)
        self.metrics['entropy'].append(avg_entropy)
        
        self.training_step += 1
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'entropy': avg_entropy,
            'average_reward': np.mean(rewards.cpu().numpy()),
            'average_value': np.mean(values.cpu().numpy()),
            'advantage_mean': advantages.mean().item(),
            'advantage_std': advantages.std().item()
        }
    
    def reset_episode(self):
        """Reset pour un nouvel épisode"""
        self.hidden_actor = None
        self.hidden_critic = None
        self.episodes += 1
    
    def save_model(self, path: str):
        """Sauvegarde le modèle"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'actor_scheduler_state_dict': self.actor_scheduler.state_dict(),
            'critic_scheduler_state_dict': self.critic_scheduler.state_dict(),
            'training_step': self.training_step,
            'episodes': self.episodes,
            'config': self.config
        }, path)
        
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Charge le modèle"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state_dict'])
        self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state_dict'])
        
        self.training_step = checkpoint['training_step']
        self.episodes = checkpoint['episodes']
        
        self.logger.info(f"Model loaded from {path}")
    
    def get_metrics(self) -> Dict[str, float]:
        """Retourne les métriques d'entraînement"""
        return {
            'avg_actor_loss': np.mean(self.metrics['actor_loss']) if self.metrics['actor_loss'] else 0,
            'avg_critic_loss': np.mean(self.metrics['critic_loss']) if self.metrics['critic_loss'] else 0,
            'avg_entropy': np.mean(self.metrics['entropy']) if self.metrics['entropy'] else 0,
            'avg_reward': np.mean(self.metrics['rewards']) if self.metrics['rewards'] else 0,
            'training_steps': self.training_step,
            'episodes': self.episodes
        }


# Fonctions utilitaires
def compute_trading_reward(
    pnl: float,
    position_change: float,
    current_position: float,
    sharpe_ratio: float,
    config: PPOConfig
) -> float:
    """Calcule la récompense adaptée au trading"""
    # Base reward: PnL
    reward = pnl
    
    # Transaction cost penalty
    if position_change != 0:
        reward -= abs(position_change) * config.transaction_cost
    
    # Sharpe ratio bonus
    if sharpe_ratio > 1.5:
        reward += 0.1 * (sharpe_ratio - 1.5)
    
    # Position size penalty (éviter les positions extrêmes)
    reward -= config.position_penalty * (current_position ** 2)
    
    return reward


def create_state_representation(
    market_features: Dict[str, float],
    portfolio_state: Dict[str, float],
    technical_indicators: Dict[str, float],
    market_microstructure: Dict[str, float]
) -> np.ndarray:
    """Crée la représentation d'état pour PPO"""
    state = []
    
    # Market features
    state.extend([
        market_features.get('price', 0),
        market_features.get('volume', 0),
        market_features.get('volatility', 0),
        market_features.get('spread', 0)
    ])
    
    # Portfolio state
    state.extend([
        portfolio_state.get('position', 0),
        portfolio_state.get('unrealized_pnl', 0),
        portfolio_state.get('realized_pnl', 0),
        portfolio_state.get('cash_balance', 0)
    ])
    
    # Technical indicators
    state.extend([
        technical_indicators.get('rsi', 50) / 100,
        technical_indicators.get('macd', 0),
        technical_indicators.get('bb_position', 0.5),
        technical_indicators.get('atr', 0)
    ])
    
    # Market microstructure
    state.extend([
        market_microstructure.get('bid_ask_imbalance', 0),
        market_microstructure.get('order_flow_imbalance', 0),
        market_microstructure.get('trade_intensity', 0)
    ])
    
    return np.array(state, dtype=np.float32)


# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
    config = PPOConfig(
        state_dim=50,
        continuous_action_dim=1,
        discrete_action_dim=3,
        hidden_dims=[256, 128],
        use_attention=True,
        use_lstm=True
    )
    
    # Créer l'agent
    agent = PPOAgent(config)
    
    # Info
    total_params = sum(p.numel() for p in agent.actor.parameters()) + \
                   sum(p.numel() for p in agent.critic.parameters())
    
    print(f"PPO Agent créé avec {total_params:,} paramètres")
    print(f"Architecture: Attention={config.use_attention}, LSTM={config.use_lstm}")
    print(f"Actions: {config.discrete_action_dim} discrètes + {config.continuous_action_dim} continues")
    print(f"Device: {config.device}")