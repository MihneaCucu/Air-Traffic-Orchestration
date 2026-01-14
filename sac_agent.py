"""
SAC (Soft Actor-Critic) Agent pentru Discrete Action Spaces
============================================================

Implementare custom de SAC adaptată pentru acțiuni discrete (ex: Discrete(3))
Bazat pe "Soft Actor-Critic for Discrete Action Settings"

Componente:
-----------
1. ReplayBuffer - stochează experiențe pentru off-policy learning
2. QNetwork - estimează Q-values pentru fiecare acțiune
3. PolicyNetwork - alege acțiuni bazat pe probabilități
4. DiscreteSAC - agentul complet cu training loop

Avantaje SAC:
-------------
- Explorare mai bună prin maximizarea entropiei
- Off-policy learning (mai eficient cu datele)
- Stable training prin soft updates
- Automatic temperature tuning
"""

import os
# CRITICAL: Fix PyTorch threading deadlock on macOS Python 3.13
# Must be set BEFORE importing torch
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
# Configure torch to use single thread
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """
    Experience Replay Buffer

    Stochează tranziții (s, a, r, s', done) pentru antrenament off-policy.
    Permite agentului să învețe din experiențe trecute, nu doar cele recente.
    """

    def __init__(self, capacity=100000):
        """
        Args:
            capacity: Numărul maxim de tranziții stocate
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Adaugă o tranziție în buffer"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample un batch random de tranziții

        Returns:
            Tuple de numpy arrays: (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Q-Network (Critic)

    Estimează Q(s,a) pentru toate acțiunile posibile.
    Output: vector de dimensiune [batch_size, num_actions]
    """

    def __init__(self, state_dim, action_dim, hidden_dims=None):
        """
        Args:
            state_dim: Dimensiunea state space
            action_dim: Numărul de acțiuni discrete
            hidden_dims: Lista cu dimensiunile layer-elor ascunse
        """
        super(QNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        Forward pass

        Args:
            state: Tensor [batch_size, state_dim]
        Returns:
            Q-values: Tensor [batch_size, action_dim]
        """
        return self.network(state)


class PolicyNetwork(nn.Module):
    """
    Policy Network (Actor)

    Output: probabilități pentru fiecare acțiune (discrete distribution)
    Folosește Softmax pentru a obține probabilități valide
    """

    def __init__(self, state_dim, action_dim, hidden_dims=None):
        """
        Args:
            state_dim: Dimensiunea state space
            action_dim: Numărul de acțiuni discrete
            hidden_dims: Lista cu dimensiunile layer-elor ascunse
        """
        super(PolicyNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        Forward pass - returnează probabilități

        Args:
            state: Tensor [batch_size, state_dim]
        Returns:
            probs: Tensor [batch_size, action_dim] - probabilități
        """
        logits = self.network(state)
        probs = F.softmax(logits, dim=-1)
        return probs

    def get_action(self, state, deterministic=False):
        """
        Sample o acțiune din policy

        Args:
            state: Tensor [batch_size, state_dim]
            deterministic: Dacă True, alege acțiunea cu prob maximă (greedy)

        Returns:
            action: Tensor [batch_size]
            probs: Tensor [batch_size, action_dim]
        """
        probs = self.forward(state)

        if deterministic:
            # Greedy: alege acțiunea cu probabilitatea maximă
            action = torch.argmax(probs, dim=-1)
        else:
            # Stochastic: sample din distribuție
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        return action, probs


class DiscreteSAC:
    """
    Soft Actor-Critic pentru Discrete Action Spaces

    Algoritm:
    ---------
    1. Sample batch din replay buffer
    2. Update Q-networks (critics) cu TD error
    3. Update policy (actor) pentru a maximiza Q - α*log(π)
    4. Update temperature α pentru entropy tuning
    5. Soft update target networks

    Features:
    ---------
    - Double Q-learning (2 Q-networks) pentru stabilitate
    - Automatic entropy tuning (alpha adjustment)
    - Target networks cu soft updates (Polyak averaging)
    """

    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        auto_entropy_tuning=True,
        hidden_dims=None,
        device='cpu'
    ):
        """
        Args:
            state_dim: Dimensiunea observațiilor
            action_dim: Numărul de acțiuni discrete
            lr: Learning rate
            gamma: Discount factor pentru future rewards
            tau: Soft update coefficient pentru target networks
            alpha: Entropy temperature (dacă nu e auto-tuned)
            auto_entropy_tuning: Dacă True, alpha se ajustează automat
            hidden_dims: Dimensiuni layer-e ascunse
            device: 'cpu' sau 'cuda'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        if hidden_dims is None:
            hidden_dims = [256, 256]

        # ===== NETWORKS =====
        # Policy network (Actor)
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims).to(device)

        # Q-networks (Critics) - folosim 2 pentru stabilitate
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims).to(device)

        # Target Q-networks - pentru stable learning
        self.q1_target = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dims).to(device)

        # Copiază parametrii inițiali
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # ===== OPTIMIZERS =====
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        # ===== ENTROPY TEMPERATURE =====
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            # Target entropy = -log(1/|A|) * 0.98 (puțin sub uniform)
            self.target_entropy = -np.log(1.0 / action_dim) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

    def select_action(self, state, deterministic=False):
        """
        Alege o acțiune pentru un singur state

        Args:
            state: numpy array [state_dim]
            deterministic: Dacă True, alege greedy

        Returns:
            action: int - acțiunea selectată
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _ = self.policy.get_action(state, deterministic)

        return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        """
        Un pas de antrenament (update toate networks)

        Args:
            replay_buffer: ReplayBuffer cu experiențe
            batch_size: Dimensiunea batch-ului

        Returns:
            dict cu losses pentru logging
        """
        if len(replay_buffer) < batch_size:
            return {}

        # ===== SAMPLE BATCH =====
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ===== UPDATE Q-NETWORKS =====
        with torch.no_grad():
            # Probabilități pentru next state
            next_probs = self.policy(next_states)

            # Q-values pentru next state (din target networks)
            next_q1 = self.q1_target(next_states)
            next_q2 = self.q2_target(next_states)
            next_q = torch.min(next_q1, next_q2)  # Double Q-learning trick

            # V(s') = E_π[Q(s',a') - α*log(π(a'|s'))]
            # Soft value function (include entropy bonus)
            next_v = (next_probs * (next_q - self.alpha * torch.log(next_probs + 1e-8))).sum(dim=1, keepdim=True)

            # Target Q-value: r + γ * (1 - done) * V(s')
            target_q = rewards + (1 - dones) * self.gamma * next_v

        # Current Q-values pentru acțiunea luată
        current_q1 = self.q1(states).gather(1, actions.unsqueeze(1))
        current_q2 = self.q2(states).gather(1, actions.unsqueeze(1))

        # Q-network losses (MSE)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        # Update Q1
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        # Update Q2
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # ===== UPDATE POLICY =====
        probs = self.policy(states)

        # Q-values pentru state curent
        q1 = self.q1(states)
        q2 = self.q2(states)
        q = torch.min(q1, q2)

        # Policy loss: maximizează E_π[Q(s,a) - α*log(π(a|s))]
        # Echivalent cu minimizarea: E_π[α*log(π(a|s)) - Q(s,a)]
        inside_term = self.alpha * torch.log(probs + 1e-8) - q
        policy_loss = (probs * inside_term).sum(dim=1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ===== UPDATE TEMPERATURE (ALPHA) =====
        entropy = 0.0
        if self.auto_entropy_tuning:
            # Entropy curent al policy-ului
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

            # Alpha loss: ajustează α pentru a match target entropy
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        # ===== SOFT UPDATE TARGET NETWORKS =====
        # θ_target = τ * θ + (1 - τ) * θ_target (Polyak averaging)
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Return metrics pentru logging
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha.item() if self.auto_entropy_tuning else self.alpha,
            'entropy': entropy.item() if self.auto_entropy_tuning else 0.0
        }

    def save(self, filepath):
        """Salvează modelul"""
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q1_optimizer': self.q1_optimizer.state_dict(),
            'q2_optimizer': self.q2_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_entropy_tuning else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_entropy_tuning else None,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
        }, filepath)
        print(f"✓ Model salvat: {filepath}")

    def load(self, filepath):
        """Încarcă modelul"""
        # PyTorch 2.6+ requires weights_only=False for loading optimizer states
        # This is safe for our own saved models
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.policy.load_state_dict(checkpoint['policy'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])

        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer'])

        if self.auto_entropy_tuning and checkpoint['log_alpha'] is not None:
            self.log_alpha.data = checkpoint['log_alpha'].data
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.alpha = self.log_alpha.exp()

        print(f"✓ Model încărcat: {filepath}")

