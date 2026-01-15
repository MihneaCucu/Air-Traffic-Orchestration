import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random


class ReplayBuffer:

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
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

    def __init__(self, state_dim, action_dim, hidden_dims=None):
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
        return self.network(state)


class PolicyNetwork(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dims=None):
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
        logits = self.network(state)
        probs = F.softmax(logits, dim=-1)
        return probs

    def get_action(self, state, deterministic=False):
        probs = self.forward(state)

        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()

        return action, probs


class DiscreteSAC:

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
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = device

        if hidden_dims is None:
            hidden_dims = [256, 256]

        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dims).to(device)

        self.q1 = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims).to(device)

        self.q1_target = QNetwork(state_dim, action_dim, hidden_dims).to(device)
        self.q2_target = QNetwork(state_dim, action_dim, hidden_dims).to(device)

        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)

        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            self.target_entropy = -np.log(1.0 / action_dim) * 0.98
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = alpha

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _ = self.policy.get_action(state, deterministic)

        return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:
            return {}

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_probs = self.policy(next_states)

            next_q1 = self.q1_target(next_states)
            next_q2 = self.q2_target(next_states)
            next_q = torch.min(next_q1, next_q2)

            next_v = (next_probs * (next_q - self.alpha * torch.log(next_probs + 1e-8))).sum(dim=1, keepdim=True)

            target_q = rewards + (1 - dones) * self.gamma * next_v

        current_q1 = self.q1(states).gather(1, actions.unsqueeze(1))
        current_q2 = self.q2(states).gather(1, actions.unsqueeze(1))

        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        probs = self.policy(states)

        q1 = self.q1(states)
        q2 = self.q2(states)
        q = torch.min(q1, q2)

        inside_term = self.alpha * torch.log(probs + 1e-8) - q
        policy_loss = (probs * inside_term).sum(dim=1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        entropy = 0.0
        if self.auto_entropy_tuning:
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha.item() if self.auto_entropy_tuning else self.alpha,
            'entropy': entropy.item() if self.auto_entropy_tuning else 0.0
        }

    def save(self, filepath):
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
        print(f" Model salvat: {filepath}")

    def load(self, filepath):
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

        print(f" Model încărcat: {filepath}")
    def predict(self, obs, deterministic=True):
        action = self.select_action(obs, deterministic=deterministic)
        return action, None
    def learn(self, total_timesteps, env=None):
        if env is None:
            raise ValueError("SAC.learn() requires env parameter")
        batch_size = 256
        buffer_size = 100000
        learning_starts = 1000
        replay_buffer = ReplayBuffer(capacity=buffer_size)
        timestep = 0
        episode = 0
        state, _ = env.reset()
        episode_reward = 0
        print(f"SAC Training: 0/{total_timesteps} steps...")
        while timestep < total_timesteps:
            if timestep < learning_starts:
                action = env.action_space.sample()
            else:
                action = self.select_action(state, deterministic=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            timestep += 1
            if timestep >= learning_starts and len(replay_buffer) >= batch_size:
                if timestep % 4 == 0:
                    self.update(replay_buffer, batch_size=batch_size)
            if done:
                episode += 1
                if episode % 10 == 0:
                    print(f"SAC: Step {timestep}/{total_timesteps} | Episode {episode} | Reward: {episode_reward:.2f}")
                state, _ = env.reset()
                episode_reward = 0
        print(f"SAC Training complete: {total_timesteps} steps")
