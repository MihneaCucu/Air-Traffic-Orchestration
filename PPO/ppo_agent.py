import sys
sys.path.insert(0, '..')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import os
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=64):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_size, action_dim)
        self.critic = nn.Linear(hidden_size, 1)
    def forward(self, state):
        shared_features = self.shared(state)
        action_probs = torch.softmax(self.actor(shared_features), dim=-1)
        state_value = self.critic(shared_features)
        return action_probs, state_value
    def get_action(self, state, deterministic=False):
        action_probs, state_value = self.forward(state)
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
            return action.item(), None, None
        dist = Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.item(), action_log_prob, state_value


class PPOAgent:
    def __init__(
        self,
        env,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=1,
        batch_size=16,
        device="cpu"
    ):
        self.env = env
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.policy = ActorCritic(self.state_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.reset_storage()
    def reset_storage(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    def predict(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.policy.get_action(state_tensor, deterministic)
        return action, None
    def store_transition(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    def compute_gae(self, next_value):
        advantages = []
        gae = 0
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = self.values[t + 1]
            delta = self.rewards[t] + self.gamma * next_value_t * (1 - self.dones[t]) - self.values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(self.values).to(self.device)
        return advantages, returns
    def update_policy(self, advantages, returns):
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(self.update_epochs):
            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns)
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()
        return actor_loss.item(), critic_loss.item(), entropy.item()
    def learn(self, total_timesteps=500000, n_steps=64, log_interval=10):
        state, _ = self.env.reset()
        episode_reward = 0
        episode_count = 0
        rewards_history = deque(maxlen=100)
        print(f"Training PPO for {total_timesteps} timesteps...")
        timestep = 0
        update_count = 0
        while timestep < total_timesteps:
            self.reset_storage()
            for _ in range(n_steps):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, log_prob, value = self.policy.get_action(state_tensor)
                next_state, reward, done, truncated, _ = self.env.step(action)
                episode_reward += reward
                self.store_transition(
                    state, action, log_prob.item() if log_prob is not None else 0,
                    reward, value.item() if value is not None else 0, done
                )
                state = next_state
                timestep += 1
                if done or truncated:
                    rewards_history.append(episode_reward)
                    episode_count += 1
                    episode_reward = 0
                    state, _ = self.env.reset()
                if timestep >= total_timesteps:
                    break
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                _, next_value = self.policy(next_state_tensor)
                next_value = next_value.item()
            advantages, returns = self.compute_gae(next_value)
            actor_loss, critic_loss, entropy = self.update_policy(advantages, returns)
            update_count += 1
            if update_count % log_interval == 0:
                avg_reward = np.mean(rewards_history) if rewards_history else 0
                print(f"Step: {timestep}/{total_timesteps} | "
                      f"Episodes: {episode_count} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Actor Loss: {actor_loss:.4f} | "
                      f"Critic Loss: {critic_loss:.4f} | "
                      f"Entropy: {entropy:.4f}")
        print("Training completed!")
    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"Model saved to {path}")
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
